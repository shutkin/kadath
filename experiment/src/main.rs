mod signal_optimization;
mod math_utils;
mod simulation;
mod stats_collector;
mod deal;
mod signals;

use crate::deal::DealDirection;
use crate::signal_optimization::{CostFunctionImpl, SignalParams, calibrate_params, find_threshold};
use crate::simulation::run_simulation;
use crate::stats_collector::future_spreads_correlations;
use chrono::{DateTime, Datelike, TimeDelta, Utc};
use log::{info, warn};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::math_util::standard_deviation;
use model::{order_book_cache, DataSource, Instrument, OrderBook};
use simplelog::{LevelFilter, SimpleLogger};
use std::thread;
use db::Db;

pub const IMBALANCE_LEVELS: [usize; 5] = [3, 5, 10, 20, 50];
pub const DEFAULT_HOLD_MS: u16 = 1100;
const STD_DEVIATION_WINDOW_SECONDS: i64 = 1050;
const TREND_TAU_SECONDS: f64 = 1.0 * 60.0;

pub fn commission(revenue: f64, cost: f64) -> f64 {
    // T-Invest
    //(revenue + cost) * 0.015 / 100.0

    // Alor
    5.0
}

fn find_instrument_id(instruments: &[Instrument], ticker: &str) -> Option<i32> {
    instruments
        .iter()
        .find(|instrument| instrument.ticker == ticker)
        .and_then(|instrument| instrument.id)
}

async fn get_order_books(tickers: &[&str], ids: &[i32], diapason: TimeDiapason, db: Option<&Db>)
                         -> Result<(Vec<OrderBook>, Vec<OrderBook>), CommonError> {
    if let (Some(order_books1), Some(order_books2)) =
        (order_book_cache::read(tickers[0], diapason, true)?,
         order_book_cache::read(tickers[1], diapason, true)?) {
        Ok((order_books1, order_books2))
    } else {
        let (order_books1, order_books2) =
            (db.unwrap().get_order_books(ids[0], diapason).await?,
             db.unwrap().get_order_books(ids[1], diapason).await?);
        order_book_cache::write(tickers[0], diapason, &order_books1, true)?;
        order_book_cache::write(tickers[1], diapason, &order_books2, true)?;
        Ok((order_books1, order_books2))
    }
}

#[tokio::main]
async fn __main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = db::Db::new(&db_url).await?;

    let tickers = ["GLU6", "GLZ6", "GLH7", "GLM7"];

    let all_instruments = db.get_instruments(DataSource::Alor).await?;
    if let (Some(inst1_id), Some(inst2_id)) = (
        find_instrument_id(&all_instruments, tickers[0]),
        find_instrument_id(&all_instruments, tickers[1]),
    ) {
        let mut train_events = Vec::<MarketEvent>::new();
        let mut diapasons = Vec::new();
        let mut total_profit = 0.0;

        let mut diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-09-07T05:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-09-07T20:00:00Z")?.to_utc(),
        );
        for day in 7..16 {
            if !matches!(diapason.from.weekday().number_from_monday(), 6 | 7) {
                info!("DAY {day}");
                diapasons.push(diapason);

                let (order_books1, order_books2) = get_order_books(&tickers, &[inst1_id, inst2_id], diapason, Some(&db)).await?;
                let (mut all_values1, mut all_values2) = (Vec::new(), Vec::new());
                convert_values(&order_books1, &order_books2, &mut all_values1, &mut all_values2);
                info!("Calculate std deviations with window {STD_DEVIATION_WINDOW_SECONDS}");
                calculate_std_deviations(&mut all_values2, STD_DEVIATION_WINDOW_SECONDS);
                calculate_std_deviations(&mut all_values1, STD_DEVIATION_WINDOW_SECONDS);
                info!("Calculate trends");
                calculate_trend(&mut all_values1);
                calculate_trend(&mut all_values2);

                //let trades1 = db.get_trades(inst1_id, diapason).await?;
                //let trades2 = db.get_trades(inst1_id, diapason).await?;

                let day_events = merge_events(&all_values1, &all_values2);

                if !train_events.is_empty() {
                    if diapasons.len() > 2 {
                        let train_start = diapasons[diapasons.len() - 3].from;
                        train_events.retain(|event| event.event_datetime() >= train_start);
                    }

                    info!("Train on {} events", train_events.len());
                    let events_clone1 = train_events.clone();
                    let events_clone2 = train_events.clone();
                    let handle_up = thread::spawn(move || {
                        let mut params = calibrate_params(&events_clone1, CostFunctionImpl::HuberLoss, DealDirection::Sell1Buy2, DEFAULT_HOLD_MS);
                        if let Some(t) = find_threshold(&events_clone1, params.up.unwrap(), DealDirection::Sell1Buy2)
                            && let Some(p) = params.up.as_mut() {
                            p.threshold = t;
                        }
                        Some(params)
                    });
                    let handle_down = thread::spawn(move || {
                        let mut params = calibrate_params(&events_clone2, CostFunctionImpl::HuberLoss, DealDirection::Buy1Sell2, DEFAULT_HOLD_MS);
                        if let Some(t) = find_threshold(&events_clone2, params.down.unwrap(), DealDirection::Buy1Sell2)
                            && let Some(p) = params.down.as_mut() {
                            p.threshold = t;
                        }
                        Some(params)
                    });
                    let up = handle_up.join().unwrap();
                    let down = handle_down.join().unwrap();
                    if let Some(up) = up && let Some(down) = down {
                        let params = SignalParams {
                            hold_ms: up.hold_ms,
                            up: up.up,
                            down: down.down,
                        };
                        info!("{params:?}");
                        let result = run_simulation(&day_events, &params, false);
                        info!(
                            "{}: deals {}, income {}, outcome {}, commission {}, profit {}",
                            diapason.from.date_naive(),
                            result.win + result.loss,
                            result.income,
                            result.outcome,
                            result.commission,
                            result.income - result.outcome - result.commission,
                        );
                        total_profit += result.income - result.outcome - result.commission;
                    }
                }

                train_events.extend(day_events);
            }

            diapason.from += TimeDelta::days(1);
            diapason.to += TimeDelta::days(1);
        }
        warn!("Total profit: {total_profit}");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = db::Db::new(&db_url).await?;

    let tickers = ["GLU6", "GLZ6", "GLH7", "GLM7"];
    //let mut diapason = TimeDiapason::new(
    //    DateTime::parse_from_rfc3339("2026-09-03T05:00:00Z")?.to_utc(),
    //    DateTime::parse_from_rfc3339("2026-09-03T20:00:00Z")?.to_utc(),
    //);
    //let end = DateTime::parse_from_rfc3339("2026-09-18T00:00:00Z")?.to_utc();

    //let tickers = ["GLZ6", "GLH7", "GLM7"];
    let mut diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-09-10T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-09-10T20:00:00Z")?.to_utc(),
    );
    let end = DateTime::parse_from_rfc3339("2026-09-11T00:00:00Z")?.to_utc();

    let all_instruments = db.get_instruments(DataSource::Alor).await?;
    if let (Some(inst1_id), Some(inst2_id)) = (
        find_instrument_id(&all_instruments, tickers[0]),
        find_instrument_id(&all_instruments, tickers[1]),
    ) {
        while diapason.to < end {
            if !matches!(diapason.from.weekday().number_from_monday(), 6 | 7) {
                info!("DAY {}", diapason.from.date_naive());
                let (order_books1, order_books2) = get_order_books(&tickers, &[inst1_id, inst2_id], diapason, Some(&db)).await?;
                let (mut all_values1, mut all_values2) = (Vec::new(), Vec::new());
                convert_values(&order_books1, &order_books2, &mut all_values1, &mut all_values2);

                info!("Calculate std deviations");
                calculate_std_deviations(&mut all_values1, STD_DEVIATION_WINDOW_SECONDS);
                calculate_std_deviations(&mut all_values2, STD_DEVIATION_WINDOW_SECONDS);

                let events = merge_events(&all_values1, &all_values2);
                info!("{}:\n{}", diapason.from.date_naive(), future_spreads_correlations(&events));
            }

            diapason.from += TimeDelta::days(1);
            diapason.to += TimeDelta::days(1);
        }
    }
    Ok(())
}

#[tokio::main]
async fn _main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = db::Db::new(&db_url).await?;

    let tickers = ["GLU6", "GLZ6", "GLH7", "GLM7"];
    let train_diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-09-03T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-09-03T20:00:00Z")?.to_utc(),
    );
    let test_diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-09-04T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-09-04T20:00:00Z")?.to_utc(),
    );
    let functions = [CostFunctionImpl::TradingSimulation];

    let all_instruments = db.get_instruments(DataSource::Alor).await?;
    if let (Some(inst1_id), Some(inst2_id)) = (
        find_instrument_id(&all_instruments, tickers[0]),
        find_instrument_id(&all_instruments, tickers[1]),
    ) {
        info!("Train on {train_diapason:?}");
        let (mut all_values1, mut all_values2) = (Vec::new(), Vec::new());

        let (order_books1, order_books2) = get_order_books(&tickers, &[inst1_id, inst2_id], train_diapason, Some(&db)).await?;
        convert_values(&order_books1, &order_books2, &mut all_values1, &mut all_values2);
        info!("Calculate std deviations with window {STD_DEVIATION_WINDOW_SECONDS}");
        calculate_std_deviations(&mut all_values1, STD_DEVIATION_WINDOW_SECONDS);
        calculate_std_deviations(&mut all_values2, STD_DEVIATION_WINDOW_SECONDS);
        let events = merge_events(&all_values1, &all_values2);
        info!("Total train events: {}", events.len());

        let parameters = functions.iter().map(|&func| {
            let params = calibrate_params(&events, func, DealDirection::Sell1Buy2, DEFAULT_HOLD_MS);
            info!("{func:?} {params:?}");
            params
        }).collect::<Vec<_>>();

        info!("Test on {test_diapason:?}");
        all_values1.clear();
        all_values2.clear();
        let (order_books1, order_books2) = get_order_books(&tickers, &[inst1_id, inst2_id], test_diapason, Some(&db)).await?;
        convert_values(&order_books1, &order_books2, &mut all_values1, &mut all_values2);
        info!("Calculate std deviations with window {STD_DEVIATION_WINDOW_SECONDS}");
        calculate_std_deviations(&mut all_values1, STD_DEVIATION_WINDOW_SECONDS);
        calculate_std_deviations(&mut all_values2, STD_DEVIATION_WINDOW_SECONDS);

        let events = merge_events(&all_values1, &all_values2);
        info!("Total test events: {}", events.len());

        functions.iter().zip(parameters.iter()).for_each(|(&func, params)| {
            let result = run_simulation(&events, params, true);
            //let result = run_simulation_on_trades(&events, params, 10, false);
            info!("{func:?}. Deals {}. Income {}, outcome {}, commission {}, net profit {}",
            result.win + result.loss, result.income, result.outcome, result.commission, result.income - result.outcome - result.commission);
        });
    }

    Ok(())
}

fn convert_values(order_books1: &[OrderBook], order_books2: &[OrderBook], all_values1: &mut Vec<OrderBookValues>, all_values2: &mut Vec<OrderBookValues>) {
    all_values1.extend(order_books1.iter().filter_map(calculate_values));
    all_values2.extend(order_books2.iter().filter_map(calculate_values));
}

fn merge_events(values1: &[OrderBookValues], values2: &[OrderBookValues]) -> Vec<MarketEvent> {
    let mut events = Vec::with_capacity(values1.len() + values2.len());
    let (mut v_index1, mut v_index2) = (0, 0);
    while v_index1 < values1.len() || v_index2 < values2.len() {
        let mut cur_event = None;
        let mut cur_event_time = None;
        
        if let Some(v) = values1.get(v_index1) {
            let event = MarketEvent::OrderBook1(*v);
            let use_it = if let Some(event_time) = cur_event_time {
                v.time < event_time
            } else { true };
            if use_it {
                cur_event_time = Some(v.time);
                cur_event = Some(event);
            }
        }

        if let Some(v) = values2.get(v_index2) {
            let event = MarketEvent::OrderBook2(*v);
            let use_it = if let Some(event_time) = cur_event_time {
                v.time < event_time
            } else { true };
            if use_it {
                cur_event_time = Some(v.time);
                cur_event = Some(event);
            }
        }
        
        if let Some(event) = cur_event {
            match event {
                MarketEvent::OrderBook1(_) => v_index1 += 1,
                MarketEvent::OrderBook2(_) => v_index2 += 1,
            }
            events.push(event);
        } else {
            break;
        }
    }
    events
}

fn instrument_csv_header(ticker: &str) -> String {
    let mut header = format!("{ticker} time,{ticker} bid,{ticker} ask,{ticker} mid,{ticker} micro,{ticker} micro-mid");
    for levels in IMBALANCE_LEVELS {
        header.push_str(&format!(",{ticker} imbalance_{levels}"));
    }
    header
}

fn find_nearest_order_book(time: DateTime<Utc>, order_books: &[OrderBook], prev_index: usize) -> Option<usize> {
    let mut best_diff: Option<TimeDelta> = None;
    let mut best_index = None;

    let i_from = (prev_index as i32 - 10).max(0) as usize;
    let i_to = (prev_index + 10).min(order_books.len() - 1);
    for i in i_from..=i_to {
        let order_book = &order_books[i];
        let diff = (order_book.timestamp - time).abs();
        if let Some(cur_best) = best_diff {
            if diff < cur_best {
                best_diff = Some(diff);
                best_index = Some(i);
            }
        } else {
            best_diff = Some(diff);
            best_index = Some(i);
        }
    }

    best_index
}

#[derive(Copy, Clone, Debug)]
enum MarketEvent {
    OrderBook1(OrderBookValues),
    OrderBook2(OrderBookValues),
}

impl MarketEvent {
    pub fn event_datetime(&self) -> DateTime<Utc> {
        match self {
            MarketEvent::OrderBook1(order_book) => order_book.time,
            MarketEvent::OrderBook2(order_book) => order_book.time,
        }
    }
}

pub fn market_events_time_diapason(events: &[MarketEvent]) -> TimeDelta {
    events[events.len() - 1].event_datetime() - events[0].event_datetime()
}

#[derive(Copy, Clone, Debug)]
struct OrderBookValues {
    time: DateTime<Utc>,
    quantity_bid: u16,
    quantity_ask: u16,
    bid: f64,
    ask: f64,
    mid: f64,
    micro: f64,
    imbalances: [f64; IMBALANCE_LEVELS.len()],
    std_derivative: f64,
    trend: f64,
}

impl OrderBookValues {
    fn format_csv(&self) -> String {
        let mut str = format!("{},{},{},{},{},{}", self.time.to_rfc3339(), self.bid, self.ask, self.mid, self.micro, self.micro - self.mid);
        for imbalance in &self.imbalances {
            str.push_str(&format!(",{imbalance}"));
        }
        str
    }
}

fn calculate_values(order_book: &OrderBook) -> Option<OrderBookValues> {
    if order_book.asks.len() < 20 || order_book.bids.len() < 20 {
        return None;
    }

    let mut asks = order_book.asks.clone();
    asks.sort_by(|a, b| {a.price.partial_cmp(&b.price).unwrap()});

    let mut bids = order_book.bids.clone();
    bids.sort_by(|a, b| {b.price.partial_cmp(&a.price).unwrap()});

    let best_bid = bids[0].price.as_f64();
    let best_ask = asks[0].price.as_f64();
    let best_bid_volume = bids[0].size as f64;
    let best_ask_volume = asks[0].size as f64;
    let mid = (best_bid + best_ask) / 2.0;
    let micro = (best_ask * best_bid_volume + best_bid * best_ask_volume) / (best_bid_volume + best_ask_volume);

    let mut imbalances = [0.0; IMBALANCE_LEVELS.len()];
    for i in 0..IMBALANCE_LEVELS.len() {
        let levels = IMBALANCE_LEVELS[i];
        let (mut bids_sum, mut asks_sum) = (0.0, 0.0);
        for level in 0..levels {
            if let Some(bid) = bids.get(level) {
                bids_sum += bid.size as f64;
            }
            if let Some(ask) = asks.get(level) {
                asks_sum += ask.size as f64;
            }
        }
        imbalances[i] = (bids_sum - asks_sum) / (bids_sum + asks_sum);
    }

    Some(OrderBookValues {
        time: order_book.timestamp,
        quantity_bid: bids[0].size as u16,
        quantity_ask: asks[0].size as u16,
        bid: best_bid,
        ask: best_ask,
        mid,
        micro,
        imbalances,
        std_derivative: 0.0,
        trend: 0.0,
    })
}

fn calculate_std_deviations(values: &mut [OrderBookValues], diapason: i64) {
    let mut window_mids = Vec::new();
    let mut window_derivatives = Vec::new();
    let mut window_times = Vec::<DateTime<Utc>>::new();

    for (i, v) in values.iter_mut().enumerate() {
        let window_start = v.time - TimeDelta::seconds(diapason);
        while !window_times.is_empty() && window_times[0] < window_start {
            window_times.remove(0);
            window_derivatives.remove(0);
            window_mids.remove(0);
        }

        let mut prev_index = window_mids.len() as i32 - 1;
        while prev_index >= 0 && window_times[prev_index as usize] == v.time {
            prev_index -= 1;
        }

        let derivative: f64 = if prev_index >= 0 {
            (v.mid - window_mids[prev_index as usize]) / (v.time.timestamp_millis() - window_times[prev_index as usize].timestamp_millis()) as f64
        } else { 0.0 };

        if derivative.abs() > f64::MIN_POSITIVE {
            let d = if let Some(std) = standard_deviation(&window_derivatives) {
                if std > 0.0000001 {
                    derivative / std
                } else {
                    if derivative > 0.0 { 100.0 } else { -100.0 }
                }
            } else {
                0.0
            };
            if !d.is_nan() {
                v.std_derivative = d;
            }
        }

        window_mids.push(v.mid);
        window_derivatives.push(derivative);
        window_times.push(v.time);
    }
}

fn calculate_trend(values: &mut [OrderBookValues]) {
    let mut trend = 0.0;
    let mut prev_time = Option::<DateTime<Utc>>::None;

    for value in values.iter_mut() {
        if let Some(cur_prev_time) = prev_time {
            let dt = (value.time - cur_prev_time).as_seconds_f64();
            let alpha = 1.0 - (-dt / TREND_TAU_SECONDS).exp();
            trend += alpha * (value.std_derivative - trend);
            value.trend = trend;
            prev_time = Some(value.time);
        } else {
            trend = value.std_derivative;
            value.trend = trend;
            prev_time = Some(value.time);
        }
    }
}