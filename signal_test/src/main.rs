use chrono::{DateTime, Datelike, TimeDelta, Timelike, Utc};
use db::{Db, QueryAsksOrBids};
use log::{LevelFilter, debug, info, warn};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, merge_events};
use model::{order_book_cache, DataSource, Instrument, OrderBook, OrderData};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use signal::strategy_optimizer::StrategyOptimizer;
use signal::{ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use simplelog::SimpleLogger;
use model::math_util::{mean, median, p10_p90, standard_deviation};

#[derive(Copy, Clone)]
struct TestDeal {
    signal: TradeSignal,
    open_time1: DateTime<Utc>,
    open_time2: DateTime<Utc>,
    close_time: DateTime<Utc>,
    open_price1: Decimal,
    open_price2: Decimal,
    close_time1: DateTime<Utc>,
    close_time2: DateTime<Utc>,
    close_price1: Option<Decimal>,
    close_price2: Option<Decimal>,
}

impl TestDeal {
    fn open(signal: TradeSignal, ob1: &OrderBook, ob2: &OrderBook) -> Option<Self> {
        match signal {
            TradeSignal::Sell1Buy2(hold) => {
                Some(TestDeal {
                    signal,
                    open_time1: ob1.timestamp,
                    open_time2: ob2.timestamp,
                    close_time1: ob1.timestamp,
                    close_time2: ob2.timestamp,
                    close_time: ob1.timestamp.max(ob2.timestamp) + TimeDelta::milliseconds(hold as i64),
                    open_price1: get_best_bid(ob1),
                    open_price2: get_best_ask(ob2),
                    close_price1: None,
                    close_price2: None,
                })
            }
            TradeSignal::Buy1Sell2(hold) => {
                Some(TestDeal {
                    signal,
                    open_time1: ob1.timestamp,
                    open_time2: ob2.timestamp,
                    close_time1: ob1.timestamp,
                    close_time2: ob2.timestamp,
                    close_time: ob1.timestamp.max(ob2.timestamp) + TimeDelta::milliseconds(hold as i64),
                    open_price1: get_best_ask(ob1),
                    open_price2: get_best_bid(ob2),
                    close_price1: None,
                    close_price2: None,
                })
            }
            _ => {
                None
            }
        }
    }

    fn close1(&mut self, ob: &OrderBook) {
        if self.close_price1.is_some() {
            return;
        }

        self.close_time1 = ob.timestamp;
        match self.signal {
            TradeSignal::Sell1Buy2(_) => {
                self.close_price1 = Some(get_best_ask(ob));
            }
            TradeSignal::Buy1Sell2(_) => {
                self.close_price1 = Some(get_best_bid(ob));
            }
            _ => {}
        }
    }

    fn close2(&mut self, ob: &OrderBook) {
        if self.close_price2.is_some() {
            return;
        }

        self.close_time2 = ob.timestamp;
        match self.signal {
            TradeSignal::Sell1Buy2(_) => {
                self.close_price2 = Some(get_best_bid(ob));
            }
            TradeSignal::Buy1Sell2(_) => {
                self.close_price2 = Some(get_best_ask(ob));
            }
            _ => {}
        }
    }

    fn close(&self) -> (Decimal, Decimal) {
        if let Some(close_price1) = self.close_price1 && let Some(close_price2) = self.close_price2 {
            match self.signal {
                TradeSignal::Sell1Buy2(_) => {
                    debug!(
                        "Sell 1 @ {} at {} buy 2 @ {} at {} -> buy 1 @ {} at {} sell 2 @ {} at {}",
                        self.open_price1, self.open_time1, self.open_price2, self.open_time2,
                        close_price1, self.close_time1, close_price2, self.close_time2,
                    );
                    (self.open_price1 + close_price2, self.open_price2 + close_price1)
                },
                TradeSignal::Buy1Sell2(_) => {
                    debug!(
                        "Buy 1 @ {} at {} sell 2 @ {} at {} -> sell 1 @ {} at {} buy 2 @ {} at {}",
                        self.open_price1, self.open_time1, self.open_price2, self.open_time2,
                        close_price1, self.close_time1, close_price2, self.close_time2,
                    );
                    (self.open_price2 + close_price1, self.open_price1 + close_price2)
                },
                _ => unreachable!(),
            }
        } else {
            (Decimal::ZERO, Decimal::ZERO)
        }
    }
}

const ENTRY_SLIPPAGE_MS: u16 = 300;
const EXIT_SLIPPAGE_MS: u16 = 100;

struct TestContext {
    fixed_hold_time: u16,
    model: WalkForwardModel,
    window_data: WindowData,
    active_deal: Option<TestDeal>,
    total_income: Decimal,
    total_outcome: Decimal,
    total_commission: Decimal,
}

#[tokio::main]
async fn _main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    //let tickers = ["GLU6", "GLZ6"];
    let tickers = ["GLZ6", "GLH7"];

    let all_instruments = db.get_instruments(DataSource::Alor).await?;
    if let (Some(inst1_id), Some(inst2_id)) = (
        find_instrument_id(&all_instruments, tickers[0]),
        find_instrument_id(&all_instruments, tickers[1]),
    ) {
        let mut diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-09-25T05:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-09-25T20:00:00Z")?.to_utc(),
        );
        let end = DateTime::parse_from_rfc3339("2026-09-26T00:00:00Z")?.to_utc();

        let mut all_events = Vec::new();

        while diapason.from < end {
            if !matches!(diapason.from.weekday().number_from_monday(), 6 | 7) {
                info!("DAY {}", diapason.from.date_naive());
                let (order_books1, order_books2) = get_order_books(&tickers, &[inst1_id, inst2_id], diapason, Some(&db)).await?;
                info!("{} {} order books, {} {} order books", order_books1.len(), tickers[0], order_books2.len(), tickers[1]);
                let events = merge_events(&[inst1_id, inst2_id], &[order_books1, order_books2]);
                all_events.extend(events);
            }

            diapason.from += TimeDelta::days(1);
            diapason.to += TimeDelta::days(1);
        }

        let optimizer = StrategyOptimizer::new(inst1_id, inst2_id);
        info!("Start optimizing");
        let config = optimizer.optimize(&all_events)?;
        std::fs::write("optimized_config_03-25.txt", format!("{config:?}"))?;
        info!("Done");
    }
    Ok(())
}

#[tokio::main]
async fn main() -> EmptyResult {
    const INST1_ID: i32 = 26;
    const INST2_ID: i32 = 121;
    const TICKERS: [&str; 2] = ["GLZ6", "GLH7"];

    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-09-25T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-09-25T20:00:00Z")?.to_utc(),
    );
    let (alor1, alor2) = get_order_books(&TICKERS, &[INST1_ID, INST2_ID], diapason, Some(&db)).await?;
    let alor_events = merge_events(&[INST1_ID, INST2_ID], &[alor1, alor2]);
    let alor_prices = get_spreads(&[INST1_ID, INST2_ID], &alor_events);
    info!("{} Alor prices", alor_prices.len());

    let t_invest1 = load_alt_order_books("GLZ6.json")?;
    let t_invest2 = load_alt_order_books("GLH7.json")?;
    let t_invest_events = merge_events(&[0, 1], &[t_invest1, t_invest2]);
    let t_invest_prices = get_spreads(&[0, 1], &t_invest_events);
    info!("{} T-Invest prices", t_invest_prices.len());

    /*for i in -10..=10 {
        let lag = i * 50;
        let (correlation, p50, p90, stddev) = spreads_correlation(lag, &alor_spreads, &t_invest_spreads);
        info!("Lag {lag}: correlation={correlation}, p50={p50}, p90={p90}, stddev={stddev}");
    }*/

    let (mean, p50, p90, stddev) = spreads_correlation(0, &alor_prices, &t_invest_prices, 0);
    info!("GLZ6 Alor - T_Invest: mean={mean}, p50={p50}, p90={p90}, stddev={stddev}");
    let (mean, p50, p90, stddev) = spreads_correlation(0, &alor_prices, &t_invest_prices, 1);
    info!("GLH7 Alor - T_Invest: mean={mean}, p50={p50}, p90={p90}, stddev={stddev}");
    let (mean, p50, p90, stddev) = spreads_correlation(0, &alor_prices, &t_invest_prices, 2);
    info!("Spread Alor - T_Invest: mean={mean}, p50={p50}, p90={p90}, stddev={stddev}");

    Ok(())
}

fn spreads_correlation(lag: i64, prices1: &[(i64, f64, f64)], prices2: &[(i64, f64, f64)], case: u8) -> (f64, f64, f64, f64) {
    let timestamp_from = prices1[0].0.max(prices2[0].0);
    let timestamp_to = prices1[prices1.len() - 1].0.min(prices2[prices2.len() - 1].0);

    let (mut i1, mut i2) = (0, 0);
    let mut timestamp = ((timestamp_from / 50) + 1) * 50;
    let (mut grid_spreads1, mut grid_spreads2) = (Vec::new(), Vec::new());
    while timestamp < timestamp_to - 50 {
        while i1 < prices1.len() - 2 && prices1[i1 + 1].0 < timestamp {
            i1 += 1;
        }
        while i2 < prices2.len() - 2 && prices2[i2 + 1].0 < timestamp + lag {
            i2 += 1;
        }
        grid_spreads1.push((prices1[i1].1, prices1[i1].2));
        grid_spreads2.push((prices2[i2].1, prices2[i2].2));

        timestamp += 50;
    }

    let errors = grid_spreads1.iter().zip(&grid_spreads2)
        .map(|(s1, s2)| {
            match case {
                0 => s1.0 - s2.0,
                1 => s1.1 - s2.1,
                _ => s1.1 - s1.0 - (s2.1 - s2.0),
            }
        })
        .collect::<Vec<_>>();
    let p50 = median(&errors);
    let (_p10, p90) = p10_p90(&errors);
    let stddev = standard_deviation(&errors).unwrap();
    let mean = mean(&errors);
    //let correlation = correlation::pearsonr(&grid_spreads1, &grid_spreads2);
    (mean, p50, p90, stddev)
}

fn get_spreads(ids: &[i32], events: &[OrderBookEvent]) -> Vec<(i64, f64, f64)> {
    let (mut ob1, mut ob2) = (None, None);
    let mut spreads = Vec::new();
    for event in events {
        if event.instrument_id == ids[0] {
            ob1 = Some(&event.order_book);
        } else if event.instrument_id == ids[1] {
            ob2 = Some(&event.order_book);
        } else {
            continue;
        }

        if let Some(ob1) = ob1 && let Some(ob2) = ob2 {
            let price1 = (get_best_ask(ob1) + get_best_bid(ob1)).to_f64().unwrap() * 0.5;
            let price2 = (get_best_ask(ob2) + get_best_bid(ob2)).to_f64().unwrap() * 0.5;
            spreads.push((event.order_book.timestamp.timestamp_millis(), price1, price2));
        }
    }

    spreads
}

fn get_best_ask(order_book: &OrderBook) -> Decimal {
    order_book.asks.iter().map(|data| data.price).min().unwrap()
}

fn get_best_bid(order_book: &OrderBook) -> Decimal {
    order_book.bids.iter().map(|data| data.price).max().unwrap()
}

#[tokio::main]
async fn __main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let tickers = ["GLZ6", "GLH7"];

    let all_instruments = db.get_instruments(DataSource::Alor).await?;
    if let (Some(inst1_id), Some(inst2_id)) = (
        find_instrument_id(&all_instruments, tickers[0]),
        find_instrument_id(&all_instruments, tickers[1]),
    ) {
        //let config = ModelConfig { std_diapason_s: 1195, train_data_minutes: 52, train_interval_minutes: 1, decay_time: 21.273781754192633, fixed_hold_time: None };
        let config = ModelConfig { std_diapason_s: 1184, train_data_minutes: 22, train_interval_minutes: 1, decay_time: 40.90421962773701, fixed_hold_time: None };
        let model = WalkForwardModel::new_with_config(0, 1, config);
        let mut window_data = WindowData::new();

        let mut diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-09-25T05:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-09-25T20:00:00Z")?.to_utc(),
        );
        //let end = Utc::now();
        let end = DateTime::parse_from_rfc3339("2026-09-26T00:00:00Z")?.to_utc();

        let (mut total_income, mut total_outcome, mut total_commission) = (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
        let mut active_deal = Option::<TestDeal>::None;

        while diapason.from < end {
            if !matches!(diapason.from.weekday().number_from_monday(), 6 | 7) {
                info!("DAY {}", diapason.from.date_naive());
                let (mut daily_income, mut daily_outcome, mut daily_commission) = (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
                let mut daily_deals = 0;

                let (order_books1, order_books2) = get_order_books(&tickers, &[inst1_id, inst2_id], diapason, Some(&db)).await?;
                info!("{} {} order books, {} {} order books", order_books1.len(), tickers[0], order_books2.len(), tickers[1]);

                let order_books1_alt = load_alt_order_books("GLZ6.json")?;
                let order_books2_alt = load_alt_order_books("GLH7.json")?;
                info!("{} {} alternative order books, {} {} alternative order books", order_books1_alt.len(), tickers[0], order_books2_alt.len(), tickers[1]);

                let events = merge_events(&[inst1_id, inst2_id, 0, 1], &[order_books1, order_books2, order_books1_alt, order_books2_alt]);
                info!("Total events: {}", events.len());

                let mut prev_hour = events[0].order_book.timestamp.hour();
                let (mut last_ob1, mut last_ob2) = (None, None);
                for (i, event) in events.iter().enumerate() {
                    if event.order_book.timestamp.hour() != prev_hour {
                        prev_hour = event.order_book.timestamp.hour();
                        info!("Hour {prev_hour}");
                    }

                    let trade_signal = if event.instrument_id == 0 || event.instrument_id == 1 {
                        model.calibrate(true);
                        model.process(&mut window_data, event)
                    } else {
                        TradeSignal::None
                    };
                    let _trade_signal = match trade_signal {
                        TradeSignal::None => TradeSignal::None,
                        TradeSignal::Buy1Sell2(hold) => TradeSignal::Sell1Buy2(hold),
                        TradeSignal::Sell1Buy2(hold) => TradeSignal::Buy1Sell2(hold),
                    };

                    if event.instrument_id == inst1_id {
                        last_ob1 = Some(event.order_book.clone());
                    } else if event.instrument_id == inst2_id {
                        last_ob2 = Some(event.order_book.clone());
                    }

                    if let Some(ob1) = &last_ob1 && let Some(ob2) = &last_ob2 {
                        if let Some(deal) = active_deal.as_mut() {
                            if ob1.timestamp > deal.close_time {
                                if let Some((hob1, _)) = on_horizon(&events, i, inst1_id, ENTRY_SLIPPAGE_MS) {
                                    deal.close1(hob1);
                                } else {
                                    deal.close1(ob1);
                                }
                            }
                            if ob2.timestamp > deal.close_time {
                                if let Some((_, hob2)) = on_horizon(&events, i, inst1_id, ENTRY_SLIPPAGE_MS) {
                                    deal.close2(hob2);
                                } else {
                                    deal.close2(ob2);
                                }
                            }

                            if deal.close_price1.is_some() && deal.close_price2.is_some() {
                                let (revenue, cost) = deal.close();
                                let commission = Decimal::from(5);
                                model.inform_deal_result(deal.signal, (revenue - cost - commission).as_f64());
                                total_income += revenue;
                                daily_income += revenue;
                                total_outcome += cost;
                                daily_outcome += cost;
                                total_commission += commission;
                                daily_commission += commission;
                                daily_deals += 1;
                                active_deal = None;
                            }
                        } else if matches!(trade_signal, TradeSignal::Buy1Sell2(_) | TradeSignal::Sell1Buy2(_)) {
                            if let Some((hob1, hob2)) = on_horizon(&events, i, inst1_id, EXIT_SLIPPAGE_MS) {
                                active_deal = TestDeal::open(trade_signal, hob1, hob2);
                            } else {
                                active_deal = TestDeal::open(trade_signal, ob1, ob2);
                            }
                        }
                    }
                }

                info!("Day {} net {} on {} deals with commission {}", diapason.from.date_naive(), daily_income - daily_outcome - daily_commission, daily_deals, daily_commission);
            }

            diapason.from += TimeDelta::days(1);
            diapason.to += TimeDelta::days(1);
        }
        info!("Net profit {}", total_income - total_outcome - total_commission);
    }
    Ok(())
}

impl TestContext {
    fn new(fixed_hold_time: u16, model: WalkForwardModel) -> Self {
        Self {
            fixed_hold_time,
            model,
            window_data: WindowData::new(),
            active_deal: None,
            total_income: Decimal::ZERO,
            total_outcome: Decimal::ZERO,
            total_commission: Decimal::ZERO,
        }
    }
}

fn on_horizon(events: &[OrderBookEvent], i: usize, first_leg_id: i32, horizon: u16) -> Option<(&OrderBook, &OrderBook)> {
    if horizon == 0 {
        return None;
    }

    let mut index = i;
    let target_time = events[i].order_book.timestamp + TimeDelta::milliseconds(horizon as i64);

    let (mut i1, mut i2) = (None, None);
    while index < events.len() {
        if events[index].order_book.timestamp >= target_time {
            if events[index].instrument_id == first_leg_id {
                i1 = Some(index);
            } else {
                i2 = Some(index);
            }
        }

        if let Some(i1) = i1 && let Some(i2) = i2 {
            return Some((&events[i1].order_book, &events[i2].order_book));
        }

        index += 1;
    }

    None
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
            (db.unwrap().get_order_books(Some(ids[0]), diapason, QueryAsksOrBids::BOTH).await?,
             db.unwrap().get_order_books(Some(ids[1]), diapason, QueryAsksOrBids::BOTH).await?);
        order_book_cache::write(tickers[0], diapason, &order_books1, true)?;
        order_book_cache::write(tickers[1], diapason, &order_books2, true)?;
        Ok((order_books1, order_books2))
    }
}

fn alternative_order_books(order_books: Vec<OrderBook>, ob_alternative: &[OrderBook]) -> Vec<OrderBook> {
    let mut result = Vec::with_capacity(order_books.len());

    let mut prev_alt_index = 0;
    for mut order_book in order_books {
        let alt_from: i32 = (prev_alt_index - 500).max(0);
        let alt_to: i32 = (prev_alt_index + 50000).min(ob_alternative.len() as i32);
        let mut nearest_distance = 10000;
        let mut nearest_alt_ob = None;
        for i in alt_from..alt_to {
            let alt_ob = &ob_alternative[i as usize];
            //info!("{} - {}", order_book.timestamp, alt_ob.timestamp);
            let distance = (order_book.timestamp.timestamp_millis() - alt_ob.timestamp.timestamp_millis()).abs();
            if distance < nearest_distance {
                nearest_distance = distance;
                nearest_alt_ob = Some((i, alt_ob));
            }
        }
        if let Some((i, nearest_alt_ob)) = nearest_alt_ob {
            prev_alt_index = i;
            order_book.asks = nearest_alt_ob.asks.clone();
            order_book.bids = nearest_alt_ob.bids.clone();
            result.push(order_book);
        } else {
            warn!("Timestamp {} not found", order_book.timestamp);
        }
    }

    result
}

fn check_events(events: &[OrderBookEvent], inst1_id: i32) {
    let mut first_first = 0;
    let mut first_second = 0;
    let mut second_first = 0;
    let mut second_second = 0;
    events.windows(2).for_each(|window| {
        if window.len() > 1 {
            if window[0].instrument_id == inst1_id {
                if window[1].instrument_id == inst1_id {
                    first_first += 1;
                } else {
                    first_second += 1;
                }
            } else {
                if window[1].instrument_id == inst1_id {
                    second_first += 1;
                } else {
                    second_second += 1;
                }
            }
        }
    });
    info!("1->1: {first_first}, 1->2: {first_second}, 2->1: {second_first}, 2->2: {second_second}");
}

fn load_alt_order_books(path: &str) -> Result<Vec<OrderBook>, CommonError> {
    let json = std::fs::read_to_string(path)?;
    let ob_alt = serde_json::from_str::<Vec<OrderBookAlt>>(&json)?;
    Ok(ob_alt.into_iter()
        .map(|alt| OrderBook {
            id: alt.id,
            timestamp: alt.timestamp,
            asks: alt.asks.into_iter().map(|v| OrderData { price: v.price, size: v.size.to_i32().unwrap() }).collect(),
            bids: alt.bids.into_iter().map(|v| OrderData { price: v.price, size: v.size.to_i32().unwrap() }).collect(),
        }).collect())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderBookAlt {
    pub id: Option<i64>,
    pub timestamp: DateTime<Utc>,
    pub asks: Vec<OrderDataAlt>,
    pub bids: Vec<OrderDataAlt>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct OrderDataAlt {
    pub price: Decimal,
    pub size: Decimal,
}
