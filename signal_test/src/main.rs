use chrono::{DateTime, Datelike, TimeDelta, Timelike, Utc};
use db::Db;
use linfa::DatasetBase;
use linfa::prelude::{Fit, Predict};
use linfa_clustering::KMeans;
use log::{LevelFilter, debug, error, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, merge_events};
use model::math_util::{mean, median, p10_p90};
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use ndarray::Array2;
use plotters::prelude::*;
use rust_decimal::Decimal;
use signal::strategy_optimizer::StrategyOptimizer;
use signal::{ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use simplelog::SimpleLogger;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Copy, Clone, Debug)]
struct TestDeal {
    open_index: usize,
    close_index: Option<usize>,
    open_time: DateTime<Utc>,
    close_time: DateTime<Utc>,
    open_price: Decimal,
    close_price: Option<Decimal>,
}

impl TestDeal {
    fn open(index: usize, ob: &OrderBook, hold: u16) -> Self {
        TestDeal {
            open_index: index,
            open_time: ob.timestamp,
            close_time: ob.timestamp + TimeDelta::milliseconds(hold as i64),
            open_price: get_best_ask(ob),
            close_price: None,
            close_index: None,
        }
    }

    fn close(&mut self, index: usize, ob: &OrderBook) -> Option<(Decimal, Decimal)> {
        if self.close_price.is_some() {
            return None;
        }

        self.close_index = Some(index);
        self.close_time = ob.timestamp;
        let close_price = get_best_bid(ob);
        self.close_price = Some(close_price);

        info!("{} Buy @ {} => {} Sell @ {}", self.open_time, self.open_price, ob.timestamp, close_price);
        
        Some((close_price, self.open_price))
    }
}

#[tokio::main]
async fn _main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let ticker = "GLZ6";

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    let inst_id = find_instrument_id(&all_instruments, ticker).ok_or("Failed to fins instrument")?;
    let mut diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-09T14:19:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-09T20:00:00Z")?.to_utc(),
    );
    let end = DateTime::parse_from_rfc3339("2026-10-10T00:00:00Z")?.to_utc();

    let mut all_events = Vec::new();

    while diapason.from < end {
        if !matches!(diapason.from.weekday().number_from_monday(), 6 | 7) {
            info!("DAY {}", diapason.from.date_naive());
            let order_books = get_order_books(ticker, inst_id, diapason, Some(&db)).await?;
            info!("{} {} order books", order_books.len(), ticker);
            let events = merge_events(&[inst_id], &[order_books]);
            all_events.extend(events);
        }

            diapason.from += TimeDelta::days(1);
            diapason.to += TimeDelta::days(1);
        }

        let optimizer = StrategyOptimizer::new(inst_id);
        info!("Start optimizing");
        let config = optimizer.optimize(&all_events, 5000)?;
        std::fs::write("optimized_config_10_09.txt", format!("{config:?}"))?;
        info!("Done");
    Ok(())
}

fn get_best_ask(order_book: &OrderBook) -> Decimal {
    order_book.asks.iter().map(|data| data.price).min().unwrap()
}

fn get_best_bid(order_book: &OrderBook) -> Decimal {
    order_book.bids.iter().map(|data| data.price).max().unwrap()
}

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let ticker = "GLZ6";

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    let inst_id = find_instrument_id(&all_instruments, ticker).ok_or("Failed to fins instrument")?;
    
    let config_buy = ModelConfig {
        hold_time_ms: 25000,
        derivatives_normalize_diapason_s: 325.0,
        trend_tau_s: 176.0,
        train_data_minutes: 45,
        train_interval_minutes: 1,
        decay_time: 360.0,
    };
    let model_buy = WalkForwardModel::new_buying(inst_id, config_buy);
    let mut window_data_buy = WindowData::new();

    let config_sell = ModelConfig {
        hold_time_ms: 1000,
        derivatives_normalize_diapason_s: 200.0,
        trend_tau_s: 175.0,
        train_data_minutes: 60,
        train_interval_minutes: 1,
        decay_time: 360.0,
    };
    let model_sell = WalkForwardModel::new_selling(inst_id, config_sell);
    let mut window_data_sell = WindowData::new();

    let mut diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-09T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-09T20:00:00Z")?.to_utc(),
    );
    let end = DateTime::parse_from_rfc3339("2026-10-13T00:00:00Z")?.to_utc();

    let (mut total_income, mut total_outcome, mut total_commission) = (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
    let mut active_deal = Option::<TestDeal>::None;

    let mut deals_visualisations = Vec::new();

    while diapason.from < end {
        if !matches!(diapason.from.weekday().number_from_monday(), 6 | 7) {
            info!("DAY {}", diapason.from.date_naive());
            let (mut daily_income, mut daily_outcome, mut daily_commission) = (Decimal::ZERO, Decimal::ZERO, Decimal::ZERO);
            let mut daily_deals = 0;

            let order_books = get_order_books(ticker, inst_id, diapason, Some(&db)).await?;
            if order_books.is_empty() {
                diapason.from += TimeDelta::days(1);
                diapason.to += TimeDelta::days(1);
                continue;
            }
            info!("{} {} order books", order_books.len(), ticker);

            let events = merge_events(&[inst_id], &[order_books]);
            debug!("Total events: {}", events.len());

            let mut process_delays = Vec::new();

            let mut prev_hour = events[0].order_book.timestamp.hour();
            for (i, event) in events.iter().enumerate() {
                if event.order_book.timestamp.hour() != prev_hour {
                    prev_hour = event.order_book.timestamp.hour();
                    debug!("Hour {prev_hour} PnL {}", daily_income - daily_outcome);
                }

                let started = Instant::now();
                let resp = model_buy.calibrate(true);
                if let Some(calc) = &resp.calculator {
                    let elapsed = Instant::now().duration_since(started).as_millis();
                    let pref = calc.get_performances();
                    info!(
                        "Calibration takes {elapsed}ms. Predictions inside training interval {:?}: {} correct, {} incorrect, score {}",
                        pref.train_diapason,
                        pref.training_correct_predictions,
                        pref.training_incorrect_predictions,
                        pref.training_score
                    );

                    if !process_delays.is_empty() {
                        info!(
                            "Process delay mean {}, median {}, p10-p90 {:?}, max {}",
                            mean(&process_delays), median(&process_delays), p10_p90(&process_delays),
                            process_delays.iter().copied().reduce(f64::max).unwrap_or_default(),
                        );
                        process_delays.clear();
                    }
                }

                let started = Instant::now();
                let signal_buy = model_buy.process(&mut window_data_buy, event);
                let elapsed = Instant::now().duration_since(started).as_micros() as f64;
                process_delays.push(elapsed);

                if let Some(deal) = active_deal.as_mut() {
                    if event.order_book.timestamp > deal.close_time {
                        if let Some((income, outcome)) = deal.close(i, &event.order_book) {

                            if let Some(visualization) = build_deal_visualization(&events, &deal) {
                                deals_visualisations.push(visualization);
                            }

                            total_income += income;
                            total_outcome += outcome;
                            daily_income += income;
                            daily_outcome += outcome;
                            active_deal = None;
                        }
                    } else {
                        /*match signal_sell {
                            TradeSignal::Sell(_) => {
                                let cur_sell_price = get_best_bid(&event.order_book);
                                if cur_sell_price > deal.open_price {
                                    if let Some((income, outcome)) = deal.close(&event.order_book) {
                                        total_income += income;
                                        total_outcome += outcome;
                                        daily_income += income;
                                        daily_outcome += outcome;
                                        active_deal = None;
                                    }
                                }
                            }
                            _ => {}
                        }*/
                    }
                } else {
                    match signal_buy {
                        TradeSignal::Buy(hold) => {
                            let deal = TestDeal::open(i, &event.order_book, hold.max(35000));
                            active_deal = Some(deal);
                            daily_deals += 1;
                        }
                        _ => {}
                    }
                }
            }
            info!("Day {} net {} on {} deals with commission {}", diapason.from.date_naive(), daily_income - daily_outcome - daily_commission, daily_deals, daily_commission);
        }
        diapason.from += TimeDelta::days(1);
        diapason.to += TimeDelta::days(1);
    }
    info!("Average deal PnL {}", (total_income.as_f64() - total_outcome.as_f64()) / deals_visualisations.len() as f64);

    let average_deal = average_deal_visual(&deals_visualisations)?;
    for (i, (cnt, avg)) in average_deal.iter().enumerate() {
        if let Err(err) = visualize_deal(avg, &format!("{} deals", cnt), &format!("average_deal_{i}.png")) {
            error!("Can't visualize deal: {err}");
        }
    }

    Ok(())
}

fn average_deal_visual(deals: &[Vec<f64>]) -> Result<Vec<(usize, Vec<f64>)>, CommonError> {
    let len = deals.iter().map(|deal| deal.len()).min().unwrap_or_default();

    info!("Clustering data");
    let centroids = deals.iter().map(|deal| {
        let mean = mean(deal);
        let mut centroid = deal[0..len].iter().map(|&d| d - mean).collect::<Vec<f64>>();
        let max = centroid.iter().map(|x| x.abs()).reduce(f64::max).unwrap_or_default();
        for i in 0..len {
            centroid[i] = centroid[i] / max;
        }
        centroid
    }).collect::<Vec<Vec<f64>>>();

    let n_samples = centroids.len();

    let flat: Vec<f64> = centroids.into_iter().flatten().collect();

    let records = Array2::from_shape_vec((n_samples, len), flat)?;

    let dataset = DatasetBase::from(records);
    let model = KMeans::params(5).tolerance(1e-2).fit(&dataset)?;
    let labels = model.predict(&dataset);

    info!("Build average visualization data for {} deals", deals.len());
    let mut result = Vec::with_capacity(5);

    for label in 0..5 {
        let mut label_avg = vec![0.0; len];
        let mut label_deals = HashSet::new();
        for i in 0..len {
            let mut buf = Vec::new();
            for (deal_i, deal) in deals.iter().enumerate() {
                let deal_label = labels[deal_i];
                if deal_label == label {
                    buf.push(deal[i]);
                    label_deals.insert(deal_i);
                }
            }
            label_avg[i] = mean(&buf);
        }
        result.push((label_deals.len(), label_avg));
    }

    Ok(result)
}

fn build_deal_visualization(events: &[OrderBookEvent], deal: &TestDeal) -> Option<Vec<f64>> {
    if let Some(close_index) = deal.close_index {
        let start = (deal.open_index as i32 - 500).max(0) as usize;
        let end = (close_index + 500).min(events.len() - 1);

        let start_timestamp = (deal.open_time - TimeDelta::seconds(20)).timestamp_millis() / 500;
        let end_timestamp = (deal.close_time + TimeDelta::seconds(20)).timestamp_millis() / 500;

        let mut half_second_map = HashMap::new();
        for i in start..=end {
            let t = events[i].order_book.timestamp.timestamp_millis() / 500;
            if t >= start_timestamp && t <= end_timestamp {
                let mid_price = (get_best_ask(&events[i].order_book).as_f64() + get_best_bid(&events[i].order_book).as_f64()) * 0.5;
                half_second_map.entry(t).or_insert_with(Vec::new).push(mid_price);
            }
        }

        let mut result = Vec::new();
        let first_prices = half_second_map.get(&half_second_map.keys().copied().min()?)?;
        let mut cur_price = mean(first_prices);
        for t in start_timestamp..=end_timestamp {
            if let Some(prices) = half_second_map.get(&t) {
                cur_price = mean(prices);
            }
            result.push(cur_price);
        }
        Some(result)
    } else {
        None
    }
}

fn visualize_deal(prices: &[f64], title: &str, filename: &str) -> EmptyResult {
    let min_price = prices.iter().copied().reduce(f64::min).unwrap_or_default();
    let max_price = prices.iter().copied().reduce(f64::max).unwrap_or_default();

    let mut times = Vec::with_capacity(prices.len());
    let mut trade_times = Vec::with_capacity(prices.len());
    for i in 0 .. prices.len() {
        let second = i as i32 / 2 - 20;
        times.push(second);

        trade_times.push(if i == 40 || i == 90 {2.0 * max_price} else {0.0});
    }

    let root = BitMapBackend::new(filename, (400, 400)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 25).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-20..times[times.len() - 1], min_price..max_price)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(AreaSeries::new(
            times.iter().map(|&t| (t, if t == 0 || t == 25 {10.0 * max_price} else {0.0})),
            0.0,
            RED.mix(0.5),
        ))?
        .label("Trade")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.5)));

    chart
        .draw_series(LineSeries::new(
            times.iter().copied().zip(prices.iter().copied()),
            BLUE.mix(0.5),
        ))?
        .label("Price")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.5)));

    // Save the buffer to disk
    root.present()?;
    Ok(())

}

fn find_value_on_horizon(values: &[OrderBookEvent], index: usize, inst_id: i32, target_time: DateTime<Utc>) -> Option<&OrderBook> {
    let mut i = index;
    while i < values.len() {
        if values[i].instrument_id == inst_id && values[i].order_book.timestamp >= target_time {
            return Some(&values[i].order_book);
        }
        i += 1;
    }
    None
}

fn find_instrument_id(instruments: &[Instrument], ticker: &str) -> Option<i32> {
    instruments
        .iter()
        .find(|instrument| instrument.ticker == ticker)
        .and_then(|instrument| instrument.id)
}

async fn get_order_books(ticker: &str, id: i32, diapason: TimeDiapason, db: Option<&Db>)
                         -> Result<Vec<OrderBook>, CommonError> {
    if let Some(order_books) = order_book_cache::read(ticker, diapason, true)? {
        Ok(order_books)
    } else {
        let order_books = db.unwrap().get_order_books_compact(id, diapason).await?;
        order_book_cache::write(ticker, diapason, &order_books, true)?;
        Ok(order_books)
    }
}
