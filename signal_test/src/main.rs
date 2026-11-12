use chrono::{DateTime, Datelike, TimeDelta, Timelike, Utc};
use db::Db;
use linfa::DatasetBase;
use linfa::prelude::{Fit, Predict};
use linfa_clustering::KMeans;
use log::{LevelFilter, debug, error, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, merge_events};
use model::math_util::{mean};
use model::{DataSource, Instrument, OrderBook, TradeDirection, order_book_cache};
use ndarray::Array2;
use plotters::prelude::*;
use rust_decimal::Decimal;
use signal::{DealData, ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use simplelog::SimpleLogger;
use std::collections::{HashMap, HashSet};
use signal::gradient_boost::GbdtConfig;

#[derive(Copy, Clone)]
struct TestDeal {
    prediction: (f64, f64),
    open_index: usize,
    close_index: Option<usize>,
    open_time: DateTime<Utc>,
    open_price: (f64, f64),
    close_price: Option<(f64, f64)>,

    deal_data: DealData,
}

impl TestDeal {
    fn open(prediction: (f64, f64), index: usize, book: &OrderBook) -> Self {
        let cur_time = book.timestamp;
        TestDeal {
            prediction,
            open_index: index,
            open_time: cur_time,
            open_price: (get_best_ask(book).as_f64(), get_mid_price(book).as_f64()),
            close_price: None,
            close_index: None,
            deal_data: DealData::new(book, TradeDirection::Buy),
        }
    }

    fn close(&mut self, index: usize, book: &OrderBook) {
        if self.close_price.is_some() {
            return;
        }

        self.close_index = Some(index);
        self.close_price = Some((get_best_ask(book).as_f64(), get_mid_price(book).as_f64()));
    }

    fn pnl(&self) -> (f64, f64) {
        let real = self.close_price.unwrap_or_default().0 - self.open_price.0;
        let mid = self.close_price.unwrap_or_default().1 - self.open_price.1;
        (real, mid)
    }
}

fn setup_logging(log_filename: &str) -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        // Format the log output: [Year-Month-Day Hour:Minute:Second][Level] Message
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        // Set the minimum log level to filter messages
        .level(LevelFilter::Info)
        // Direct the logs to a file
        .chain(fern::log_file(log_filename)?)
        .chain(std::io::stdout())
        .apply()?;
    Ok(())
}


fn get_mid_price(order_book: &OrderBook) -> f64 {
    (get_best_ask(order_book).as_f64() + get_best_bid(order_book).as_f64()) * 0.5
}

fn get_best_ask(order_book: &OrderBook) -> Decimal {
    order_book.asks.iter().map(|data| data.price).min().unwrap()
}

fn get_best_bid(order_book: &OrderBook) -> Decimal {
    order_book.bids.iter().map(|data| data.price).max().unwrap()
}

const OPTIMIZED_CONFIGS: [(GbdtConfig, GbdtConfig); 0] = [
];

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let ticker = "SiZ6";

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    let instrument_id = find_instrument_id(&all_instruments, ticker).ok_or("Failed to find instrument")?;

    let config = ModelConfig {
        derivatives_normalize_diapason_s: 57.28,
        trend_tau_s: 614.61,
        train_buf_size: 142000,
        train_min_size: 100000,
        optimize_iterations: 50,
        hold_time: TimeDelta::milliseconds(2000),
        expected_delta: 5.0,
        prediction_percentile_linear: 0.96,
        prediction_percentile_gbm: 0.93,
    };

    let mut window_data = WindowData::new(&config);
    let model = WalkForwardModel::new(instrument_id, config);

    let mut diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-22T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-22T20:00:00Z")?.to_utc(),
    );
    let end = DateTime::parse_from_rfc3339("2026-10-24T00:00:00Z")?.to_utc();

    let mut active_deal = Option::<TestDeal>::None;
    let mut deals = Vec::new();
    let mut deals_visualizations = Vec::new();

    let mut do_not_trade_until = None;

    let mut optimized_config_index = 0;
    let mut last_optimize_time = diapason.from;
    let mut last_train_time = diapason.from + TimeDelta::hours(6);

    while diapason.from < end {
        if !matches!(diapason.from.weekday().number_from_monday(), 6 | 7) {
            info!("DAY {}", diapason.from.date_naive());

            let books = get_order_books(ticker, instrument_id, diapason, Some(&db)).await?;
            if books.is_empty() {
                diapason.from += TimeDelta::days(1);
                diapason.to += TimeDelta::days(1);
                continue;
            }
            info!("{} {} order books", books.len(), ticker);

            let events = merge_events(&[instrument_id], &[&books]);
            debug!("Total events: {}", events.len());

            let mut prev_hour = events[0].order_book.timestamp.hour();
            for (i, event) in events.iter().enumerate() {
                if event.order_book.timestamp.hour() != prev_hour {
                    prev_hour = event.order_book.timestamp.hour();
                    info!("Hour {prev_hour}");
                }

                if last_optimize_time < event.order_book.timestamp - TimeDelta::hours(5)
                    && model.optimize_config(OPTIMIZED_CONFIGS.get(optimized_config_index).copied()) {
                    info!("Optimized on {}", event.order_book.timestamp);
                    last_optimize_time = event.order_book.timestamp;
                    optimized_config_index += 1;
                }

                if last_train_time < event.order_book.timestamp - TimeDelta::minutes(30)
                    && model.train() {
                    last_train_time = event.order_book.timestamp;
                }

                let signal = model.process(&mut window_data, event);

                if let Some(deal) = active_deal.as_mut() {
                    if model.is_it_time_to_close_deal(&event.order_book, &mut deal.deal_data) {
                        deal.close(i, &event.order_book);
                        let pnl = deal.pnl();
                        info!(
                            "{} Prediction {:?}: {:?}",
                            deal.open_time,
                            deal.prediction,
                            pnl,
                        );
                        if pnl.1 < -5.0 {
                            do_not_trade_until = Some(event.order_book.timestamp + TimeDelta::minutes(4));
                        }
                        deals.push(*deal);
                        if let Some(visualization) = build_deal_visualization(&events, deal) {
                            deals_visualizations.push(visualization);
                        }
                        active_deal = None;
                    }
                } else {
                    let do_trade = do_not_trade_until
                        .map(|until| event.order_book.timestamp > until)
                        .unwrap_or(true);
                    if do_trade {
                        do_not_trade_until = None;
                        if let TradeSignal::Sell(p0, p1) = signal {
                            let deal = TestDeal::open((p0, p1), i, &event.order_book);
                            active_deal = Some(deal);
                        }
                    }
                }
            }
        }
        diapason.from += TimeDelta::days(1);
        diapason.to += TimeDelta::days(1);
    }

    let mut pnl = 0.0;
    let (mut neutrals, mut positives, mut negatives) = (0, 0, 0);
    for deal in &deals {
        let deal_pnl = deal.pnl();
        pnl += deal_pnl.0;
        if deal_pnl.0 >= 1.5 {
            positives += 1;
        } else if deal_pnl.0 < -0.5 {
            negatives += 1;
        } else {
            neutrals += 1;
        }
    }
    info!("Total PnL {pnl}, PnL per deal {}", pnl / deals.len() as f64);
    info!("Positives {positives}, negatives {negatives}, neutrals {neutrals}");

    match average_deal_visual(&deals_visualizations) {
        Ok(visualizations) => {
            for (i, (deals, data)) in visualizations.iter().enumerate() {
                if let Err(err) = visualize_deal(data, &format!("{deals} deals"), &format!("deals_{i}.png")) {
                    error!("Failed to print deals visualisation {err}");
                }
            }
        }
        Err(err) => {
            error!("Average deal visualizations failed: {err}");
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
        let start = (deal.open_index as i32 - 1000).max(0) as usize;
        let end = (close_index + 5000).min(events.len() - 1);

        let start_timestamp = (deal.open_time - TimeDelta::seconds(10)).timestamp_millis() / 500;
        let end_timestamp = (deal.open_time + TimeDelta::seconds(40)).timestamp_millis() / 500;

        let mut half_second_map = HashMap::new();
        for event in &events[start..=end] {
            let mid = get_mid_price(&event.order_book);
            let t = event.order_book.timestamp.timestamp_millis() / 500;
            if mid > 0.0 && t >= start_timestamp && t <= end_timestamp {
                half_second_map.entry(t).or_insert_with(Vec::new).push(mid);
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
            times.iter().map(|&t| (t, if t == 0 || t == 15 {10.0 * max_price} else {0.0})),
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
        let order_books = db.unwrap().get_order_books(id, diapason).await?;
        order_book_cache::write(ticker, diapason, &order_books, true)?;
        Ok(order_books)
    }
}
