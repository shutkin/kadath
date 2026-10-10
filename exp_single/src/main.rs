mod orderbook_values;
mod signal_calculator;
mod charts;
mod global_optimizer;

use crate::charts::draw_chart;
use crate::orderbook_values::OrderBookValues;
use crate::signal_calculator::SignalCalculator;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{LevelFilter, error, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use simplelog::SimpleLogger;
use crate::global_optimizer::global_optimize;

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let ticker = "GLZ6";

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    let inst_id = find_instrument_id(&all_instruments, ticker).expect("Instrument not found");
    let diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-09T14:19:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-09T20:00:00Z")?.to_utc(),
    );

    let mut values = get_order_books_values(ticker, inst_id, diapason, &db).await?;
    info!("Got {} order book values", values.len());

    for hold_time in [2_000, 5_000, 8_000, 12_000, 20_000] {
        let horizons = build_target(&values, hold_time);
        let penalty = hold_time as f64 / 30_000.0;
        if let Some(global) = global_optimize(&values, &horizons, penalty) {
            OrderBookValues::calculate_window_values(&mut values, global.derivatives_normalize, global.trend_tau);
            if let Some(result) = SignalCalculator::optimize(&values, &horizons, penalty) {
                info!("Hold {hold_time}: {result:?}");
                let predictions = values.iter().enumerate()
                    .map(|(i, v)| {
                        if i > 500 {
                            result.calculator.calculate(v)
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();
                if let Some(predictions) = scale_prediction(&predictions, 10.0) {
                    let char_filename = format!("hold_{}.png", hold_time);
                    if let Err(err) = draw_chart(&values, &horizons, &predictions, &char_filename) {
                        error!("{err}");
                    } else {
                        info!("Chart saved to {char_filename}");
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn scale_prediction(data: &[f64], scale: f64) -> Option<Vec<f64>> {
    Some(data.iter().copied().map(|x| {
        x.max(-1.0).min(1.0) * scale
    }).collect())
}
fn build_target(values: &[OrderBookValues], hold: i32) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let target_time = values[i].time + TimeDelta::milliseconds(hold as i64);
        if let Some(h) = find_value_on_horizon(values, i, target_time) {
            result.push(h.mid - values[i].mid);
        } else {
            result.push(0.0);
        }
    }

    result
}

fn find_value_on_horizon(
    values: &[OrderBookValues],
    index: usize,
    target_time: DateTime<Utc>,
) -> Option<OrderBookValues> {
    let mut i = index;
    while i < values.len() {
        if values[i].time >= target_time {
            return Some(values[i]);
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

async fn get_order_books_values(
    ticker: &str,
    id: i32,
    diapason: TimeDiapason,
    db: &Db,
) -> Result<Vec<OrderBookValues>, CommonError> {
    let order_books = get_order_books(ticker, id, diapason, db).await?;
    let mut values = order_books
        .iter()
        .filter_map(OrderBookValues::calculate)
        .collect::<Vec<_>>();
    Ok(values)
}

async fn get_order_books(
    ticker: &str,
    id: i32,
    diapason: TimeDiapason,
    db: &Db,
) -> Result<Vec<OrderBook>, CommonError> {
    if let Some(order_books1) = order_book_cache::read(ticker, diapason, true)? {
        Ok(order_books1)
    } else {
        let order_books1 = db.get_order_books_compact(id, diapason).await?;
        order_book_cache::write(ticker, diapason, &order_books1, true)?;
        Ok(order_books1)
    }
}
