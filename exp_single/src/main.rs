mod orderbook_values;
mod signal_calculator;
mod charts;
mod global_optimizer;

use std::collections::HashMap;
use crate::charts::draw_chart;
use crate::orderbook_values::OrderBookValues;
use crate::signal_calculator::SignalCalculator;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{LevelFilter, error, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use simplelog::SimpleLogger;
use crate::global_optimizer::{global_optimize, GlobalParams, GlobalParamsPair};

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

    let values = get_order_books_values(ticker, inst_id, diapason, &db).await?;
    info!("Got {} order book values", values.len());

    let predefined_global = HashMap::<i32, GlobalParamsPair>::from([
        (1000, GlobalParamsPair {
            positive: GlobalParams { derivatives_normalize: 310.57039322142003, trend_tau: 175.0958973445236 },
            negative: GlobalParams { derivatives_normalize: 323.5415439983544, trend_tau: 194.32615414121923 },
        }),
        (2000, GlobalParamsPair {
            positive: GlobalParams { derivatives_normalize: 197.4146592029373, trend_tau: 221.85718232374626 },
            negative: GlobalParams { derivatives_normalize: 123.84916265010177, trend_tau: 414.4764462270979 },
        }),
        (5000, GlobalParamsPair {
            positive: GlobalParams { derivatives_normalize: 319.56008478808786, trend_tau: 186.84468494599795 },
            negative: GlobalParams { derivatives_normalize: 265.37959943793913, trend_tau: 414.3870940034053 },
        }),
        (8000, GlobalParamsPair {
            positive: GlobalParams { derivatives_normalize: 236.5761576638887, trend_tau: 156.7790896889341 },
            negative: GlobalParams { derivatives_normalize: 199.05476741479973, trend_tau: 174.16445560511082 },
        }),
        (12000, GlobalParamsPair {
            positive: GlobalParams { derivatives_normalize: 178.5188522823508, trend_tau: 305.7950389128884 },
            negative: GlobalParams { derivatives_normalize: 173.60311997864903, trend_tau: 234.95745365183876 },
        }),
        (20000, GlobalParamsPair {
            positive: GlobalParams { derivatives_normalize: 271.97923439323483, trend_tau: 187.23093437360188 },
            negative: GlobalParams { derivatives_normalize: 306.53057325190787, trend_tau: 251.51433853946128 },
        }),
        (25000, GlobalParamsPair {
            positive: GlobalParams { derivatives_normalize: 324.5607531339353, trend_tau: 176.17347347575284 }, // Score 2608.81666666647
            negative: GlobalParams { derivatives_normalize: 199.05476741479973, trend_tau: 174.16445560511082 }, // No Score
        })
    ]);

    for hold_time in [1_000, 2_000, 5_000, 8_000, 12_000, 20_000, 25_000] {
        let horizons = build_horizon(&values, hold_time);
        let penalty = hold_time as f64 / 30_000.0;
        if let Some(global) = predefined_global.get(&hold_time).copied().or_else(|| global_optimize(&values, &horizons, penalty)) {
            let mut values_positive = values.clone();
            OrderBookValues::calculate_window_values(&mut values_positive, global.positive.derivatives_normalize, global.positive.trend_tau);
            let mut values_negative = values.clone();
            OrderBookValues::calculate_window_values(&mut values_negative, global.negative.derivatives_normalize, global.negative.trend_tau);
            if let Some(result_positive) = SignalCalculator::optimize(&values_positive, &horizons, true, penalty)
                && let Some(result_negative) = SignalCalculator::optimize(&values_negative, &horizons, false, penalty) {
                info!("Hold {hold_time}: positive {result_positive:?}, negative {result_negative:?}");
                let (mut success, mut failure) = (0, 0);
                let predictions = values_positive.iter().zip(values_negative.iter()).enumerate()
                    .map(|(i, (vp, vn))| {
                        if i > 500 {
                            let positive = result_positive.calculator.calculate(vp).max(0.0);
                            let negative = result_negative.calculator.calculate(vn).min(0.0);
                            let signal = if positive > -negative {
                                positive
                            } else {
                                negative
                            };
                            let r = signal * horizons[i];
                            if r.abs() > 0.00001 {
                                if r > 0.0 {
                                    success += 1;
                                } else {
                                    failure += 1;
                                }
                            }
                            signal
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();
                info!("Success: {success}, Failure: {failure}, Rate {:.2}%", success as f64 * 100.0 / (success + failure) as f64);
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
fn build_horizon(values: &[OrderBookValues], hold: i32) -> Vec<f64> {
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

fn find_value_on_horizon(values: &[OrderBookValues], index: usize, target_time: DateTime<Utc>) -> Option<OrderBookValues> {
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
        let order_books = db.get_order_books(id, diapason).await?;
        order_book_cache::write(ticker, diapason, &order_books, true)?;
        Ok(order_books)
    }
}
