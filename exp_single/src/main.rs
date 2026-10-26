mod orderbook_values;
mod signal_calculator;
mod charts;
mod global_optimizer;
mod gradient_boost;

use crate::global_optimizer::{GlobalParams, global_optimize};
use crate::gradient_boost::{GradientBoost, PredictionType};
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{LevelFilter, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use simplelog::SimpleLogger;
use std::collections::HashMap;
use model::math_util::{mean, median, p05_p95};

#[tokio::main]
async fn _main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let ticker = "SiZ6";

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    let inst_id = find_instrument_id(&all_instruments, ticker).expect("Instrument not found");
    let diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-22T10:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-22T18:00:00Z")?.to_utc(),
    );

    let all_values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
    info!("Got {} order book values", all_values.len());

    let all_horizons = build_horizon(&all_values, 2000);

    let result = global_optimize(
        &all_values,
        &all_horizons,
        TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-10-22T10:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-10-22T15:59:00Z")?.to_utc(),
        ),
        TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-10-22T16:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-10-22T18:00:00Z")?.to_utc(),
        ),
    );
    info!("{result:?}");

    Ok(())
}

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let ticker = "SiZ6";

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    let inst_id = find_instrument_id(&all_instruments, ticker).expect("Instrument not found");
    let diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-22T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-22T20:00:00Z")?.to_utc(),
    );

    let mut all_values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
    info!("Got {} order book values", all_values.len());

    let predefined_global = HashMap::<i32, GlobalParams>::from([
        (1000, GlobalParams { derivatives_normalize: 200.0, trend_tau: 180.0 }),
        (2000, GlobalParams { derivatives_normalize: 91.69980212324158, trend_tau: 301.09667694458244 }),
        (3000, GlobalParams { derivatives_normalize: 200.0, trend_tau: 180.0 }),
    ]);

    const HOLD_TIME: i32 = 2_000;
    const TRAIN_DIAPASON: i64 = 5 * 60;
    const TEST_DIAPASON: i64 = 30;

    let horizons = build_horizon(&all_values, HOLD_TIME);

    const THRESHOLD: f64 = 5.0;
    let (mut correct, mut incorrect) = (0_u32, 0_u32);
    let (mut missed, mut total) = (0_u32, 0_u32);

    let mut prediction_buckets_positive = vec![Vec::new(); 6];
    let mut prediction_buckets_negative = vec![Vec::new(); 6];
    const PREDICTION_MAX_POSITIVE: f64 = 0.25;
    const PREDICTION_MAX_NEGATIVE: f64 = 0.45;
    const PREDICTION_MIN: f64 = 0.03;

    if let Some(global) = predefined_global.get(&HOLD_TIME).copied() {
        OrderBookValues::calculate_window_values(&mut all_values, global.derivatives_normalize, global.trend_tau);
        let mut train_diapason = TimeDiapason::new(diapason.from, diapason.from + TimeDelta::minutes(TRAIN_DIAPASON));
        while train_diapason.to <= diapason.to - TimeDelta::minutes(TEST_DIAPASON) {
            let test_diapason = TimeDiapason::new(train_diapason.to + TimeDelta::seconds(1), train_diapason.to + TimeDelta::minutes(TEST_DIAPASON));
            info!("Train diapason {train_diapason:?}, test diapason {test_diapason:?}");

            let (train_values, train_horizons) = filter_values_with_horizons(&all_values, &horizons, train_diapason);
            let (test_values, test_horizons) = filter_values_with_horizons(&all_values, &horizons, test_diapason);

            let mut classificator_positive = GradientBoost::new(PredictionType::ClassificationPositive(THRESHOLD));
            classificator_positive.train(&train_values, &train_horizons);
            let mut classificator_negative = GradientBoost::new(PredictionType::ClassificationNegative(-THRESHOLD));
            classificator_negative.train(&train_values, &train_horizons);

            let predictions_positive = classificator_positive.test(&test_values);
            info!(
                "Predictions positive min {}, max {}",
                predictions_positive.iter().copied().reduce(f64::min).unwrap(),
                predictions_positive.iter().copied().reduce(f64::max).unwrap(),
            );
            let predictions_negative = classificator_negative.test(&test_values);
            info!(
                "Predictions negative min {}, max {}",
                predictions_negative.iter().copied().reduce(f64::min).unwrap(),
                predictions_negative.iter().copied().reduce(f64::max).unwrap(),
            );

            for bucket_index in 0..prediction_buckets_positive.len() {
                let prediction_from = PREDICTION_MIN + bucket_index as f64 * (PREDICTION_MAX_POSITIVE - PREDICTION_MIN) / prediction_buckets_positive.len() as f64;
                let prediction_to = PREDICTION_MIN + (bucket_index + 1) as f64 * (PREDICTION_MAX_POSITIVE - PREDICTION_MIN) / prediction_buckets_positive.len() as f64;
                prediction_buckets_positive[bucket_index].extend(
                    predictions_positive.iter()
                        .zip(test_horizons.iter())
                        .filter(|(p, _)| *p >= &prediction_from && *p < &prediction_to)
                        .map(|(_, h)| *h)
                );
            }

            for bucket_index in 0..prediction_buckets_negative.len() {
                let prediction_from = PREDICTION_MIN + bucket_index as f64 * (PREDICTION_MAX_NEGATIVE - PREDICTION_MIN) / prediction_buckets_negative.len() as f64;
                let prediction_to = PREDICTION_MIN + (bucket_index + 1) as f64 * (PREDICTION_MAX_NEGATIVE - PREDICTION_MIN) / prediction_buckets_negative.len() as f64;
                prediction_buckets_negative[bucket_index].extend(
                    predictions_negative.iter()
                        .zip(test_horizons.iter())
                        .filter(|(p, _)| *p >= &prediction_from && *p < &prediction_to)
                        .map(|(_, h)| *h)
                );
            }

            predictions_positive.iter()
                .zip(predictions_negative.iter())
                .zip(test_horizons.iter()).for_each(|((positive, negative), horizon)| {
                if *positive > PREDICTION_MAX_POSITIVE * 0.65 && *negative < PREDICTION_MAX_NEGATIVE * 0.25 {
                    if *horizon > 0.1 {
                        correct += 1;
                    } else {
                        incorrect += 1;
                    }
                } else if *horizon >= THRESHOLD {
                    missed += 1;
                }
                total += 1;
            });

            let categories = test_horizons.iter()
                .map(|horizon| if *horizon >= THRESHOLD {0.2} else {0.0})
                .collect::<Vec<_>>();
            let correlation = correlation::pearsonr(&categories, &predictions_positive);
            info!("Correlation {}", correlation);

            train_diapason.from += TimeDelta::minutes(TEST_DIAPASON);
            train_diapason.to += TimeDelta::minutes(TEST_DIAPASON);
        }

        for i in 0..prediction_buckets_positive.len() {
            if prediction_buckets_positive[i].len() > 2 {
                let prediction_from = PREDICTION_MIN + i as f64 * (PREDICTION_MAX_POSITIVE - PREDICTION_MIN) / prediction_buckets_positive.len() as f64;
                let prediction_to = PREDICTION_MIN + (i + 1) as f64 * (PREDICTION_MAX_POSITIVE - PREDICTION_MIN) / prediction_buckets_positive.len() as f64;
                info!(
                    "predictions {prediction_from:.3} - {prediction_to:.3}: {}, horizon > threshold {:.2}%, horizon < -threshold {:.2}%, mean {}, median {}, p05-p95 {:?}",
                    prediction_buckets_positive[i].len(),
                    prediction_buckets_positive[i].iter().filter(|h| **h > THRESHOLD).count() as f64 * 100.0 / prediction_buckets_positive[i].len() as f64,
                    prediction_buckets_positive[i].iter().filter(|h| **h < -THRESHOLD).count() as f64 * 100.0 / prediction_buckets_positive[i].len() as f64,
                    mean(&prediction_buckets_positive[i]), median(&prediction_buckets_positive[i]), p05_p95(&prediction_buckets_positive[i])
                );
            }
        }

        for i in 0..prediction_buckets_negative.len() {
            if prediction_buckets_negative[i].len() > 2 {
                let prediction_from = PREDICTION_MIN + i as f64 * (PREDICTION_MAX_NEGATIVE - PREDICTION_MIN) / prediction_buckets_negative.len() as f64;
                let prediction_to = PREDICTION_MIN + (i + 1) as f64 * (PREDICTION_MAX_NEGATIVE - PREDICTION_MIN) / prediction_buckets_negative.len() as f64;
                info!(
                    "predictions {prediction_from:.3} - {prediction_to:.3}: {}, horizon > threshold {:.2}%, horizon < -threshold {:.2}%, mean {}, median {}, p05-p95 {:?}",
                    prediction_buckets_negative[i].len(),
                    prediction_buckets_negative[i].iter().filter(|h| **h > THRESHOLD).count() as f64 * 100.0 / prediction_buckets_negative[i].len() as f64,
                    prediction_buckets_negative[i].iter().filter(|h| **h < -THRESHOLD).count() as f64 * 100.0 / prediction_buckets_negative[i].len() as f64,
                    mean(&prediction_buckets_negative[i]), median(&prediction_buckets_negative[i]), p05_p95(&prediction_buckets_negative[i])
                );
            }
        }

        info!("Correct {correct}, incorrect {incorrect} {:.2}%", 100.0 * correct as f64 / (correct + incorrect) as f64);
        info!("Missed {missed}, total {total}");

        /*
        let char_filename = format!("hold_{}.png", hold_time);
        if let Err(err) = draw_chart(&test_values, &test_horizons, &predictions, &char_filename) {
            error!("{err}");
        } else {
            info!("Chart saved to {char_filename}");
        }*/
    }

    Ok(())
}

pub fn filter_values_with_horizons(all_values: &[OrderBookValues], all_horizons: &[f64], diapason: TimeDiapason) -> (Vec<OrderBookValues>, Vec<f64>) {
    all_values.iter().zip(all_horizons.iter())
        .filter(|(v, _)| v.time >= diapason.from && v.time < diapason.to)
        .map(|(v, h)| (v.clone(), *h))
        .collect()
}

pub fn scale_prediction(data: &[f64], scale: f64) -> Option<Vec<f64>> {
    Some(data.iter().copied().map(|x| {
        x.clamp(-1.0, 1.0) * scale
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
    diapason: TimeDiapason,
    id: Option<i32>,
    db: Option<&Db>,
) -> Result<Vec<OrderBookValues>, CommonError> {
    let order_books = get_order_books(ticker, diapason, id, db).await?;
    let values = order_books
        .iter()
        .filter_map(OrderBookValues::calculate)
        .collect::<Vec<_>>();
    Ok(values)
}

async fn get_order_books(
    ticker: &str,
    diapason: TimeDiapason,
    id: Option<i32>,
    db: Option<&Db>,
) -> Result<Vec<OrderBook>, CommonError> {
    if let Some(order_books1) = order_book_cache::read(ticker, diapason, true)? {
        Ok(order_books1)
    } else {
        let order_books = db.unwrap().get_order_books(id.unwrap(), diapason).await?;
        order_book_cache::write(ticker, diapason, &order_books, true)?;
        Ok(order_books)
    }
}
