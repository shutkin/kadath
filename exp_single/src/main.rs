use std::io::Write;
mod orderbook_values;
mod signal_calculator;
mod charts;
mod global_optimizer;
mod gradient_boost;

use crate::global_optimizer::{global_optimize, GlobalParams};
use crate::gradient_boost::{optimize_gbdt, optimize_light_gbm, GradientBoost, MachineConfig, PredictionType, GbdtConfig, test_features_rolling_window};
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, Duration, TimeDelta, Utc};
use db::Db;
use log::{LevelFilter, debug, error, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::math_util::median;
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use simplelog::SimpleLogger;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::thread;
use gbdt::config::Config;
use serde_json::Value;

const HOLD_TIME: i32 = 4_000;
const TRAIN_DIAPASON: i64 = 5 * 60;
const TEST_DIAPASON: i64 = 45;
const THRESHOLD_POSITIVE: f64 = 5.0;
const THRESHOLD_NEGATIVE: f64 = 0.0;

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let ticker = "SiZ6";
    let positive = PredictionType::ClassificationPositive(THRESHOLD_POSITIVE);
    let negative = PredictionType::ClassificationNegative(THRESHOLD_NEGATIVE);

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    let inst_id = find_instrument_id(&all_instruments, ticker).expect("Instrument not found");
    let diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-22T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-22T13:00:00Z")?.to_utc(),
    );

    let mut values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
    info!("Got {} order book values", values.len());
    OrderBookValues::calculate_window_values(&mut values, 90.0, 300.0);

    test_features_rolling_window(90, 300, &values);

    let horizons = build_horizon(&values, HOLD_TIME);

    let mut predictions_with_horizons = Vec::new();
    let mut train_diapason = TimeDiapason::new(diapason.from, diapason.from + TimeDelta::minutes(TRAIN_DIAPASON));

    let (mut config_positive, mut config_negative) = (None, Option::<MachineConfig>::None);

    let optimizer_train_diapason = TimeDiapason::new(diapason.from, diapason.from + TimeDelta::minutes(TRAIN_DIAPASON));
    let optimizer_test_diapason = TimeDiapason::new(optimizer_train_diapason.to + TimeDelta::seconds(1), optimizer_train_diapason.to + TimeDelta::minutes(30));
    let (train_values, train_horizons) = filter_values_with_horizons(&values, &horizons, optimizer_train_diapason);
    let (test_values, test_horizons) = filter_values_with_horizons(&values, &horizons, optimizer_test_diapason);
    let train_values_arc = Arc::new(train_values);
    let train_horizons_arc = Arc::new(train_horizons);
    let test_values_arc = Arc::new(test_values);
    let test_horizons_arc = Arc::new(test_horizons);

    let train_values_clone = train_values_arc.clone();
    let train_horizons_clone = train_horizons_arc.clone();
    let test_values_clone = test_values_arc.clone();
    let test_horizons_clone = test_horizons_arc.clone();
    let h1 = thread::spawn(move || {
        optimize_gbdt(positive, train_values_clone, train_horizons_clone, test_values_clone, test_horizons_clone)
    });

    let train_values_clone = train_values_arc.clone();
    let train_horizons_clone = train_horizons_arc.clone();
    let test_values_clone = test_values_arc.clone();
    let test_horizons_clone = test_horizons_arc.clone();
    let h2 = thread::spawn(move || {
        optimize_gbdt(negative, train_values_clone, train_horizons_clone, test_values_clone, test_horizons_clone)
    });

    let config = h1.join().unwrap()?;
    info!("Best positive config: {config:?}");
    config_positive = Some(config);

    let config = h2.join().unwrap()?;
    info!("Best negative config: {config:?}");
    config_negative = Some(config);

    train_diapason.from += TimeDelta::minutes(30);
    train_diapason.to += TimeDelta::minutes(30);

    while train_diapason.to <= diapason.to - TimeDelta::minutes(TEST_DIAPASON) {
        let test_diapason = TimeDiapason::new(train_diapason.to + TimeDelta::seconds(1), train_diapason.to + TimeDelta::minutes(TEST_DIAPASON));
        info!("Train diapason {train_diapason:?}, test diapason {test_diapason:?}");

        let (train_values, train_horizons) = filter_values_with_horizons(&values, &horizons, train_diapason);
        let (test_values, test_horizons) = filter_values_with_horizons(&values, &horizons, test_diapason);

        if let Some(config_positive) = &config_positive && let Some(config_negative) = &config_negative {
            let classificator_positive = GradientBoost::train(positive, config_positive, &train_values, &train_horizons)?;
            let classificator_negative = GradientBoost::train(negative, config_negative, &train_values, &train_horizons)?;

            let predictions_positive = classificator_positive.test(&test_values)?;
            let predictions_negative = classificator_negative.test(&test_values)?;

            let n = predictions_positive.len();
            info!("{n} predictions");

            // Helper для получения нормированных рангов (от 0.0 до 1.0)
            // Принимает вектор значений и признак инверсии (для негативного классификатора)
            let get_normalized_ranks = |preds: &[f64], invert: bool| -> Vec<f64> {
                // Храним пары (исходный_индекс, значение)
                let mut indexed: Vec<(usize, f64)> = preds.iter()
                    .enumerate()
                    .map(|(i, &v)| if invert { (i, 1.0 - v) } else { (i, v) })
                    .collect();

                // Сортируем по значению (от худшего к лучшему)
                indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

                // Записываем нормированный ранг на позицию исходного индекса
                let mut ranks = vec![0.0; n];
                for (rank_idx, &(original_idx, _)) in indexed.iter().enumerate() {
                    // Ранг от 0.0 (худший результат) до 1.0 (лучший результат)
                    ranks[original_idx] = rank_idx as f64 / (n as f64);
                }
                ranks
            };

            let ranks_pos = get_normalized_ranks(&predictions_positive, true);
            debug!("ranks_pos: {:?}", &ranks_pos[0..32]);
            let ranks_neg = get_normalized_ranks(&predictions_negative, false);
            debug!("ranks_neg: {:?}", &ranks_neg[0..32]);

            // 3. Комбинируем ранги через перемножение
            let predictions: Vec<f64> = ranks_pos.iter()
                .zip(ranks_neg.iter())
                .map(|(&r_pos, &r_neg)| r_pos * (0.5 + 0.5 * r_neg))
                .collect();
            predictions_with_horizons.extend(
                predictions.into_iter().zip(test_horizons.iter().copied())
            );
        }

        train_diapason.from += TimeDelta::minutes(TEST_DIAPASON);
        train_diapason.to += TimeDelta::minutes(TEST_DIAPASON);
    }

    let min = predictions_with_horizons.iter().map(|(p, _)| *p).reduce(f64::min).unwrap();
    let max = predictions_with_horizons.iter().map(|(p, _)| *p).reduce(f64::max).unwrap();
    info!("Min {min:?}, Max {max:?}");

    for i in 0..32 {
        let threshold = min + (i + 1) as f64 * (max - min) / 33.0;
        let (mut correct, mut incorrect) = (0, 0);
        let (mut commission, mut pnl) = (0.0, 0.0);
        for (prediction, horizon) in &predictions_with_horizons {
            if *prediction > threshold {
                if *horizon < 0.0 {
                    correct += 1;
                } else {
                    incorrect += 1;
                }

                pnl += -horizon;
                commission += 1.0;
            }
        }
        info!("Threshold {threshold}: {:.2}%, PnL {pnl}, commission {commission}, gross {}",
            correct as f64 * 100.0 / (correct + incorrect) as f64, pnl - commission);
    }

    Ok(())
}

#[tokio::main]
async fn _main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let ticker = "SiZ6";
    let positive = PredictionType::ClassificationPositive(THRESHOLD_POSITIVE);
    let negative = PredictionType::ClassificationNegative(THRESHOLD_NEGATIVE);

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    let inst_id = find_instrument_id(&all_instruments, ticker).expect("Instrument not found");
    let diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-28T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-28T20:00:00Z")?.to_utc(),
    );

    let values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
    info!("Got {} order book values", values.len());
    //OrderBookValues::calculate_window_values(&mut values, 90.0, 300.0);
    let horizons = build_horizon(&values, 2000);

    let config = MachineConfig::GBDT(GbdtConfig { max_depth: 2, iterations: 55, shrinkage: 0.005131322352883715, min_leaf_size: 458, data_sample_ratio: 0.9121054660755452, feature_sample_ratio: 0.7838238646796116 });
    let train_diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-28T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-28T15:00:00Z")?.to_utc(),
    );
    let test_diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-28T15:00:01Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-28T20:00:00Z")?.to_utc(),
    );
    let params = global_optimize(config, PredictionType::ClassificationPositive(1.5), &values, &horizons, train_diapason, test_diapason);
    info!("Optimized {params:#?}");

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

fn export_to_csv(filename: &str, order_book_values: &[OrderBookValues], horizons: &[f64]) -> Result<usize, CommonError> {
    let mut file = File::create(filename)?;

    writeln!(file, "timestamp_ms,imbalance_3,imbalance_5,imbalance_10,imbalance_20,imbalance_50,imbalance_derivative_norm_3,imbalance_derivative_norm_5,imbalance_derivative_norm_10,imbalance_derivative_norm_20,imbalance_derivative_norm_50,mid_price_minus_trend,mid_price_minus_trend_derivative_norm,horizon")?;

    let mut written_count = 0;

    for (i, book) in order_book_values.iter().enumerate() {
        let timestamp_ms = book.time.timestamp_millis();
        let imbalances = &book.imbalances;
        let imbalances_derivatives = &book.imbalances_derivative;
        writeln!(file, "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                 timestamp_ms,
                 imbalances[0], imbalances[1], imbalances[2], imbalances[3], imbalances[4],
                 imbalances_derivatives[0], imbalances_derivatives[1], imbalances_derivatives[2], imbalances_derivatives[3], imbalances_derivatives[4],
                 book.trend_deviation,
                 book.trend_deviation_derivative,
                 horizons[i],
        )?;

        written_count += 1;
    }

    Ok(written_count)
}
