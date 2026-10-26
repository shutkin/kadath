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
use log::{LevelFilter, debug, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::math_util::median;
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use simplelog::SimpleLogger;
use std::collections::HashMap;
use std::thread;

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

    let mut values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
    info!("Got {} order book values", values.len());
    OrderBookValues::calculate_window_values(&mut values, 90.0, 300.0);
    let horizons = build_horizon(&values, 2000);

    let mut predictions_with_horizons = Vec::new();
    let mut train_diapason = TimeDiapason::new(diapason.from, diapason.from + TimeDelta::minutes(TRAIN_DIAPASON));
    while train_diapason.to <= diapason.to - TimeDelta::minutes(TEST_DIAPASON) {
        let test_diapason = TimeDiapason::new(train_diapason.to + TimeDelta::seconds(1), train_diapason.to + TimeDelta::minutes(TEST_DIAPASON));
        info!("Train diapason {train_diapason:?}, test diapason {test_diapason:?}");

        let (train_values, train_horizons) = filter_values_with_horizons(&values, &horizons, train_diapason);
        let (test_values, test_horizons) = filter_values_with_horizons(&values, &horizons, test_diapason);

        let mut classificator_positive = GradientBoost::new(PredictionType::ClassificationPositive(THRESHOLD), None);
        classificator_positive.train(&train_values, &train_horizons);

        let mut classificator_negative = GradientBoost::new(PredictionType::ClassificationNegative(-THRESHOLD), None);
        classificator_negative.train(&train_values, &train_horizons);

        let predictions_positive = classificator_positive.test(&test_values);
        let predictions_negative = classificator_negative.test(&test_values);

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

        // 2. Считаем ранги:
        // Для позитивного: чем больше P_positive, тем выше ранг
        let ranks_pos = get_normalized_ranks(&predictions_positive, false);
        info!("ranks_pos: {:?}", &ranks_pos[0..32]);
        // Для негативного: чем МЕНЬШЕ P_negative (больше 1 - P_negative), тем выше ранг
        let ranks_neg = get_normalized_ranks(&predictions_negative, true);
        info!("ranks_neg: {:?}", &ranks_neg[0..32]);

        // 3. Комбинируем ранги через перемножение
        let predictions: Vec<f64> = ranks_pos.iter()
            .zip(ranks_neg.iter())
            .map(|(&r_pos, &r_neg)| r_pos * r_neg)
            .collect();
        predictions_with_horizons.extend(
            predictions.into_iter().zip(test_horizons.into_iter())
        );

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
                if *horizon > 0.0 {
                    correct += 1;
                } else {
                    incorrect += 1;
                }

                pnl += horizon;
                commission += 1.0;
            }
        }
        info!("Threshold {threshold}: {:.2}%, PnL {pnl}, commission {commission}, gross {}",
            correct as f64 * 100.0 / (correct + incorrect) as f64, pnl - commission);
    }

    Ok(())
}

const HOLD_TIME: i32 = 2_000;
const TRAIN_DIAPASON: i64 = 5 * 60;
const TEST_DIAPASON: i64 = 30;
const THRESHOLD: f64 = 5.0;

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
        DateTime::parse_from_rfc3339("2026-10-22T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-22T20:00:00Z")?.to_utc(),
    );

    let mut all_values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
    info!("Got {} order book values", all_values.len());

    let predefined_global = HashMap::<i32, GlobalParams>::from([
        (1000, GlobalParams { derivatives_normalize: 200.0, trend_tau: 180.0 }),
        (2000, GlobalParams { derivatives_normalize: 90.0, trend_tau: 300.0 }),
        (3000, GlobalParams { derivatives_normalize: 200.0, trend_tau: 180.0 }),
    ]);

    if let Some(global) = predefined_global.get(&HOLD_TIME).copied() {
        OrderBookValues::calculate_window_values(&mut all_values, global.derivatives_normalize, global.trend_tau);
        let horizons = build_horizon(&all_values, HOLD_TIME);

        let mut cfg = (2, 100, 0.05, 300);

        for is_positive in [true, false] {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_cfg = None;

            for depth in [2, 3] {
                cfg.0 = depth;
                for iterations in [50, 150, 300] {
                    cfg.1 = iterations;
                    for min_leaf in [100, 200, 400] {
                        cfg.3 = min_leaf;
                        let handles = if is_positive {
                            [0.0125, 0.025, 0.05, 0.1]
                        } else {
                            [0.025, 0.05, 0.1, 0.2]
                        }.into_iter().map(|shrinkage| {
                            cfg.2 = shrinkage;
                            let values_clone = all_values.clone();
                            let horizons_clone = horizons.clone();
                            thread::spawn(move || {
                                (score(diapason, &values_clone, &horizons_clone, is_positive, cfg), cfg)
                            })
                        }).collect::<Vec<_>>();
                        for handle in handles {
                            let (score, cfg) = handle.join().unwrap();
                            if score > best_score {
                                best_score = score;
                                best_cfg = Some(cfg);
                            }
                        }
                    }
                }
            }
            info!("{} {best_cfg:?} {best_score}", if is_positive { "positive" } else { "negative" });
        }
    }

    Ok(())
}

fn score(
    diapason: TimeDiapason,
    all_values: &[OrderBookValues],
    horizons: &[f64],
    is_positive: bool,
    cfg: (u32, usize, f32, usize)
) -> f64 {
    info!("Test {} {cfg:?}", if is_positive { "positive" } else { "negative" });

    let mut predictions_with_horizons = Vec::new();
    let mut train_diapason = TimeDiapason::new(diapason.from, diapason.from + TimeDelta::minutes(TRAIN_DIAPASON));
    while train_diapason.to <= diapason.to - TimeDelta::minutes(TEST_DIAPASON) {
        let test_diapason = TimeDiapason::new(train_diapason.to + TimeDelta::seconds(1), train_diapason.to + TimeDelta::minutes(crate::TEST_DIAPASON));
        debug!("Train diapason {train_diapason:?}, test diapason {test_diapason:?}");

        let (train_values, train_horizons) = filter_values_with_horizons(all_values, horizons, train_diapason);
        let (test_values, test_horizons) = filter_values_with_horizons(all_values, horizons, test_diapason);

        let mut classificator = if is_positive {
            GradientBoost::new(PredictionType::ClassificationPositive(THRESHOLD), Some(cfg))
        } else {
            GradientBoost::new(PredictionType::ClassificationNegative(-THRESHOLD), Some(cfg))
        };
        classificator.train(&train_values, &train_horizons);
        let predictions = classificator.test(&test_values);

        predictions_with_horizons.extend(
            predictions.into_iter().zip(test_horizons.into_iter())
        );

        train_diapason.from += TimeDelta::minutes(TEST_DIAPASON);
        train_diapason.to += TimeDelta::minutes(TEST_DIAPASON);
    }

    let prediction_min = predictions_with_horizons.iter().map(|(p, _)| *p).reduce(f64::min).unwrap();
    let prediction_max = predictions_with_horizons.iter().map(|(p, _)| *p).reduce(f64::max).unwrap();
    const BUCKETS: usize = 16;
    let mut horizon_buckets = vec![Vec::new(); BUCKETS];
    for i in 0..BUCKETS {
        let prediction_from = prediction_min + (prediction_max - prediction_min) * i as f64 / BUCKETS as f64;
        let prediction_to = prediction_min + (prediction_max - prediction_min) * (i + 1) as f64 / BUCKETS as f64;
        horizon_buckets[i].extend(
            predictions_with_horizons.iter()
                .filter(|(p, _)| *p >= prediction_from && *p < prediction_to)
                .map(|(_, h)| *h)
                .collect::<Vec<f64>>()
        );
    }

    let medians_with_size = horizon_buckets.iter()
        .map(|horizons| (median(horizons), horizons.len()))
        .collect::<Vec<(f64, usize)>>();
    let score = medians_with_size.iter()
        .filter(|(median, _)| is_positive && *median >= 0.0 || !is_positive && *median <= 0.0)
        .map(|(median, len)| (median.abs() + 1.0).ln() * *len as f64)
        .sum::<f64>();
    info!("{cfg:?}: {score} {medians_with_size:?}");
    score
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
