mod orderbook_values;
mod charts;
mod global_optimizer;
mod gradient_boost;
mod nelder_mead;

use crate::global_optimizer::global_optimize;
use crate::gradient_boost::{GbdtConfig, GradientBoost, MachineConfig, PredictionType, optimize_gbdt};
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{LevelFilter, debug, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::math_util::OutputType::{DerivativeNorm, Raw, StandardDeviation, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{percentile, FeaturesRollingWindow, OutputType};
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use simplelog::SimpleLogger;
use std::sync::Arc;
use std::thread;
use crate::nelder_mead::calibrate_linear_signal;

const HOLD_TIME: i32 = 4_000;
const TRAIN_DIAPASON: i64 = 5 * 60;
const TEST_DIAPASON: i64 = 45;
const THRESHOLD_POSITIVE: f32 = 5.0;
const THRESHOLD_NEGATIVE: f32 = 0.0;

const FEATURES_NAMES: [&str; 7] = ["mid", "spread", "i3", "i5", "i10", "i20", "i50"];
const FEATURES_OUTPUT: [OutputType; 13] = [
    TrendDeviation("mid"),
    TrendDeviationDerivativeNorm("mid"),
    StandardDeviation("spread"),
    Raw("i3"),
    Raw("i5"),
    Raw("i10"),
    Raw("i20"),
    Raw("i50"),
    DerivativeNorm("i3"),
    DerivativeNorm("i5"),
    DerivativeNorm("i10"),
    DerivativeNorm("i20"),
    DerivativeNorm("i50"),
];


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
        DateTime::parse_from_rfc3339("2026-10-22T11:00:00Z")?.to_utc(),
    );
    let train_end = DateTime::parse_from_rfc3339("2026-10-22T10:00:00Z")?.to_utc();
    let values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
    info!("Got {} order book values", values.len());
    let (train_values, test_values, train_horizons, test_horizons) =
        features_with_horizons(&values, train_end, 57.28, 614.61, TimeDelta::milliseconds(HOLD_TIME as i64));

    let mut train_diapason = TimeDiapason::new(diapason.from, diapason.from + TimeDelta::minutes(TRAIN_DIAPASON));

    let (mut config_positive, mut config_negative) = (None, Option::<MachineConfig>::None);

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


    if let Some(config_positive) = &config_positive && let Some(config_negative) = &config_negative {
        let diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-10-22T06:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-10-22T12:00:00Z")?.to_utc(),
        );
        let train_end = DateTime::parse_from_rfc3339("2026-10-22T11:00:00Z")?.to_utc();
        let values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
        info!("Got {} order book values", values.len());
        let (train_values, test_values, train_horizons, test_horizons) =
            features_with_horizons(&values, train_end, 57.28, 614.61, TimeDelta::milliseconds(HOLD_TIME as i64));
        let classificator_positive = GradientBoost::train(positive, config_positive, &train_values, &train_horizons)?;
        let classificator_negative = GradientBoost::train(negative, config_negative, &train_values, &train_horizons)?;
        let linear = calibrate_linear_signal(None, &train_values, &train_horizons).ok_or("Unable to calibrate linear signal")?;

        info!("Train Linear model...");
        let predictions_linear = linear.test(&test_values);
        info!("Train UP model...");
        let predictions_positive = classificator_positive.test(&test_values)?;
        info!("Train DOWN model...");
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
        let predictions_gbm: Vec<f64> = ranks_pos.iter()
            .zip(ranks_neg.iter())
            .map(|(&r_pos, &r_neg)| r_pos * (0.5 + 0.5 * r_neg))
            .collect();

        const PERCENTILES: [f64; 4] = [0.97, 0.99, 0.993, 0.997];

        for p_linear in PERCENTILES {
            let threshold_linear = percentile(&predictions_linear, p_linear);
            for p_up in PERCENTILES {
                let threshold_up = percentile(&predictions_gbm, p_up);

                let pnl_vec = predictions_linear.iter().zip(predictions_gbm.iter()).zip(test_horizons.iter())
                    .filter(|((l, u), _)| **l >= threshold_linear && **u >= threshold_up)
                    .map(|(_, pnl)| *pnl)
                    .collect::<Vec<_>>();
                let total_pnl = pnl_vec.iter().copied().sum::<f32>();
                info!("Linear {p_linear}, Up {p_up} => {total_pnl}, {}", pnl_vec.len());
            }
        }
    }

    Ok(())
}

fn features_with_horizons(
    events: &[OrderBookValues],
    train_end: DateTime<Utc>,
    derivatives_normalize: f64,
    trend_tau: f64,
    hold: TimeDelta,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let mut window1 = FeaturesRollingWindow::new(
        TimeDelta::milliseconds((1000.0 * derivatives_normalize) as i64),
        TimeDelta::milliseconds((1000.0 * trend_tau) as i64),
        &FEATURES_NAMES,
        &FEATURES_OUTPUT,
    );
    let mut window2 = FeaturesRollingWindow::new(
        TimeDelta::milliseconds((1000.0 * derivatives_normalize) as i64),
        TimeDelta::milliseconds((1000.0 * trend_tau) as i64),
        &FEATURES_NAMES,
        &FEATURES_OUTPUT,
    );

    let mut train_features = Vec::new();
    let mut test_features = Vec::new();
    let mut train_horizons = Vec::new();
    let mut test_horizons = Vec::new();

    for (i, values) in events.iter().enumerate() {
        let features = window1.process(values.to_features());

        if let Some(h_values) = find_value_on_horizon(events, i, values.time + hold) {
            let pnl = (h_values.bid - values.ask) as f32;

            if values.time + hold <= train_end {
                train_features.push(features);
                train_horizons.push(pnl);
            } else if values.time > train_end {
                test_features.push(features);
                test_horizons.push(pnl);
            }
        }
    }
    (train_features, test_features, train_horizons, test_horizons)
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
