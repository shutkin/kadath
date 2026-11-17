mod orderbook_values;
mod charts;
mod global_optimizer;
mod gradient_boost;
mod nelder_mead;
mod trend_filter;

use crate::global_optimizer::{GlobalParams, global_optimize};
use crate::gradient_boost::{optimize_gbdt, optimize_light_gbm, GbdtConfig, GradientBoost, MachineConfig, PredictionType};
use crate::nelder_mead::calibrate_linear_signal;
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{LevelFilter, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::math_util::OutputType::{DerivativeNorm, Raw, StandardDeviation, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{get_normalized_ranks, percentile, FeaturesRollingWindow, OutputType};
use model::{DataSource, Instrument, OrderBook, TradeDirection, order_book_cache};
use simplelog::SimpleLogger;
use std::sync::Arc;
use std::thread;
use crate::trend_filter::TrendFilter;

const HOLD_TIME: i32 = 14144;
const TRAIN_DIAPASON: i64 = 5 * 60;
const TEST_DIAPASON: i64 = 45;
const THRESHOLD_POSITIVE: f32 = 4.88;
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

const OPTIMIZE_GBM: bool = false;
const OPTIMIZE_GLOBAL: bool = false;

const DIRECTION: TradeDirection = TradeDirection::Buy;

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

    //let global_params = GlobalParams { derivatives_normalize: 185.1804471910355, trend_tau: 207.2308381991384 };
    //let global_params = GlobalParams { derivatives_normalize: 57.28, trend_tau: 614.61 };
    let global_params = GlobalParams { derivatives_normalize: 434.58, trend_tau: 49.9 };

    let (mut config_positive, mut config_negative) = (None, Option::<MachineConfig>::None);

    if OPTIMIZE_GBM {
        let diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-11-13T05:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-11-13T11:00:00Z")?.to_utc(),
        );
        let train_end = DateTime::parse_from_rfc3339("2026-11-13T10:00:00Z")?.to_utc();
        let values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
        info!("Got {} order book values", values.len());
        let fh =
            get_features_with_horizons(&values, train_end, global_params, TimeDelta::milliseconds(HOLD_TIME as i64), DIRECTION);

        let train_values_arc = Arc::new(fh.train_features);
        let train_horizons_arc = Arc::new(fh.train_horizons);
        let test_values_arc = Arc::new(fh.test_features);
        let test_horizons_arc = Arc::new(fh.test_horizons);

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
    } else {
        config_positive = Some(MachineConfig::GBDT(GbdtConfig { max_depth: 7, iterations: 98, shrinkage: 0.0053386633647191485, min_leaf_size: 378, data_sample_ratio: 0.7917056472120758, feature_sample_ratio: 0.5066184969063494 }));
        config_negative = Some(MachineConfig::GBDT(GbdtConfig { max_depth: 6, iterations: 95, shrinkage: 0.005878415064157608, min_leaf_size: 243, data_sample_ratio: 0.6650489555889336, feature_sample_ratio: 0.8102900523207806 }));
    }

    if let Some(config_positive) = &config_positive && let Some(config_negative) = &config_negative {
        let diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-11-13T06:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-11-13T13:00:00Z")?.to_utc(),
        );
        let train_end = DateTime::parse_from_rfc3339("2026-11-13T11:00:00Z")?.to_utc();
        let values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
        info!("Got {} order book values", values.len());

        if OPTIMIZE_GLOBAL {
            let global = global_optimize(config_positive.clone(), config_negative.clone(), positive, &values, train_end, HOLD_TIME as u16);
            info!("{global:?}");
        }

        let fh = get_features_with_horizons(&values, train_end, global_params, TimeDelta::milliseconds(HOLD_TIME as i64), DIRECTION);

        info!("Train Linear model...");
        let linear = calibrate_linear_signal(None, &fh.train_features, &fh.train_horizons).ok_or("Unable to calibrate linear signal")?;
        let predictions_linear = linear.test(&fh.test_features);
        info!("Train UP model...");
        let classificator_positive = GradientBoost::train(positive, config_positive, &fh.train_features, &fh.train_horizons)?;
        let predictions_positive = classificator_positive.test(&fh.test_features)?;
        info!("Train DOWN model...");
        let classificator_negative = GradientBoost::train(negative, config_negative, &fh.train_features, &fh.train_horizons)?;
        let predictions_negative = classificator_negative.test(&fh.test_features)?;
        info!("{} predictions", predictions_linear.len());

        let ranks_linear = rolling_normalize(&predictions_linear, false);
        let ranks_pos = rolling_normalize(&predictions_positive, false);
        let ranks_neg = rolling_normalize(&predictions_negative, true);
        info!("{} ranked predictions", ranks_linear.len());

        let ranks_gbm: Vec<f64> = ranks_pos.iter()
            .zip(ranks_neg.iter())
            .map(|(&r_pos, &r_neg)| r_pos * (0.0 + 1.0 * r_neg))
            .collect();

        let mut best_score = 0.0;
        let mut best_trend_length = 0.0;
        let mut best_threshold_linear = 0.0;
        let mut best_threshold_gbm = 0.0;

        for trend_length in [0.0, 2.25, 5.0, 10.0] {
            info!("Trend Filter length {trend_length:?}");
            let mut trend_filter = TrendFilter::new(TradeDirection::Buy, trend_length);
            let trend_filters = trend_filter.test(&values[values.len() - fh.test_features.len()..]);
            
            const PERCENTILES: [f64; 6] = [0.9, 0.93, 0.96, 0.99, 0.993, 0.997];

            for p_linear in PERCENTILES {
                let threshold_linear = p_linear;// percentile(&predictions_linear, p_linear);
                for p_gbm in PERCENTILES {
                    let threshold_gbm = p_gbm;// percentile(&ranks_gbm, p_gbm);

                    let pnl_vec = trend_filters.iter().skip(NORMALIZE_WINDOW_SIZE)
                        .zip(ranks_linear.iter())
                        .zip(ranks_gbm.iter())
                        .zip(fh.test_horizons.iter().skip(NORMALIZE_WINDOW_SIZE))
                        .filter(|(((tf, l), gbm), _)| **tf && **l >= threshold_linear && **gbm >= threshold_gbm)
                        .map(|(_, pnl)| *pnl)
                        .collect::<Vec<_>>();
                    let total_pnl = pnl_vec.iter().copied().sum::<f32>();
                    info!("Linear {p_linear}, GBM {p_gbm} => {total_pnl}, {} ({})", pnl_vec.len(), total_pnl / pnl_vec.len() as f32);

                    let score = total_pnl - pnl_vec.len() as f32;
                    if score > best_score {
                        best_score = score;
                        best_trend_length = trend_length;
                        best_threshold_linear = threshold_linear;
                        best_threshold_gbm = threshold_gbm;
                    }
                }
            }
        }

        info!("Best score {best_score} on trend_length {best_trend_length}, threshold_linear {best_threshold_linear}, threshold_gbm {best_threshold_gbm}");
    }

    Ok(())
}

const NORMALIZE_WINDOW_SIZE: usize = 25000;

fn rolling_normalize(x: &[f64], invert: bool) -> Vec<f64> {
    let mut result = Vec::with_capacity(x.len() - NORMALIZE_WINDOW_SIZE);

    for i in 0..x.len() - NORMALIZE_WINDOW_SIZE {
        let ranks = get_normalized_ranks(&x[i..i + NORMALIZE_WINDOW_SIZE], invert);
        result.push(ranks[ranks.len() - 1]);
    }

    result
}

struct FeaturesWithHorizons {
    train_features: Vec<Vec<f32>>,
    test_features: Vec<Vec<f32>>,
    train_horizons: Vec<f32>,
    test_horizons: Vec<f32>,
    train_timestamps: Vec<DateTime<Utc>>,
    test_timestamps: Vec<DateTime<Utc>>,
}

fn get_features_with_horizons(
    events: &[OrderBookValues],
    train_end: DateTime<Utc>,
    global_params: GlobalParams,
    hold: TimeDelta,
    direction: TradeDirection,
) -> FeaturesWithHorizons {
    let mut window = FeaturesRollingWindow::new(
        TimeDelta::milliseconds((1000.0 * global_params.derivatives_normalize) as i64),
        TimeDelta::milliseconds((1000.0 * global_params.trend_tau) as i64),
        &FEATURES_NAMES,
        &FEATURES_OUTPUT,
    );

    let mut train_features = Vec::new();
    let mut test_features = Vec::new();
    let mut train_horizons = Vec::new();
    let mut test_horizons = Vec::new();
    let mut train_timestamps = Vec::new();
    let mut test_timestamps = Vec::new();

    for (i, values) in events.iter().enumerate() {
        let features = window.process(values.to_features());

        if let Some(h_values) = find_value_on_horizon(events, i, values.time + hold) {
            let pnl = match direction {
                TradeDirection::Buy => h_values.bid - values.ask,
                TradeDirection::Sell => values.bid - h_values.ask,
            } as f32;

            if values.time + hold <= train_end {
                train_features.push(features);
                train_horizons.push(pnl);
                train_timestamps.push(values.time);
            } else if values.time > train_end {
                test_features.push(features);
                test_horizons.push(pnl);
                test_timestamps.push(values.time);
            }
        }
    }

    FeaturesWithHorizons {
        train_features,
        test_features,
        train_horizons,
        test_horizons,
        train_timestamps,
        test_timestamps,
    }
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
    //let params = global_optimize(config, PredictionType::ClassificationPositive(1.5), &values, &horizons, train_diapason, test_diapason);
    //info!("Optimized {params:#?}");

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
