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
use log::{error, info, LevelFilter};
use model::common::{export_features_to_csv, CommonError, EmptyResult, PredictionDebugInfo, TimeDiapason};
use model::math_util::OutputType::{DerivativeNorm, Raw, StandardDeviation, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{FeaturesRollingWindow, OutputType, get_normalized_rank};
use model::{DataSource, Instrument, OrderBook, TradeDirection, order_book_cache};
use simplelog::SimpleLogger;
use std::sync::Arc;
use std::thread;
use crate::trend_filter::TrendFilter;

const HOLD_TIME: f64 = 8.0;
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

    //let global_params = GlobalParams { derivatives_normalize: 57.28, trend_tau: 614.61 };
    //let global_params = GlobalParams { derivatives_normalize: 154.06066359974238, trend_tau: 188.50253617850086 };
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
            get_features_with_horizons(&values, train_end, global_params, HOLD_TIME, 0.0, DIRECTION);

        let train_values_arc = Arc::new(fh.train_features);
        let train_horizons_arc = Arc::new(fh.train_horizons);
        let test_values_arc = Arc::new(fh.test_features);
        let test_horizons_arc = Arc::new(fh.test_horizons);

        let train_values_clone = train_values_arc.clone();
        let train_horizons_clone = train_horizons_arc.clone();
        let test_values_clone = test_values_arc.clone();
        let test_horizons_clone = test_horizons_arc.clone();
        let h1 = thread::spawn(move || {
            optimize_light_gbm(positive, train_values_clone, train_horizons_clone, test_values_clone, test_horizons_clone)
        });

        let train_values_clone = train_values_arc.clone();
        let train_horizons_clone = train_horizons_arc.clone();
        let test_values_clone = test_values_arc.clone();
        let test_horizons_clone = test_horizons_arc.clone();
        let h2 = thread::spawn(move || {
            optimize_light_gbm(negative, train_values_clone, train_horizons_clone, test_values_clone, test_horizons_clone)
        });

        let config = h1.join().unwrap()?;
        info!("Best positive config: {config}");
        config_positive = Some(config);

        let config = h2.join().unwrap()?;
        info!("Best negative config: {config}");
        config_negative = Some(config);
    } else {
        config_positive = Some(MachineConfig::LightGBM(serde_json::from_str(r#"{"bagging_fraction":0.7348924472483985,"bagging_freq":5,"boosting_type":"gbdt","feature_fraction":0.5598798946712412,"lambda_l1":0.00013137942540926634,"lambda_l2":0.01833097440970101,"learning_rate":0.008480676750000433,"max_depth":-1,"metric":"binary_logloss","min_child_samples":427,"n_jobs":-1,"num_iterations":2000,"num_leaves":12,"objective":"binary","random_state":42,"verbosity":-1}"#)?));
        config_negative = Some(MachineConfig::LightGBM(serde_json::from_str(r#"{"bagging_fraction":0.9299833245094198,"bagging_freq":3,"boosting_type":"gbdt","feature_fraction":0.9052532226346467,"lambda_l1":0.0007166624155583156,"lambda_l2":0.005054111952699828,"learning_rate":0.0051878076001226365,"max_depth":-1,"metric":"binary_logloss","min_child_samples":74,"n_jobs":-1,"num_iterations":2000,"num_leaves":16,"objective":"binary","random_state":42,"verbosity":-1}"#)?));
    }

    if let Some(config_positive) = &config_positive && let Some(config_negative) = &config_negative {
        let diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-11-13T06:48:20Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-11-13T13:58:32Z")?.to_utc(),
        );
        let train_end = DateTime::parse_from_rfc3339("2026-11-13T11:59:59Z")?.to_utc();
        let values = get_order_books_values(ticker, diapason, Some(inst_id), Some(&db)).await?;
        info!("Got {} order book values", values.len());

        if OPTIMIZE_GLOBAL {
            let global = global_optimize(config_positive.clone(), config_negative.clone(), positive, &values, train_end, HOLD_TIME);
            info!("{global:?}");
        }

        let fh = get_features_with_horizons(&values, train_end, global_params, HOLD_TIME, 0.0, DIRECTION);
        export_features_to_csv(&fh.train_features, &fh.train_horizons, "experiment_train_data.csv");

        let test_milliseconds = (fh.test_timestamps[fh.test_timestamps.len() - 1] - fh.test_timestamps[0]).num_milliseconds();
        let events_per_hold = HOLD_TIME * 1000.0 * values.len() as f64 / test_milliseconds as f64;
        info!("Events per hold: {events_per_hold}");

        info!("Train Linear model on {} values...", fh.train_features.len());
        let linear = calibrate_linear_signal(None, &fh.train_features, &fh.train_horizons, events_per_hold as usize)
            .ok_or("Unable to calibrate linear signal")?;
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
        info!("{} positive and {} negative ranked predictions", ranks_pos.len(), ranks_neg.len());

        let ranks_gbm: Vec<f64> = ranks_pos.iter()
            .zip(ranks_neg.iter())
            .map(|(&r_pos, &r_neg)| r_pos * (0.75 + 0.25 * r_neg))
            .collect();

        let mut best_score = 0.0;
        let mut best_trend_length = 0.0;
        let mut best_threshold_linear = 0.0;
        let mut best_threshold_gbm = 0.0;

        for trend_length in [0.0, 2.25, 5.0, 10.0] {
            info!("Trend Filter length {trend_length:?}");
            let fh = get_features_with_horizons(&values, train_end, global_params, HOLD_TIME, trend_length, DIRECTION);

            let mut predictions = Vec::with_capacity(fh.test_timestamps.len() - NORMALIZE_WINDOW_SIZE);
            for i in NORMALIZE_WINDOW_SIZE .. fh.test_timestamps.len() {
                let info = PredictionDebugInfo {
                    is_filtered_by_trend: fh.test_filtered_by_trend[i],
                    linear_raw: predictions_linear[i],
                    gbm_positive_raw: predictions_positive[i],
                    gbm_negative_raw: predictions_negative[i],
                    linear_ranked: ranks_linear[i - NORMALIZE_WINDOW_SIZE],
                    gbm_ranked: ranks_gbm[i - NORMALIZE_WINDOW_SIZE],
                };
                predictions.push(
                    (fh.test_timestamps[i], info, fh.test_horizons[i]),
                )
            }
            if let Err(err) = save_predictions_csv(&predictions, "experimental_predictions.csv") {
                error!("Error saving predictions csv: {err}");
            }

            const PERCENTILES: [f64; 6] = [0.9, 0.93, 0.96, 0.99, 0.993, 0.996];
            for threshold_linear in PERCENTILES {
                for threshold_gbm in PERCENTILES {
                    let pnls = predictions.iter()
                        .filter(|(_, info, _)| info.is_filtered_by_trend && info.linear_ranked >= threshold_linear && info.gbm_ranked >= threshold_gbm)
                        .map(|(_, _, horizon)| *horizon)
                        .collect::<Vec<_>>();
                    let total_pnl = pnls.iter().copied().sum::<f32>();
                    info!("Linear {threshold_linear}, GBM {threshold_gbm} => {total_pnl}, {} ({})", pnls.len(), total_pnl / pnls.len() as f32);

                    let score = total_pnl - pnls.len() as f32;
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
        let rank = get_normalized_rank(&x[i..i + NORMALIZE_WINDOW_SIZE], invert);
        result.push(rank);
    }

    result
}

fn save_predictions_csv(data: &[(DateTime<Utc>, PredictionDebugInfo, f32)], filename: &str) -> Result<(), CommonError> {
    let mut csv = Vec::with_capacity(data.len() + 1);
    csv.push("time,trend_filter,linear,gbm positive,gbm negative,linear ranked,gbm ranked,horizon".to_string());

    for (time, info, horizon) in data {
        csv.push(format!("{time},{},{:.6},{:.6},{:.6},{:.6},{:.6},{horizon:.1}",
                         info.is_filtered_by_trend, info.linear_raw, info.gbm_positive_raw,
                         info.gbm_negative_raw, info.linear_ranked, info.gbm_ranked));
    }

    std::fs::write(filename, csv.join("\n"))?;

    Ok(())
}

struct FeaturesWithHorizons {
    train_features: Vec<Vec<f32>>,
    test_features: Vec<Vec<f32>>,
    train_horizons: Vec<f32>,
    test_horizons: Vec<f32>,
    test_filtered_by_trend: Vec<bool>,
    test_timestamps: Vec<DateTime<Utc>>,
}

fn get_features_with_horizons(
    events: &[OrderBookValues],
    train_end: DateTime<Utc>,
    global_params: GlobalParams,
    hold_seconds: f64,
    trend_filter_seconds: f64,
    direction: TradeDirection,
) -> FeaturesWithHorizons {
    let hold = TimeDelta::milliseconds((1000.0 * hold_seconds) as i64);

    let mut window = FeaturesRollingWindow::new(
        TimeDelta::milliseconds((1000.0 * global_params.derivatives_normalize) as i64),
        TimeDelta::milliseconds((1000.0 * global_params.trend_tau) as i64),
        &FEATURES_NAMES,
        &FEATURES_OUTPUT,
    );

    let mut trend_filter = TrendFilter::new(TradeDirection::Buy, trend_filter_seconds);

    let mut train_features = Vec::new();
    let mut test_features = Vec::new();
    let mut train_horizons = Vec::new();
    let mut test_horizons = Vec::new();
    let mut test_filtered_by_trend = Vec::new();
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
            } else if values.time > train_end {
                test_features.push(features);
                test_horizons.push(pnl);
                test_timestamps.push(values.time);

                test_filtered_by_trend.push(
                    if trend_filter_seconds > 0.0 {
                        trend_filter.process(values)
                    } else {
                        true
                    }
                );
            }
        }
    }

    FeaturesWithHorizons {
        train_features,
        test_features,
        train_horizons,
        test_horizons,
        test_filtered_by_trend,
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
