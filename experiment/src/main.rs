mod math_utils;
mod simulation;
mod stats_collector;
mod deal;
mod gradient_boost;
mod nelder_mead;

use std::cmp::Ordering::Equal;
use crate::gradient_boost::{GbdtConfig, GradientBoost, MachineConfig, PredictionType, calculate_predictions_brier, calculate_predictions_pnl, optimize_gbdt, optimize_light_gbm};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{error, info, warn};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, merge_events};
use model::math_util::OutputType::{DerivativeNorm, Raw, StandardDeviation, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{percentile, ExponentFunction, FeaturesRollingWindow, OutputType, TimedFeatures};
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use ndarray::Array1;
use serde_json::Value;
use simplelog::{LevelFilter, SimpleLogger};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread;
use crate::nelder_mead::calibrate_linear_signal;

pub const IMBALANCE_LEVELS: [usize; 5] = [3, 5, 10, 20, 50];

pub fn commission(revenue: f64, cost: f64) -> f64 {
    // T-Invest
    //(revenue + cost) * 0.015 / 100.0

    // Alor
    5.0
}

fn find_instrument_id(instruments: &[Instrument], ticker: &str) -> Option<i32> {
    instruments
        .iter()
        .find(|instrument| instrument.ticker == ticker)
        .and_then(|instrument| instrument.id)
}

async fn get_order_books(tickers: &[&str], ids: &[i32], diapason: TimeDiapason, db: Option<&Db>)
                         -> Result<(Vec<OrderBook>, Vec<OrderBook>), CommonError> {
    if let (Some(order_books1), Some(order_books2)) =
        (order_book_cache::read(tickers[0], diapason, true)?,
         order_book_cache::read(tickers[1], diapason, true)?) {
        Ok((order_books1, order_books2))
    } else {
        let (order_books1, order_books2) =
            (db.unwrap().get_order_books(ids[0], diapason).await?,
             db.unwrap().get_order_books(ids[1], diapason).await?);
        order_book_cache::write(tickers[0], diapason, &order_books1, true)?;
        order_book_cache::write(tickers[1], diapason, &order_books2, true)?;
        Ok((order_books1, order_books2))
    }
}

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
const DERIVATIVE_NORM_DIAPASON: f64 = 57.28;
const TREND_TAU: f64 = 614.61;
pub const HORIZON_MS: u16 = 4000;
const THRESHOLD: f32 = 5.0;

const OPTIMIZE: bool = true;
const OPTIMIZE_GLOBAL_PARAMS: bool = false;

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = db::Db::new(&db_url).await?;

    let tickers = ["SiZ6", "SiH7"];
    let optimize_diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-28T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-28T10:00:00.07Z")?.to_utc(),
    );
    let optimize_train_end = DateTime::parse_from_rfc3339("2026-10-28T09:00:00Z")?.to_utc();
    let check_diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-28T06:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-28T12:00:00Z")?.to_utc(),
    );
    let check_train_end = DateTime::parse_from_rfc3339("2026-10-28T11:00:00Z")?.to_utc();

    let prediction_type_profit = PredictionType::ClassificationPositive(THRESHOLD);
    let prediction_type_up = PredictionType::ClassificationPositive(0.0);

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    if let (Some(first_leg_id), Some(second_leg_id)) = (
        find_instrument_id(&all_instruments, tickers[0]),
        find_instrument_id(&all_instruments, tickers[1]),
    ) {
        let mut global = GlobalParams::default();

        let (cfg_up, cfg_profit) = if OPTIMIZE {
            let (order_books1, order_books2) = get_order_books(&tickers, &[first_leg_id, second_leg_id], optimize_diapason, Some(&db)).await?;
            info!("Got {} {} books and {} {} books for optimization", order_books1.len(), tickers[0], order_books2.len(), tickers[1]);

            if OPTIMIZE_GLOBAL_PARAMS {
                let cfg_json_up = r#"
{"bagging_fraction":0.9095846010507107,"bagging_freq":5,"boosting_type":"gbdt","feature_fraction":0.8618735092340805,"lambda_l1":0.12233067115578736,"lambda_l2":0.0036432070771574556,"learning_rate":0.025273059075476657,"max_depth":-1,"metric":"binary_logloss","min_child_samples":1086,"n_jobs":-1,"num_iterations":2000,"num_leaves":114,"objective":"binary","random_state":42,"verbosity":-1}            "#;
                let value: Value = serde_json::from_str(cfg_json_up)?;
                let cfg_up = MachineConfig::LightGBM(value);
                if let Some(best_global) = global_optimize(first_leg_id, second_leg_id, cfg_up, prediction_type_up, &order_books1, &order_books2, optimize_train_end) {
                    info!("Best global {best_global:?}");
                    global = best_global;
                }
            }

            let events = merge_events(&[first_leg_id, second_leg_id], &[&order_books1, &order_books2]);
            let (train, test, train_h, test_h) =
                features_with_horizons(first_leg_id, second_leg_id, &events, optimize_train_end, global);
            info!("Optimize train {} events, test {} events", train.len(), test.len());

            let train_arc = Arc::new(train);
            let train_h_arc = Arc::new(train_h);
            let test_arc = Arc::new(test);
            let test_h_arc = Arc::new(test_h);

            let train_clone = train_arc.clone();
            let train_h_clone = train_h_arc.clone();
            let test_clone = test_arc.clone();
            let test_h_clone = test_h_arc.clone();
            let h_up = thread::spawn(move || {
                optimize_gbdt(prediction_type_up, train_clone, train_h_clone, test_clone, test_h_clone)
            });

            let train_clone = train_arc.clone();
            let train_h_clone = train_h_arc.clone();
            let test_clone = test_arc.clone();
            let test_h_clone = test_h_arc.clone();
            let h_profit = thread::spawn(move || {
                optimize_gbdt(prediction_type_profit, train_clone, train_h_clone, test_clone, test_h_clone)
            });

            let cfg_up = h_up.join().unwrap()?;
            let cfg_profit = h_profit.join().unwrap()?;

            info!("\nUP: {cfg_up}\nPROFIT: {cfg_profit}");
            (cfg_up, cfg_profit)
        } else {
            let cfg_json_up = r#"
{"bagging_fraction":0.9956681735258639,"bagging_freq":1,"boosting_type":"gbdt","feature_fraction":0.7035543125048269,"lambda_l1":0.0056260035551993085,"lambda_l2":0.0010141596166938515,"learning_rate":0.009129879827747886,"max_depth":-1,"metric":"binary_logloss","min_child_samples":462,"n_jobs":-1,"num_iterations":2000,"num_leaves":18,"objective":"binary","random_state":42,"verbosity":-1}
"#;
            let cfg_json_profit = r#"
{"bagging_fraction":0.8087537633948574,"bagging_freq":3,"boosting_type":"gbdt","feature_fraction":0.9759985420473922,"lambda_l1":0.04218193773157231,"lambda_l2":0.053364693959552914,"learning_rate":0.016821711543696317,"max_depth":-1,"metric":"binary_logloss","min_child_samples":326,"n_jobs":-1,"num_iterations":2000,"num_leaves":51,"objective":"binary","random_state":42,"verbosity":-1}
"#;

            let value: Value = serde_json::from_str(cfg_json_up)?;
            let cfg_up = MachineConfig::LightGBM(value);

            let value: Value = serde_json::from_str(cfg_json_profit)?;
            let cfg_profit = MachineConfig::LightGBM(value);
            (cfg_up, cfg_profit)
        };

        let (order_books1, order_books2) = get_order_books(&tickers, &[first_leg_id, second_leg_id], check_diapason, Some(&db)).await?;
        info!("Got {} {} books and {} {} books for check", order_books1.len(), tickers[0], order_books2.len(), tickers[1]);
        let events = merge_events(&[first_leg_id, second_leg_id], &[&order_books1, &order_books2]);
        let (train, test, train_h, test_h) =
            features_with_horizons(first_leg_id, second_leg_id, &events, check_train_end, global);
        info!("Check train {} events, test {} events", train.len(), test.len());
        export_to_csv(&test, &test_h, "test_data.csv");

        info!("Train Linear model...");
        let linear = calibrate_linear_signal(None, &train, &train_h).ok_or("Linear Signal not calibrated")?;
        info!("Train UP model...");
        let gb_up = GradientBoost::train(prediction_type_up, &cfg_up, &train, &train_h)?;
        info!("Train PROFIT model...");
        let gb_profit = GradientBoost::train(prediction_type_profit, &cfg_profit, &train, &train_h)?;
        info!("Training completed");

        let test_clone = test.clone();
        let h_up = thread::spawn(move || {
            gb_up.test(&test_clone)
        });
        let test_clone = test.clone();
        let h_profit = thread::spawn(move || {
            gb_profit.test(&test_clone)
        });
        let h_linear = thread::spawn(move || {
            linear.test(&test)
        });
        let predictions_up = h_up.join().unwrap()?;
        let predictions_profit = h_profit.join().unwrap()?;
        let predictions_linear = h_linear.join().unwrap();

        const PERCENTILES: [f64; 4] = [0.97, 0.99, 0.993, 0.997];

        for p_linear in PERCENTILES {
            let threshold_linear = percentile(&predictions_linear, p_linear);
            for p_up in PERCENTILES {
                let threshold_up = percentile(&predictions_profit, p_up);

                let pnl_vec = predictions_linear.iter().zip(predictions_profit.iter()).zip(test_h.iter())
                    .filter(|((l, u), _)| **l >= threshold_linear && **u >= threshold_up)
                    .map(|(_, pnl)| *pnl)
                    .collect::<Vec<_>>();
                let total_pnl = pnl_vec.iter().copied().sum::<f32>();
                info!("Linear {p_linear}, Up {p_up} => {total_pnl}, {}", pnl_vec.len());
            }
        }

        let predictions = predictions_up.into_iter()
            .zip(predictions_profit.into_iter())
            .zip(predictions_linear.into_iter())
            .map(|((p1, p2), p3)| vec![p1, p2, p3])
            .collect::<Vec<_>>();
        export_to_csv(&predictions, &test_h, "predictions.csv");
    }

    Ok(())
}

fn optimize_predictions(predictions: &[Vec<f64>], horizons: &[f64]) {
    let initial = Array1::from_vec(vec![1.0, -0.1, -0.1, -0.1]);
}

fn find_nearest(features: &[f32], data: &[Vec<f32>]) -> usize {
    let mut best_diff = 0.0;
    let mut best_index = 0;
    for (i, data_features) in data.iter().enumerate() {
        let mut diff = 0.0;
        for (a, b) in features.iter().zip(data_features.iter()) {
            diff += (a - b).abs() / (1.0 + a + b);
        }
        if best_diff > diff {
            best_index = i;
            best_diff = diff;
        }
    }
    best_index
}

fn import_from_csv(filename: &str) -> Result<(Vec<Vec<f32>>, Vec<f32>), CommonError> {
    let data = std::fs::read_to_string(filename)?;
    let mut lines = data.split("\n");
    lines.next(); // skip header

    let mut features = Vec::new();
    let mut horizons = Vec::new();
    for line in lines {
        let mut values = line.split(",").map(|x| x.parse::<f32>()).collect::<Result<Vec<f32>, _>>()?;
        let horizon = values.pop().unwrap();
        horizons.push(horizon);
        features.push(values);
    }

    Ok((features, horizons))
}

fn export_to_csv<D: std::fmt::Display>(data: &[Vec<D>], horizons: &[f32], filename: &str) {
    let size = data[0].len();
    let mut csv = Vec::new();

    let header = format!("{},horizon", (0..size).map(|i| format!("f{i}")).collect::<Vec<_>>().join(","));
    csv.push(header);

    for (features, horizon) in data.iter().zip(horizons.iter()) {
        if features.len() != size {
            warn!("Feature length mismatch: expected {}, got {}", size, features.len())
        }
        let row = format!("{},{horizon:.2}", features.iter().map(|f| format!("{f:.6}")).collect::<Vec<_>>().join(","));
        csv.push(row);
    }

    if let Err(err) = std::fs::write(filename, csv.join("\n")) {
        error!("Failed to export data to CSV: {}", err);
    }
}

fn features_with_horizons(
    first_leg_id: i32,
    second_leg_id: i32,
    events: &[OrderBookEvent],
    train_end: DateTime<Utc>,
    global_params: GlobalParams,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let mut window1 = FeaturesRollingWindow::new(
        TimeDelta::milliseconds((1000.0 * global_params.derivatives_normalize) as i64),
        TimeDelta::milliseconds((1000.0 * global_params.trend_tau) as i64),
        &FEATURES_NAMES,
        &FEATURES_OUTPUT,
    );
    let mut window2 = FeaturesRollingWindow::new(
        TimeDelta::milliseconds((1000.0 * global_params.derivatives_normalize) as i64),
        TimeDelta::milliseconds((1000.0 * global_params.trend_tau) as i64),
        &FEATURES_NAMES,
        &FEATURES_OUTPUT,
    );

    let mut train_features = Vec::new();
    let mut test_features = Vec::new();
    let mut train_horizons = Vec::new();
    let mut test_horizons = Vec::new();

    let (mut v1, mut v2) = (None, None);
    let (mut f1, mut f2) = (None, None);

    for (i, event) in events.iter().enumerate() {
        if let Some(values) = OrderBookValues::from_order_book(&event.order_book) {

            if event.instrument_id == first_leg_id {
                f1 = Some(window1.process(values.to_features()));
                v1 = Some(values);
            } else if event.instrument_id == second_leg_id {
                f2 = Some(window2.process(values.to_features()));
                v2 = Some(values);
            }

            if let Some(mut features) = f1.clone() && let Some(f2) = f2.clone() {
                features.extend(f2);

                if let Some((h_ask1, _h_bid1)) = get_price_on_horizon(events, first_leg_id, i)
                    && let Some((_h_ask2, h_bid2)) = get_price_on_horizon(events, second_leg_id, i) {
                    let pnl = (v1.unwrap().bid - h_ask1 + h_bid2 - v2.unwrap().ask) as f32;

                    if values.time + TimeDelta::milliseconds(HORIZON_MS as i64) <= train_end {
                        train_features.push(features);
                        train_horizons.push(pnl);
                    } else if values.time > train_end {
                        test_features.push(features);
                        test_horizons.push(pnl);
                    }
                }
            }
        }
    }
    (train_features, test_features, train_horizons, test_horizons)
}

fn get_price_on_horizon(events: &[OrderBookEvent], instrument_id: i32, index: usize) -> Option<(f64, f64)> {
    let mut i = index;
    let target_time = events[index].order_book.timestamp + TimeDelta::milliseconds(HORIZON_MS as i64);
    while i < events.len() {
        if events[i].instrument_id == instrument_id
            && events[i].order_book.timestamp >= target_time
            && let Some(values) = OrderBookValues::from_order_book(&events[i].order_book) {
            return Some((values.ask, values.bid))
        }
        i += 1;
    }

    None
}

#[derive(Copy, Clone, Debug)]
pub struct OrderBookValues {
    time: DateTime<Utc>,
    bid: f64,
    ask: f64,
    mid: f64,
    imbalances: [f64; IMBALANCE_LEVELS.len()],
}

impl OrderBookValues {
    pub fn from_order_book(order_book: &OrderBook) -> Option<Self> {
        if order_book.asks.len() < 50 || order_book.bids.len() < 50 {
            return None;
        }

        let mut asks = order_book.asks.clone();
        asks.sort_by(|a, b| {a.price.partial_cmp(&b.price).unwrap()});

        let mut bids = order_book.bids.clone();
        bids.sort_by(|a, b| {b.price.partial_cmp(&a.price).unwrap()});

        let best_bid = bids[0].price.as_f64();
        let best_ask = asks[0].price.as_f64();
        let best_bid_volume = bids[0].size as f64;
        let best_ask_volume = asks[0].size as f64;
        let mid = (best_bid + best_ask) / 2.0;
        let micro = (best_ask * best_bid_volume + best_bid * best_ask_volume) / (best_bid_volume + best_ask_volume);

        let mut imbalances = [0.0; IMBALANCE_LEVELS.len()];
        for i in 0..IMBALANCE_LEVELS.len() {
            let levels = IMBALANCE_LEVELS[i];
            let (mut bids_sum, mut asks_sum) = (0.0, 0.0);
            for level in 0..levels {
                if let Some(bid) = bids.get(level) {
                    bids_sum += bid.size as f64;
                }
                if let Some(ask) = asks.get(level) {
                    asks_sum += ask.size as f64;
                }
            }
            imbalances[i] = (bids_sum - asks_sum) / (bids_sum + asks_sum);
        }

        Some(OrderBookValues {
            time: order_book.timestamp,
            bid: best_bid,
            ask: best_ask,
            mid,
            imbalances,
        })
    }

    pub fn to_features(&self) -> TimedFeatures {
        TimedFeatures {
            time: self.time,
            features: vec![self.mid, self.ask - self.bid, self.imbalances[0], self.imbalances[1], self.imbalances[2], self.imbalances[3], self.imbalances[4]],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GlobalParams {
    pub derivatives_normalize: f64,
    pub trend_tau: f64,
}

impl Default for GlobalParams {
    fn default() -> Self {
        Self {
            derivatives_normalize: DERIVATIVE_NORM_DIAPASON,
            trend_tau: TREND_TAU,
        }
    }
}

impl GlobalParams {
    pub fn from_param(param: &Array1<f64>, functions: &[Option<ExponentFunction>]) -> Self {
        Self {
            derivatives_normalize: functions[0].map(|f| f.calculate(param[0])).unwrap_or(param[0]),
            trend_tau: functions[1].map(|f| f.calculate(param[1])).unwrap_or(param[1]),
        }
    }
}

fn global_optimize(
    first_leg_id: i32,
    second_leg_id: i32,
    machine_config: MachineConfig,
    prediction_type: PredictionType,
    books1: &[OrderBook],
    books2: &[OrderBook],
    train_end: DateTime<Utc>,
) -> Option<GlobalParams> {
    let functions = [
        Some(ExponentFunction::find(15.0, 100.0, 1000.0)),
        Some(ExponentFunction::find(30.0, 300.0, 1500.0)),
    ];

    let problem = GlobalProblem {
        first_leg_id,
        second_leg_id,
        machine_config,
        prediction_type,
        books1,
        books2,
        train_end,
        functions: &functions,
        count: AtomicU8::new(0)
    };
    execute(problem, &functions)
}

fn execute(problem: GlobalProblem, functions: &[Option<ExponentFunction>]) -> Option<GlobalParams> {
    let initial = Array1::from_vec(vec![0.0, 0.0]);

    let mut simplex = Vec::with_capacity(initial.len() + 1);

    simplex.push(initial.clone());

    for i in 0..initial.len() {
        let mut point = initial.clone();
        point[i] += 0.25;
        simplex.push(point);
    }

    let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());

    match Executor::new(problem, solver)
        .configure(|state| state.max_iters(50))
        .run() {
        Ok(result) => {
            if let Some(best_param) = result.state().get_best_param() {
                let score = -result.state().get_best_cost();
                let global_param = GlobalParams::from_param(best_param, functions);
                info!("Best score: {score} with {global_param:?}");
                Some(global_param)
            } else {
                None
            }
        }
        Err(err) => {
            error!("Unable to optimize calculator {err}");
            None
        }
    }
}

struct GlobalProblem<'a> {
    first_leg_id: i32,
    second_leg_id: i32,
    train_end: DateTime<Utc>,
    machine_config: MachineConfig,
    prediction_type: PredictionType,
    books1: &'a [OrderBook],
    books2: &'a [OrderBook],
    functions: &'a [Option<ExponentFunction>],
    count: AtomicU8,
}

impl CostFunction for GlobalProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let global = GlobalParams::from_param(param, self.functions);
        let events = merge_events(&[self.first_leg_id, self.second_leg_id], &[self.books1, self.books2]);
        let (train, test, train_h, test_h) = features_with_horizons(
            self.first_leg_id,
            self.second_leg_id,
            &events,
            self.train_end,
            global
        );
        let boost = GradientBoost::train(self.prediction_type, &self.machine_config, &train, &train_h).unwrap();
        let predictions = boost.test(&test).unwrap();

        let score = calculate_predictions_brier(&predictions, &train_h, self.prediction_type);

        let count = self.count.fetch_add(1, Ordering::Relaxed);
        info!("{count}: {global:?} => {score}");
        Ok(-score)
    }
}