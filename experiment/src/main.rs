mod math_utils;
mod simulation;
mod stats_collector;
mod deal;
mod gradient_boost;

use std::cmp::Ordering::Equal;
use crate::gradient_boost::{GbdtConfig, GradientBoost, MachineConfig, PredictionType, calculate_predictions_brier, calculate_predictions_pnl, optimize_gbdt, optimize_light_gbm};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{error, info, warn};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, merge_events};
use model::math_util::OutputType::{DerivativeNorm, Raw, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{ExponentFunction, FeaturesRollingWindow, OutputType, TimedFeatures};
use model::{DataSource, Instrument, OrderBook, order_book_cache};
use ndarray::Array1;
use serde_json::Value;
use simplelog::{LevelFilter, SimpleLogger};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread;

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

const FEATURES_NAMES: [&str; 6] = ["mid", "i3", "i5", "i10", "i20", "i50"];
const FEATURES_OUTPUT: [OutputType; 12] = [
    TrendDeviation("mid"),
    TrendDeviationDerivativeNorm("mid"),
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
pub const HORIZON_MS: u16 = 2000;
const THRESHOLD: f64 = 5.0;

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
        DateTime::parse_from_rfc3339("2026-10-27T05:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-27T11:00:00Z")?.to_utc(),
    );
    let optimize_train_end = DateTime::parse_from_rfc3339("2026-10-27T10:00:00Z")?.to_utc();
    let check_diapason = TimeDiapason::new(
        DateTime::parse_from_rfc3339("2026-10-27T06:00:00Z")?.to_utc(),
        DateTime::parse_from_rfc3339("2026-10-27T12:00:00Z")?.to_utc(),
    );
    let check_train_end = DateTime::parse_from_rfc3339("2026-10-27T11:00:00Z")?.to_utc();

    let prediction_type_up = PredictionType::ClassificationPositive(THRESHOLD);
    let prediction_type_down = PredictionType::ClassificationNegative(0.0);

    let all_instruments = db.get_instruments(DataSource::TInvest).await?;
    if let (Some(first_leg_id), Some(second_leg_id)) = (
        find_instrument_id(&all_instruments, tickers[0]),
        find_instrument_id(&all_instruments, tickers[1]),
    ) {
        let mut global = GlobalParams::default();

        let (cfg_up, cfg_down) = if OPTIMIZE {
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

            let (train, test, train_h, test_h) =
                features_with_horizons(first_leg_id, second_leg_id, &order_books1, &order_books2, optimize_train_end, global);
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
                optimize_light_gbm(prediction_type_up, train_clone, train_h_clone, test_clone, test_h_clone)
            });

            let train_clone = train_arc.clone();
            let train_h_clone = train_h_arc.clone();
            let test_clone = test_arc.clone();
            let test_h_clone = test_h_arc.clone();
            let h_down = thread::spawn(move || {
                optimize_light_gbm(prediction_type_down, train_clone, train_h_clone, test_clone, test_h_clone)
            });

            let cfg_up = h_up.join().unwrap()?;
            let cfg_down = h_down.join().unwrap()?;

            info!("\nUP: {cfg_up}\nDOWN: {cfg_down}");
            (cfg_up, cfg_down)
        } else {
            let cfg_json_up = r#"
{"bagging_fraction":0.895994038988877,"bagging_freq":4,"boosting_type":"gbdt","feature_fraction":0.5118879801269481,"lambda_l1":0.003943743350232156,"lambda_l2":0.1630738251743011,"learning_rate":0.006470704862855319,"max_depth":-1,"metric":"binary_logloss","min_child_samples":169,"n_jobs":-1,"num_iterations":2000,"num_leaves":50,"objective":"binary","random_state":42,"verbosity":-1}"#;
            let cfg_json_down = r#"
{"bagging_fraction":0.8361357907636818,"bagging_freq":1,"boosting_type":"gbdt","feature_fraction":0.9596713385375366,"lambda_l1":0.549411566785072,"lambda_l2":0.00047609572483133504,"learning_rate":0.0051143561715938065,"max_depth":-1,"metric":"binary_logloss","min_child_samples":267,"n_jobs":-1,"num_iterations":2000,"num_leaves":44,"objective":"binary","random_state":42,"verbosity":-1}"#;

            let value: Value = serde_json::from_str(cfg_json_up)?;
            let cfg_up = MachineConfig::LightGBM(value);

            let value: Value = serde_json::from_str(cfg_json_down)?;
            let cfg_down = MachineConfig::LightGBM(value);
            (cfg_up, cfg_down)
        };

        let (order_books1, order_books2) = get_order_books(&tickers, &[first_leg_id, second_leg_id], check_diapason, Some(&db)).await?;
        info!("Got {} {} books and {} {} books for check", order_books1.len(), tickers[0], order_books2.len(), tickers[1]);
        let (train, test, train_h, test_h) =
            features_with_horizons(first_leg_id, second_leg_id, &order_books1, &order_books2, check_train_end, global);
        info!("Check train {} events, test {} events", train.len(), test.len());

        info!("Train models...");
        let gb_up = GradientBoost::train(prediction_type_up, &cfg_up, &train, &train_h)?;
        //let gb_down = GradientBoost::train(prediction_type_down, &cfg_down, &train, &train_h)?;
        info!("Training completed");

        let h = thread::spawn(move || {
            gb_up.test(&test)
        });
        let predictions = h.join().unwrap()?;

        //let predictions_down = gb_down.test(&test)?;
        //let predictions = predictions_up.into_iter().zip(predictions_down.into_iter())
        //    .map(|(p_up, p_down)| p_up * (1.0 - p_down))
        //    .collect::<Vec<f64>>();

        let mut predictions_ord = predictions.clone();
        predictions_ord.sort_by(|p1, p2| p1.partial_cmp(p2).unwrap_or(Equal));

        for percentile in [0.99, 0.995, 0.999, 0.9995, 0.99966, 0.9999] {
            let index = ((predictions_ord.len() - 1) as f64 * percentile) as usize;
            let threshold = predictions_ord[index];
            let (mut correct, mut incorrect) = (0, 0);
            let (mut commission, mut pnl) = (0.0, 0.0);
            for (prediction, horizon) in predictions.iter().zip(test_h.iter()) {
                if *prediction > threshold {
                    if *horizon > 0.0 {
                        correct += 1;
                    } else {
                        incorrect += 1;
                    }

                    pnl += horizon;
                    commission += 5.0;
                }
            }
            info!("Percentile {percentile} ({threshold}): {:.2}%, PnL {pnl}, commission {commission}, gross {}",
            correct as f64 * 100.0 / (correct + incorrect) as f64, pnl - commission);
        }
    }

    Ok(())
}

fn export_to_csv(data: &[Vec<f32>], horizons: &[f64]) {
    let size = data[0].len();
    let mut csv = Vec::new();

    let header = format!("{},horizon", (0..size).map(|i| format!("f{i}")).collect::<Vec<_>>().join(","));
    csv.push(header);

    for (features, horizon) in data.iter().zip(horizons.iter()) {
        if features.len() != size {
            warn!("Feature length mismatch: expected {}, got {}", size, features.len())
        }
        let row = format!("{},{horizon:.2}", features.iter().map(|f| format!("{f:.4}")).collect::<Vec<_>>().join(","));
        csv.push(row);
    }

    if let Err(err) = std::fs::write("spread_data.csv", csv.join("\n")) {
        error!("Failed to export data to CSV: {}", err);
    }
}

fn features_with_horizons(
    first_leg_id: i32,
    second_leg_id: i32,
    order_books1: &[OrderBook],
    order_books2: &[OrderBook],
    train_end: DateTime<Utc>,
    global_params: GlobalParams,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f64>, Vec<f64>) {
    let events = merge_events(&[first_leg_id, second_leg_id], &[order_books1, order_books2]);

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

    let (mut f1, mut f2) = (None, None);
    let (mut h1, mut h2) = (None, None);

    for (i, event) in events.iter().enumerate() {
        if let Some(values) = OrderBookValues::from_order_book(&event.order_book)
            && let Some(h_price) = get_price_on_horizon(&events, event.instrument_id, i) {

            if event.instrument_id == first_leg_id {
                f1 = Some(window1.process(values.to_features()));
                h1 = Some(h_price - values.mid)
            } else if event.instrument_id == second_leg_id {
                f2 = Some(window2.process(values.to_features()));
                h2 = Some(h_price - values.mid)
            }

            if let Some(mut features) = f1.clone() && let Some(f2) = f2.clone()
                && let Some(h1) = h1 && let Some(h2) = h2 {
                features.extend(f2);
                let horizon = h1 - h2;

                if values.time <= train_end {
                    train_features.push(features);
                    train_horizons.push(horizon);
                } else {
                    test_features.push(features);
                    test_horizons.push(horizon);
                }
            }
        }
    }
    (train_features, test_features, train_horizons, test_horizons)
}

fn get_price_on_horizon(events: &[OrderBookEvent], instrument_id: i32, index: usize) -> Option<f64> {
    let mut i = index;
    let target_time = events[index].order_book.timestamp + TimeDelta::milliseconds(HORIZON_MS as i64);
    while i < events.len() {
        if events[i].instrument_id == instrument_id
            && events[i].order_book.timestamp >= target_time
            && let Some(values) = OrderBookValues::from_order_book(&events[i].order_book) {
            return Some(values.mid)
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
            features: vec![self.mid, self.imbalances[0], self.imbalances[1], self.imbalances[2], self.imbalances[3], self.imbalances[4]],
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
        let (train, test, train_h, test_h) = features_with_horizons(
            self.first_leg_id,
            self.second_leg_id,
            self.books1,
            self.books2,
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