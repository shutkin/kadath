use crate::gradient_boost::{MachineConfig, PredictionType};
use crate::nelder_mead::calibrate_linear_signal;
use crate::orderbook_values::OrderBookValues;
use crate::get_features_with_horizons;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::{DateTime, Utc};
use log::{error, info};
use model::TradeDirection;
use model::math_util::{ExponentFunction, percentile};
use ndarray::Array1;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Copy, Clone, Debug)]
pub struct GlobalParams {
    pub derivatives_normalize: f64,
    pub trend_tau: f64,
}

impl GlobalParams {
    pub fn from_param(param: &Array1<f64>, functions: &[Option<ExponentFunction>]) -> Self {
        Self {
            derivatives_normalize: functions[0].map(|f| f.calculate(param[0])).unwrap_or(param[0]),
            trend_tau: functions[1].map(|f| f.calculate(param[1])).unwrap_or(param[1]),
        }
    }
}

pub fn global_optimize(
    config_up: MachineConfig,
    config_down: MachineConfig,
    prediction_type: PredictionType,
    values: &[OrderBookValues],
    train_end: DateTime<Utc>,
    hold_ms: f64,
) -> Option<GlobalParams> {
    let functions = [
        Some(ExponentFunction::find(40.0, 200.0, 800.0)),
        Some(ExponentFunction::find(30.0, 160.0, 600.0)),
    ];

    let problem = GlobalProblem {
        config_up,
        config_down,
        prediction_type,
        values,
        train_end,
        hold_ms,
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
        .configure(|state| state.max_iters(80))
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
    config_up: MachineConfig,
    config_down: MachineConfig,
    prediction_type: PredictionType,
    values: &'a [OrderBookValues],
    train_end: DateTime<Utc>,
    hold_ms: f64,
    functions: &'a [Option<ExponentFunction>],
    count: AtomicU8,
}

impl CostFunction for GlobalProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let global = GlobalParams::from_param(param, self.functions);

        let fh = get_features_with_horizons(self.values, self.train_end, global, self.hold_ms, 0.0, TradeDirection::Buy);
        let test_milliseconds = (fh.test_timestamps[fh.test_timestamps.len() - 1] - fh.test_timestamps[0]).num_milliseconds();
        let events_per_hold = self.hold_ms * self.values.len() as f64 / test_milliseconds as f64;

        let linear = calibrate_linear_signal(None, &fh.train_features, &fh.train_horizons, events_per_hold as usize).unwrap();
        let predictions_linear = linear.test(&fh.test_features);

        //let up = GradientBoost::train(self.prediction_type, &self.config_up, &train_values, &train_horizons).unwrap();
        //let predictions_up = up.test(&test_values).unwrap();
        //let down = GradientBoost::train(self.prediction_type, &self.config_down, &train_values, &train_horizons).unwrap();
        //let predictions_down = down.test(&test_values).unwrap();

        //let n = predictions_up.len();
        //let ranks_pos = get_normalized_ranks(&predictions_up, false, n);
        //let ranks_neg = get_normalized_ranks(&predictions_down, true, n);

        //let predictions_gbm: Vec<f64> = ranks_pos.iter()
        //    .zip(ranks_neg.iter())
        //    .map(|(&r_pos, &r_neg)| r_pos * (0.0 + 1.0 * r_neg))
        //    .collect();

        let threshold_linear = percentile(&predictions_linear, 0.99);
        //let threshold_gbm = percentile(&predictions_gbm, 0.93);

        let pnls = predictions_linear.iter().zip(fh.test_horizons.iter())
                .filter(|(p, _)| **p >= threshold_linear)
                .map(|(_, pnl)| *pnl)
                .collect::<Vec<_>>();
        let sum = pnls.iter().copied().sum::<f32>();
        let score = sum - pnls.len() as f32;

        let count = self.count.fetch_add(1, Ordering::Relaxed);
        info!("{count}: {global:?} => {score}");
        Ok(-score as f64)
    }
}