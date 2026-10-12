use std::sync::atomic::{AtomicU8, Ordering};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use log::{error, info};
use ndarray::Array1;
use crate::orderbook_values::OrderBookValues;
use crate::signal_calculator::SignalCalculator;

#[derive(Copy, Clone, Debug)]
pub struct GlobalParams {
    pub derivatives_normalize: f64,
    pub trend_tau: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct GlobalParamsPair {
    pub positive: GlobalParams,
    pub negative: GlobalParams,
}

impl GlobalParams {
    pub fn from_param(param: &Array1<f64>) -> Self {
        Self {
            derivatives_normalize: (0.021 * param[0] + 3.9).exp(),
            trend_tau: (0.021 * param[1] + 3.9).exp(),
        }
    }
}

pub fn global_optimize(values: &[OrderBookValues], horizons: &[f64], penalty: f64) -> Option<GlobalParamsPair> {
    let values_positive = values.to_vec();
    let horizons_positive = horizons.to_vec();
    let handle_positive = std::thread::spawn(move || {
        let problem = GlobalProblem {
            values: &values_positive,
            horizons: &horizons_positive,
            is_positive: true,
            penalty,
            count: AtomicU8::new(0)
        };
        execute(problem)
    });

    let values_negative = values.to_vec();
    let horizons_negative = horizons.to_vec();
    let handle_negative = std::thread::spawn(move || {
        let problem = GlobalProblem {
            values: &values_negative,
            horizons: &horizons_negative,
            is_positive: false,
            penalty,
            count: AtomicU8::new(0)
        };
        execute(problem)
    });

    let positive = handle_positive.join().ok().flatten();
    let negative = handle_negative.join().ok().flatten();
    if let Some(positive) = positive && let Some(negative) = negative {
        Some(GlobalParamsPair { positive, negative })
    } else {
        None
    }
}

fn execute(problem: GlobalProblem) -> Option<GlobalParams> {
    let initial = Array1::from_vec(vec![60.0, 60.0]);

    let mut simplex = Vec::with_capacity(initial.len() + 1);

    simplex.push(initial.clone());

    for i in 0..initial.len() {
        let mut point = initial.clone();
        point[i] += 20.0;
        simplex.push(point);
    }

    let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());

    match Executor::new(problem, solver)
        .configure(|state| state.max_iters(80))
        .run() {
        Ok(result) => {
            if let Some(best_param) = result.state().get_best_param() {
                let score = -result.state().get_best_cost();
                let global_param = GlobalParams::from_param(best_param);
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
    values: &'a [OrderBookValues],
    horizons: &'a [f64],
    is_positive: bool,
    penalty: f64,
    count: AtomicU8,
}

impl CostFunction for GlobalProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let global = GlobalParams::from_param(param);
        let mut values = self.values.to_vec();
        OrderBookValues::calculate_window_values(&mut values, global.derivatives_normalize, global.trend_tau);
        let count = self.count.fetch_add(1, Ordering::Relaxed);
        let score = if let Some(result) = SignalCalculator::optimize(&values, self.horizons, self.is_positive, self.penalty) {
            result.score
        } else {
            0.0
        };
        info!("{} {count}: {global:?} => {score}", if self.is_positive {"positive"} else {"negative"});
        Ok(-score)
    }
}