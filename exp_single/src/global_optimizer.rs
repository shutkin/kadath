use crate::gradient_boost::{GradientBoost, MachineConfig, PredictionType};
use crate::orderbook_values::OrderBookValues;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use log::{error, info};
use model::common::TimeDiapason;
use model::math_util::ExponentFunction;
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
    machine_config: MachineConfig,
    prediction_type: PredictionType,
    values: &[OrderBookValues],
    horizons: &[f64],
    train_diapason: TimeDiapason,
    test_diapason: TimeDiapason,
) -> Option<GlobalParams> {
    let functions = [
        Some(ExponentFunction::find(40.0, 200.0, 800.0)),
        Some(ExponentFunction::find(30.0, 160.0, 600.0)),
    ];

    let problem = GlobalProblem {
        machine_config,
        prediction_type,
        values,
        horizons,
        train_diapason,
        test_diapason,
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
    machine_config: MachineConfig,
    prediction_type: PredictionType,
    values: &'a [OrderBookValues],
    horizons: &'a [f64],
    train_diapason: TimeDiapason,
    test_diapason: TimeDiapason,
    functions: &'a [Option<ExponentFunction>],
    count: AtomicU8,
}

impl CostFunction for GlobalProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let global = GlobalParams::from_param(param, self.functions);
        let mut values = self.values.to_vec();
        /*OrderBookValues::calculate_window_values(&mut values, global.derivatives_normalize, global.trend_tau);

        let (train_values, train_horizons) = filter_values_with_horizons(&values, self.horizons, self.train_diapason);
        let machine = GradientBoost::train(self.prediction_type, &self.machine_config, &train_values, &train_horizons).unwrap();

        let (test_values, test_horizons) = filter_values_with_horizons(&values, self.horizons, self.test_diapason);
        let predictions = machine.test(&test_values).unwrap();

        let error = calculate_predictions_error(&predictions, &test_horizons, self.prediction_type);

        let count = self.count.fetch_add(1, Ordering::Relaxed);
        info!("{count}: {global:?} => {error}");
        Ok(error)*/
        Ok(0.0)
    }
}