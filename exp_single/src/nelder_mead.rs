use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::{DateTime, Utc};
use log::{debug, error, info};
use model::common::CommonError;
use ndarray::Array1;

const COMMISSION: f64 = 0.5;

pub struct LinearSignal {
    factors: Vec<f32>,
    threshold: f32,
    train_score: f32,
}

impl LinearSignal {
    pub fn calculate(&self, features: &[f32]) -> f64 {
        let sum = self.factors.iter().zip(features)
            .map(|(factor, feature)| (*factor as f64) * (*feature as f64))
            .sum::<f64>();
        sum - self.threshold as f64
    }

    pub fn test(&self, values: &[Vec<f32>]) -> Vec<f64> {
        values.iter().map(|values| self.calculate(values)).collect()
    }

    fn from_param(param: &Array1<f32>) -> Self {
        let threshold = param[0];
        let factors = param.iter().skip(1).copied().collect();
        Self {
            factors,
            threshold,
            train_score: 0.0,
        }
    }

    fn to_param(&self) -> Array1<f32> {
        let mut vec = vec![self.threshold];
        vec.extend(self.factors.iter().copied());
        Array1::from_vec(vec)
    }
}

enum ProblemKind { HORIZON, TRADING }

trait ScoreProvider {
    fn score(&self, param: &Array1<f32>) -> f32;
}

struct ProblemWrapper {
    provider: Box<dyn ScoreProvider>,
}

impl CostFunction for ProblemWrapper {
    type Param = Array1<f32>;
    type Output = f32;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let score = self.provider.score(param);
        Ok(-score)
    }
}

struct HorizonProblem {
    values: Vec<Vec<f32>>,
    horizons: Vec<f32>,
}

impl ScoreProvider for HorizonProblem {
    fn score(&self, param: &Array1<f32>) -> f32 {
        let calculator = LinearSignal::from_param(param);

        let mut count = 0;
        let sum = self.values.iter().zip(self.horizons.iter())
            .map(|(v, &horizon)| {
                let signal = calculator.calculate(v);
                if signal > 0.0 {
                    count += 1;
                    horizon
                } else {
                    0.0
                }
            }).sum::<f32>();

        if count < 8 {
            f32::NEG_INFINITY
        } else {
            debug!("Linear score: {sum}");
            sum
        }
    }
}

struct TradingProblem {
    values: Vec<Vec<f32>>,
    horizons: Vec<f32>,
}

impl ScoreProvider for TradingProblem {
    fn score(&self, param: &Array1<f32>) -> f32 {
        struct Deal {
            closing_at: DateTime<Utc>,
            open_price: f64,
        }
        0.0
    }
}

pub fn calibrate_linear_signal(prev: Option<&LinearSignal>, values: &[Vec<f32>], horizons: &[f32]) -> Option<LinearSignal> {
    let size = values[0].len();
    let initial = prev.map(|calc| calc.to_param()).unwrap_or(Array1::from_vec(vec![0.0; size]));
    match execute_optimization(initial, values, horizons, ProblemKind::HORIZON) {
        Ok(result) => {
            info!("Calibration score: {}", result.train_score);
            Some(result)
        },
        Err(err) => {
            error!("Failed to optimize linear signal: {err}");
            None
        }
    }
}

fn execute_optimization(
    initial: Array1<f32>,
    values: &[Vec<f32>],
    horizons: &[f32],
    problem_kind: ProblemKind,
) -> Result<LinearSignal, CommonError> {
    let problem = match problem_kind {
        ProblemKind::HORIZON => ProblemWrapper { provider: Box::new(HorizonProblem {
            values: values.to_vec(), horizons: horizons.to_vec(),
        })
        },
        ProblemKind::TRADING => ProblemWrapper { provider: Box::new(TradingProblem {
            values: values.to_vec(), horizons: horizons.to_vec(),
        })
        }
    };

    let mut simplex = Vec::with_capacity(initial.len() + 1);

    simplex.push(initial.clone());

    for i in 0..initial.len() {
        let mut point = initial.clone();
        point[i] += 0.25;
        simplex.push(point);
    }

    let solver = NelderMead::<Array1<f32>, f32>::new(simplex.clone());
    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(4096))
        .run()?;

    let best_score = -result.state().get_best_cost();
    let best_param = result.state().get_best_param().ok_or("Failed to get best param from executor state")?;
    let mut signal = LinearSignal::from_param(best_param);
    signal.train_score = best_score;
    Ok(signal)
}