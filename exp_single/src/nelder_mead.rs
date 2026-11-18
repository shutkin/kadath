use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use log::{debug, error, info};
use model::common::CommonError;
use ndarray::Array1;

const COMMISSION: f32 = 0.5;

#[derive(Debug)]
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

        let mut sum_weight = 0.0;
        let mut sum_weighted_horizons = 0.0;
        self.values.iter()
            .zip(self.horizons.iter())
            .enumerate()
            .for_each(|(i, (v, &horizon))| {
                let signal = calculator.calculate(v);
                if signal > 0.0 {
                    let weight = i as f32 / self.values.len() as f32;
                    let weight = weight * weight;
                    sum_weight += weight;
                    sum_weighted_horizons += horizon * weight;
                }
            });

        if sum_weight < 2.0 {
            f32::NEG_INFINITY
        } else {
            let score = sum_weighted_horizons / sum_weight;
            debug!("Linear score: {score}");
            score
        }
    }
}

struct TradingProblem {
    values: Vec<Vec<f32>>,
    horizons: Vec<f32>,
    events_per_hold: usize,
}

impl ScoreProvider for TradingProblem {
    fn score(&self, param: &Array1<f32>) -> f32 {
        let calculator = LinearSignal::from_param(param);
        
        let mut pnl = 0.0;
        let mut skip_counter = 0;
        let mut deals = 0;

        self.values.iter()
            .zip(self.horizons.iter())
            .for_each(|(v, horizon)| {
                if skip_counter > 0 {
                    skip_counter -= 1;
                } else {
                    let signal = calculator.calculate(v);
                    if signal > 0.0 {
                        pnl += horizon - COMMISSION;
                        skip_counter = self.events_per_hold;
                        deals += 1;
                    }
                }
            });

        debug!("{calculator:?}: {pnl} on {deals} deals");
        if deals > 0 {
            pnl
        } else {
            f32::NEG_INFINITY
        }
    }
}

pub fn calibrate_linear_signal(prev: Option<&LinearSignal>, values: &[Vec<f32>], horizons: &[f32], events_per_hold: usize) -> Option<LinearSignal> {
    let size = values[0].len();
    let initial = prev.map(|calc| calc.to_param()).unwrap_or(Array1::from_vec(vec![0.0; size]));
    match execute_optimization(initial, values, horizons, events_per_hold, ProblemKind::TRADING) {
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
    events_per_hold: usize,
    problem_kind: ProblemKind,
) -> Result<LinearSignal, CommonError> {
    let problem = match problem_kind {
        ProblemKind::HORIZON => ProblemWrapper { provider: Box::new(HorizonProblem {
            values: values.to_vec(), horizons: horizons.to_vec(),
        })
        },
        ProblemKind::TRADING => ProblemWrapper { provider: Box::new(TradingProblem {
            values: values.to_vec(), horizons: horizons.to_vec(), events_per_hold,
        })
        }
    };

    let mut simplex = Vec::with_capacity(initial.len() + 1);

    simplex.push(initial.clone());

    for i in 0..initial.len() {
        let mut point = initial.clone();
        point[i] += 0.5;
        simplex.push(point);
    }

    let solver = NelderMead::<Array1<f32>, f32>::new(simplex.clone());
    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(2048))
        .run()?;

    let best_score = -result.state().get_best_cost();
    let best_param = result.state().get_best_param().ok_or("Failed to get best param from executor state")?;
    let mut signal = LinearSignal::from_param(best_param);
    signal.train_score = best_score;
    Ok(signal)
}