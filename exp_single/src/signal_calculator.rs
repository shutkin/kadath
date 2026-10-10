use crate::orderbook_values::{IMBALANCE_LEVELS, OrderBookValues};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use log::error;
use ndarray::Array1;

#[derive(Clone, Copy, Debug)]
pub struct SignalCalculator {
    pub threshold_positive: f64,
    pub threshold_negative: f64,
    pub imbalance_weights: [f64; IMBALANCE_LEVELS.len()],
    pub imbalance_derivative_weights: [f64; IMBALANCE_LEVELS.len()],
    pub trend_weight: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct OptimizationResult {
    pub score: f64,
    pub calculator: SignalCalculator,
}

impl SignalCalculator {
    fn from_param(param: &Array1<f64>) -> Self {
        Self {
            threshold_positive: 0.0001 + param[0].abs(),
            threshold_negative: -0.0001 - param[1].abs(),
            imbalance_weights: [param[2], param[3], param[4], param[5], param[6]],
            imbalance_derivative_weights: [param[7], param[8], param[9], param[10], param[11]],
            trend_weight: param[12],
        }
    }

    pub fn calculate(&self, v: &OrderBookValues) -> f64 {
        let mut score = 0.0;

        for (i, w) in self.imbalance_weights.iter().enumerate() {
            score += v.imbalances[i] * w
        }

        for (i, w) in self.imbalance_derivative_weights.iter().enumerate() {
            score += v.imbalances_derivative[i] * w
        }

        score += (v.trend - v.mid) * self.trend_weight;

        if score >= 0.0 {
            if score > self.threshold_positive {
                score
            } else {
                0.0
            }
        } else {
            if score < self.threshold_negative {
                score
            } else {
                0.0
            }
        }
    }

    pub fn optimize(values: &[OrderBookValues], horizons: &[f64], penalty: f64) -> Option<OptimizationResult> {
        let initial = Array1::from_vec(vec![0.1, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1]);

        let mut simplex = Vec::with_capacity(initial.len() + 1);

        simplex.push(initial.clone());

        for i in 0..initial.len() {
            let mut point = initial.clone();
            point[i] += 0.1;
            simplex.push(point);
        }

        let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());
        match Executor::new(
            NaivePnLProblem { values, horizons: horizons.to_vec(), penalty },
            solver,
        )
        .configure(|state| state.max_iters(512))
        .run()
        {
            Ok(result) => {
                let best_correlation = -result.state().get_best_cost();
                result
                    .state()
                    .get_best_param()
                    .map(|param| OptimizationResult {
                        score: best_correlation,
                        calculator: SignalCalculator::from_param(param),
                    })
            }
            Err(err) => {
                error!("Unable to optimize calculator {err}");
                None
            }
        }
    }
}

struct NaivePnLProblem<'a> {
    values: &'a [OrderBookValues],
    horizons: Vec<f64>,
    penalty: f64,
}

impl CostFunction for NaivePnLProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let calculator = SignalCalculator::from_param(param);

        let sum = self.values.iter().zip(self.horizons.iter()).enumerate()
            .map(|(i, (v, horizon))| {
                let signal = if i > 500 { calculator.calculate(v) } else { 0.0 };
                if signal < -f64::MIN_POSITIVE {
                    -*horizon - self.penalty
                } else if signal > f64::MIN_POSITIVE {
                    *horizon - self.penalty
                } else { 0.0 }
            }).sum::<f64>();
        Ok(-sum)
    }
}
