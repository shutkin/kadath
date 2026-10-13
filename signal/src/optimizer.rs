use crate::orderbook_values::OrderBookValues;
use crate::signal_calculator::{SignalCalculator, SignalPerformance};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::{DateTime, TimeDelta, Utc};
use log::{debug, error};
use model::common::{CommonError, TimeDiapason};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

const HOLD_TIME_TO_SCORE_PENALTY_RATIO: f64 = 1.0 / 15_000.0;

const COMMISSION: f64 = 0.5;


enum ProblemKind { HORIZON, TRADING }

trait ScoreProvider {
    fn score(&self, param: &Array1<f64>) -> f64;
}

struct ProblemWrapper {
    provider: Box<dyn ScoreProvider>,
}

impl CostFunction for ProblemWrapper {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let score = self.provider.score(param);
        Ok(-score)
    }
}

struct HorizonProblem {
    values: Vec<OrderBookValues>,
    horizons: Vec<f64>,
    hold_time: i32,
    is_positive: bool,
    decay_time: f64,
    correctness: Option<Rc<RefCell<(i32, i32)>>>,
}

impl ScoreProvider for HorizonProblem {
    fn score(&self, param: &Array1<f64>) -> f64 {
        let last_time = self.values[self.values.len() - 1].time;
        let calculator = SignalCalculator::from_optimizer_param(last_time, self.hold_time, param);

        let penalty = self.hold_time as f64 * HOLD_TIME_TO_SCORE_PENALTY_RATIO;
        let (mut correct, mut incorrect) = (0, 0);
        let sum = self.values.iter().zip(self.horizons.iter())
            .map(|(v, &horizon)| {
                let signal = calculator.calculate(v);
                let weight = (-(last_time - v.time).as_seconds_f64() / self.decay_time).exp();
                if signal < -f64::MIN_POSITIVE && !self.is_positive {
                    if horizon < -f64::MIN_POSITIVE {
                        correct += 1;
                    } else {
                        incorrect += 1;
                    }
                    (-horizon - penalty) * weight
                } else if signal > f64::MIN_POSITIVE && self.is_positive {
                    if horizon > f64::MIN_POSITIVE {
                        correct += 1;
                    } else {
                        incorrect += 1;
                    }
                    (horizon - penalty) * weight
                } else { 0.0 }
            }).sum::<f64>();
        if let Some(correctness) = self.correctness.as_ref() {
            correctness.replace((correct, incorrect));
        }
        sum
    }
}

struct TradingProblem {
    values: Vec<OrderBookValues>,
    hold_time: i32,
    is_positive: bool,
    decay_time: f64,
    correctness: Option<Rc<RefCell<(i32, i32)>>>,
}

impl ScoreProvider for TradingProblem {
    fn score(&self, param: &Array1<f64>) -> f64 {
        struct Deal {
            closing_at: DateTime<Utc>,
            open_price: f64,
        }

        let last_time = self.values[self.values.len() - 1].time;
        let calculator = SignalCalculator::from_optimizer_param(last_time, self.hold_time, param);

        let (mut correct, mut incorrect) = (0, 0);

        let mut cur_deal = Option::<Deal>::None;
        let mut pnl = 0.0;
        self.values.iter()
            .for_each(|v| {
                if let Some(deal) = &cur_deal {
                    if v.time >= deal.closing_at {
                        let price_delta = v.mid - deal.open_price;
                        let weight = (-(last_time - v.time).as_seconds_f64() / self.decay_time).exp();
                        if self.is_positive {
                            if price_delta > 0.0 {
                                correct += 1;
                            } else {
                                incorrect += 1;
                            }
                            pnl += weight * price_delta - weight * COMMISSION;
                        } else {
                            if price_delta > 0.0 {
                                incorrect += 1;
                            } else {
                                correct += 1;
                            }
                            pnl += weight * -price_delta - weight * COMMISSION;
                        }
                        cur_deal = None;
                    }
                } else {
                    let signal = calculator.calculate(v);
                    if signal > f64::MIN_POSITIVE && self.is_positive || signal < -f64::MIN_POSITIVE && !self.is_positive {
                        cur_deal = Some(Deal {open_price: v.mid, closing_at: v.time + TimeDelta::milliseconds(self.hold_time as i64)});
                    }
                }
            });
        if let Some(correctness) = self.correctness.as_ref() {
            correctness.replace((correct, incorrect));
        }

        pnl
    }
}

pub fn calibrate_signal_calculator(prev: &Option<SignalCalculator>, values: &[OrderBookValues], is_buying: bool,
                                   decay_time: f64, hold_time: i32) -> Option<SignalCalculator> {
    let initial = prev.map(|calc| calc.to_optimizer_param()).unwrap_or(Array1::from_vec(vec![0.0; 13]));
    match execute_optimization(initial, values, hold_time, is_buying, decay_time, ProblemKind::TRADING) {
        Ok(result) => {
            if let Some(calculator) = &result {
                debug!("{} calibration score: {}", calculator.get_created_on(), calculator.get_train_score());
            }
            result
        },
        Err(err) => {
            error!("Failed to optimize {} calculator: {err}", if is_buying { "buying" } else { "selling" });
            None
        }
    }
}

fn build_horizon(values: &[OrderBookValues], hold: i32) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());

    for (i, v) in values.iter().enumerate() {
        let target_time = v.time + TimeDelta::milliseconds(hold as i64);
        let horizon = if let Some(h) = find_value_on_horizon(values, i, target_time) {
            h.mid - values[i].mid
        } else {
            0.0
        };
        result.push(horizon);
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

fn execute_optimization(
    initial: Array1<f64>,
    values: &[OrderBookValues],
    hold_time: i32,
    is_positive: bool,
    decay_time: f64,
    problem_kind: ProblemKind,
) -> Result<Option<SignalCalculator>, CommonError> {
    let values = values.to_vec();
    let horizons = build_horizon(&values, hold_time);
    let problem = match problem_kind {
        ProblemKind::HORIZON => ProblemWrapper { provider: Box::new(HorizonProblem {
                values: values.clone(), horizons: horizons.clone(), is_positive, decay_time, hold_time, correctness: None,
            })
        },
        ProblemKind::TRADING => ProblemWrapper { provider: Box::new(TradingProblem {
                values: values.clone(), is_positive, decay_time, hold_time, correctness: None,
            })
        }
    };

    let mut simplex = Vec::with_capacity(initial.len() + 1);

    simplex.push(initial.clone());

    for i in 0..initial.len() {
        let mut point = initial.clone();
        point[i] += 0.12;
        simplex.push(point);
    }

    let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());
    let result = Executor::new(problem, solver, )
        .configure(|state| state.max_iters(64))
        .run()?;

    let (first_time, last_time) = (values[0].time, values[values.len() - 1].time);
    let best_score = -result.state().get_best_cost();
    let best_param = result.state().get_best_param().ok_or("Failed to get best param from executor state")?;
    
    let correctness = Rc::new(RefCell::new((0, 0)));
    match problem_kind {
        ProblemKind::HORIZON => HorizonProblem {
            values, horizons, is_positive, decay_time, hold_time, correctness: Some(correctness.clone()),
        }.score(best_param),
        ProblemKind::TRADING => TradingProblem {
            values, is_positive, decay_time, hold_time, correctness: Some(correctness.clone()),
        }.score(best_param)
    };

    let mut calculator = SignalCalculator::from_optimizer_param(last_time, hold_time, best_param);

    let correctness = correctness.take();
    calculator.set_performance(SignalPerformance {
        train_diapason: TimeDiapason::new(first_time, last_time),
        hold_time,
        training_score: best_score,
        training_correct_predictions: correctness.0,
        training_incorrect_predictions: correctness.1,
        actual_total_pnl: 0.0,
        actual_deals: 0,
        actual_wins: 0,
    });

    if best_score > 0.0 {
        Ok(Some(calculator))
    } else {
        Ok(None)
    }
}
