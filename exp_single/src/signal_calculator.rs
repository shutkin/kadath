use std::ops::Add;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use argmin::solver::particleswarm::ParticleSwarm;
use chrono::{DateTime, TimeDelta, Utc};
use log::{error, info};
use ndarray::Array1;
use crate::find_value_on_horizon;
use crate::orderbook_values::{OrderBookValues, IMBALANCE_LEVELS};

#[derive(Clone, Copy, Debug)]
pub struct SignalCalculator {
    pub threshold: f64,
    //pub derivative_weight: f64,
    pub imbalance_weights: [f64; IMBALANCE_LEVELS.len()],
    pub trend_weight: f64,
}

// SignalCalculator { threshold: 0.9166666666666666, imbalance_weights: [0.17222222222222222, 1.0222222222222221, 0.02222222222222222, 0.02222222222222222], trend_weight: 0.9555555555555555 }

impl SignalCalculator {
    pub fn default() -> Self {
        SignalCalculator { threshold: 0.9166666666666666, imbalance_weights: [0.17222222222222222, 1.0222222222222221, 0.02222222222222222, 0.02222222222222222], trend_weight: 0.9555555555555555 }
    }

    fn from_param(param: &Array1<f64>) -> Self {
        Self {
            threshold: param[0],
            //derivative_weight: param[1],
            imbalance_weights: [param[1], param[2], param[3], param[4]],
            trend_weight: param[5],
        }
    }

    pub fn calculate(&self, v: &OrderBookValues) -> f64 {
        let mut score = 0.0;// v.std_derivative * self.derivative_weight;
        for (i, w) in self.imbalance_weights.iter().enumerate() {
            score += v.imbalances[i] * w
        }
        score += v.trend_deviation * self.trend_weight;
        score - self.threshold
    }

    pub fn optimize_swarm(values: &[OrderBookValues], hold_time_ms: u32) -> Option<Self> {
        let min = Array1::from_vec(
            //      A    I3    I5    I10   I20   T
            vec![0.0, -2.0, -2.0, -2.0, -2.0, 0.0]
        );
        let max = Array1::from_vec(
            //      A     I3   I5   I10  I20  T
            vec![10.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        );

        let solver = ParticleSwarm::new((min, max), 42);
        info!("Start optimization on {} - {}", values[0].time, values[values.len() - 1].time);
        match Executor::new(TradeProblem {values, hold_time_ms}, solver)
            .configure(|state| state.max_iters(4192))
            .run() {
            Ok(result) => {
                let best_profit = -result.state().get_best_cost();
                let param = result.state().get_best_param().unwrap().position.clone();
                let calculator = SignalCalculator::from_param(&param);
                info!("Best profit {best_profit} with {calculator:?}");
                Some(calculator)
            }
            Err(err) => {
                error!("Unable to optimize calculator {err}");
                None
            }
        }
    }

    pub fn optimize(values: &[OrderBookValues], hold_time_ms: u32) -> Option<Self> {
        let initial = Array1::from_vec(
            //      I3    I5   I10   I20   T
            //vec![0.17, 1.0, 0.02, 0.02, -0.9]
            vec![0.9, 0.17, 1.0, 0.02, 0.02, -0.9]
        );

        let mut simplex = Vec::with_capacity(initial.len() + 1);

        simplex.push(initial.clone());

        for i in 0..initial.len() {
            let mut point = initial.clone();
            point[i] += match i {
                0 => 0.5,  // Threshold
                _ => 0.1,  // Imbalance, Trend
            };
            simplex.push(point);
        }

        let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());
        info!("Start optimization on {} - {}", values[0].time, values[values.len() - 1].time);
        match Executor::new(TradeProblem {values, hold_time_ms}, solver)
            .configure(|state| state.max_iters(4192))
            .run() {
            Ok(result) => {
                let best_profit = -result.state().get_best_cost();
                let calculator = result.state().get_best_param()
                    .map(SignalCalculator::from_param);
                /*if let Some(c) = calculator.as_mut() {
                    let threshold = c.find_threshold(values, hold_time_ms);
                    c.threshold = threshold;
                }*/
                info!("Best correlation {best_profit} with {calculator:?}");
                calculator
            }
            Err(err) => {
                error!("Unable to optimize calculator {err}");
                None
            }
        }
    }

    fn find_threshold(&self, values: &[OrderBookValues], hold_time: u32) -> f64 {
        let mut scores = values.iter().map(|v| self.calculate(v)).collect::<Vec<_>>();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max_score = scores[scores.len() - 10];

        let mut best_profit = f64::NEG_INFINITY;
        let mut best_threshold = 0.0;

        for i in 1..4096 {
            let mut c = *self;
            c.threshold = i as f64 * max_score / 4096.0;
            let result = trade_sim(values, &c, hold_time);
            let profit = result.income - result.outcome - result.commission;
            if profit > best_profit {
                best_profit = profit;
                best_threshold = c.threshold;
            }
        }

        info!("Best profit {best_profit} with threshold {best_threshold}");
        best_threshold
    }
}

struct TradeProblem<'a> {
    values: &'a [OrderBookValues],
    hold_time_ms: u32,
}

impl CostFunction for TradeProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let calculator = SignalCalculator::from_param(param);
        let result = trade_sim(self.values, &calculator, self.hold_time_ms);
        let profit = result.income - result.outcome - result.commission;
        Ok(-profit)
    }
}

struct SpearmanProblem<'a> {
    values: &'a [OrderBookValues],
    hold_time_ms: u32,
}

impl CostFunction for SpearmanProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let calculator = SignalCalculator::from_param(param);
        let mut scores = Vec::new();
        let mut target = Vec::new();
        for (i, v) in self.values.iter().enumerate() {
            let target_time = v.time + TimeDelta::milliseconds(self.hold_time_ms as i64);
            if let Some(hv) = find_value_on_horizon(self.values, i, target_time) {
                scores.push(calculator.calculate(v));
                target.push(hv.bid - v.ask);
            }
        }
        let correlation = correlation::spearmanr(&scores, &target);
        Ok(-correlation)
    }
}

#[derive(Default)]
pub struct TradeResult {
    pub income: f64,
    pub outcome: f64,
    pub commission: f64,
}

impl Add for TradeResult {
    type Output = TradeResult;

    fn add(self, rhs: Self) -> Self::Output {
        TradeResult {
            income: self.income + rhs.income,
            outcome: self.outcome + rhs.outcome,
            commission: self.commission + rhs.commission,
        }
    }
}

struct Deal {
    open_time: DateTime<Utc>,
    open_price: f64,
}

impl Deal {
    fn open(v: &OrderBookValues) -> Self {
        Self {
            open_time: v.time,
            open_price: v.ask,
        }
    }

    fn close(&self, v: &OrderBookValues) -> TradeResult {
        TradeResult {
            income: v.bid,
            outcome: self.open_price,
            commission: 5.0,
        }
    }
}

pub fn trade_sim(values: &[OrderBookValues], calculator: &SignalCalculator, hold_time_ms: u32) -> TradeResult {
    let mut result = TradeResult::default();
    let mut prev_signal = 0.0;
    let mut deal = Option::<Deal>::None;

    for v in values {
        let signal = calculator.calculate(v);

        if let Some(deal) = &deal {
            if v.time > deal.open_time + TimeDelta::milliseconds(hold_time_ms as i64) {
                result = result + deal.close(v);
            }
        } else {
            if signal > 0.0 && prev_signal < 0.0 {
                deal = Some(Deal::open(v));
            }
        }

        prev_signal = signal;
    }

    result
}