use std::range::Range;
use crate::deal::{Deal, DealDirection};
use crate::math_utils::huber_loss;
use crate::simulation::{find_order_books_on_horizon, run_simulation};
use crate::{market_events_time_diapason, MarketEvent, OrderBookValues, commission, IMBALANCE_LEVELS, DEFAULT_HOLD_MS};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::TimeDelta;
use log::{error, info};
use ndarray::Array1;

const SOLVER_ITERATIONS: u64 = 8192;

#[derive(Copy, Clone, Debug, Default)]
pub struct SignalParamsDir {
    pub threshold: f64,
    pub c_range: Range<f64>,
    pub a_range: Range<f64>,
    pub derivative1_weight: f64,
    pub derivative2_weight: f64,
    pub imbalance1_weights: [f64; IMBALANCE_LEVELS.len()],
    pub imbalance2_weights: [f64; IMBALANCE_LEVELS.len()],
}

impl SignalParamsDir {
    pub fn from_array(array: &Array1<f64>) -> Self {
        if array.len() == 12 {
            Self {
                threshold: 0.0,
                c_range: Range::from(-1.0 .. 1.0),
                a_range: Range::from(-1.0 .. 1.0),
                derivative1_weight: array[0].max(0.0),
                derivative2_weight: array[1],
                imbalance1_weights: [array[2], array[3], array[4], array[5], array[6]],
                imbalance2_weights: [array[7], array[8], array[9], array[10], array[11]],
            }
        } else {
            Self {
                threshold: array[0],
                c_range: Range::from(-1.0 .. 1.0),
                a_range: Range::from(-1.0 .. 1.0),
                derivative1_weight: array[1].max(0.0),
                derivative2_weight: array[2],
                imbalance1_weights: [array[3], array[4], array[5], array[6], array[7]],
                imbalance2_weights: [array[8], array[9], array[10], array[11], array[12]],
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct SignalParams {
    pub hold_ms: u16,
    pub up: Option<SignalParamsDir>,
    pub down: Option<SignalParamsDir>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Signal {
    None,
    Buy1Sell2,
    Sell1Buy2,
}

impl SignalParams {
    pub fn signal_score(&self, values1: &OrderBookValues, values2: &OrderBookValues) -> Option<f64> {
        let params = if values1.std_derivative > 0.0 {&self.up} else {&self.down};
        if let Some(params) = params {
            let c = (values1.trend + values2.trend) * 0.5;
            let a = (values1.trend + values2.trend) / (values1.trend.abs() + values2.trend.abs());
            if params.c_range.contains(&c) && params.a_range.contains(&a) {
                let mut signal = params.derivative1_weight * values1.std_derivative.abs()
                    + params.derivative2_weight * values2.std_derivative;
                for i in 0..IMBALANCE_LEVELS.len() {
                    signal += params.imbalance1_weights[i] * values1.imbalances[i]
                        + params.imbalance2_weights[i] * values2.imbalances[i];
                }

                Some(signal - params.threshold)
            } else {
                None
            }
        } else { None }
    }

    pub fn signal_score_contributions(&self, values1: &OrderBookValues, values2: &OrderBookValues)
        -> Option<(f64, f64, f64, f64)> {
        let params = if values1.std_derivative > 0.0 {&self.up} else {&self.down};
        if let Some(params) = params {
            let deviations = params.derivative1_weight * values1.std_derivative.abs()
                + params.derivative2_weight * values2.std_derivative;
            let imbalances1 = (0..IMBALANCE_LEVELS.len()).map(|i| {
                params.imbalance1_weights[i] * values1.imbalances[i]
            }).sum::<f64>();
            let imbalances2 = (0..IMBALANCE_LEVELS.len()).map(|i| {
                params.imbalance2_weights[i] * values2.imbalances[i]
            }).sum::<f64>();
            let signal = deviations + imbalances1 + imbalances2;
            Some((signal - params.threshold, deviations, imbalances1, imbalances2))
        } else { None }
    }

    pub fn signal(&self, values1: &OrderBookValues, values2: &OrderBookValues) -> (Signal, f64) {
        if let Some(score) = self.signal_score(values1, values2) {
            if score > 0.0 {
                if values1.std_derivative < 0.0 {
                    (Signal::Buy1Sell2, score)
                } else {
                    (Signal::Sell1Buy2, score)
                }
            } else {
                (Signal::None, score)
            }
        } else {
            (Signal::None, 0.0)
        }
    }
}

pub struct TrendFilter;

impl TrendFilter {
    pub fn filter(values1: &OrderBookValues, values2: &OrderBookValues) -> bool {
        let c = (values1.trend + values2.trend) * 0.5;
        let a = (values1.trend + values2.trend) / (values1.trend.abs() + values2.trend.abs());

        if values1.std_derivative >= 0.0 {
            false
        } else {
            c < 0.0 && a < 0.0
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum CostFunctionImpl {
    TradingSimulation,
    SpearmanRanking,
    HuberLoss,
}

impl CostFunctionImpl {
    fn optimize_threshold(&self) -> bool {
        match self {
            CostFunctionImpl::TradingSimulation => true,
            CostFunctionImpl::SpearmanRanking => false,
            CostFunctionImpl::HuberLoss => false,
        }
    }
}

struct HuberProblem<'a> {
    direction: DealDirection,
    events: &'a [MarketEvent],
}

impl CostFunction for HuberProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let params = create_params_for_direction(p, self.direction);
        let values = collect_signal_to_pnl_values(self.events, &params);
        if values.signal.len() > 2 {
            Ok(huber_loss(&values.signal, &values.pnl, 0.75))
        } else {
            Err(Error::msg("Insufficient data"))
        }
    }
}

struct SpearmanProblem<'a> {
    direction: DealDirection,
    events: &'a [MarketEvent],
}

impl CostFunction for SpearmanProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let params = create_params_for_direction(p, self.direction);
        let values = collect_signal_to_pnl_values(self.events, &params);
        if values.signal.len() > 2 {
            Ok(-correlation::spearmanr(&values.pnl, &values.signal))
        } else {
            Err(Error::msg("Insufficient data"))
        }
    }
}

struct TradingProblem<'a> {
    direction: DealDirection,
    events: &'a [MarketEvent],
}

impl CostFunction for TradingProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let params = create_params_for_direction(p, self.direction);
        let results = run_simulation(self.events, &params, false);
        let net = results.income - results.outcome - results.commission;
        Ok(-net)
    }
}

fn create_params_for_direction(p: &Array1<f64>, direction: DealDirection) -> SignalParams {
    let dir_params = SignalParamsDir::from_array(p);
    match direction {
        DealDirection::Sell1Buy2 => SignalParams {
            hold_ms: DEFAULT_HOLD_MS,
            up: Some(dir_params),
            down: None,
        },
        DealDirection::Buy1Sell2 => SignalParams {
            hold_ms: DEFAULT_HOLD_MS,
            up: None,
            down: Some(dir_params),
        },
    }
}

pub fn calibrate_params(events: &[MarketEvent], cost_function: CostFunctionImpl, dir: DealDirection, hold_time: u16) -> SignalParams {
    let initial = Array1::from_vec(
        if cost_function.optimize_threshold() {
            //   A    D1   D2   I3   I5   I10  I20  I50  I3   I5   I10  I20  I50
            vec![3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        } else {
            //   D1   D2   I3   I5   I10  I20  I50  I3   I5   I10  I20  I50
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
    );

    let mut simplex = Vec::with_capacity(initial.len() + 1);

    simplex.push(initial.clone());

    for i in 0..initial.len() {
        let mut point = initial.clone();
        if cost_function.optimize_threshold() {
            point[i] += match i {
                0 => 1.0,      // A
                1 | 2 => 0.2,  // D1, D2
                _ => 0.1,      // Imbalance
            };
        } else {
            point[i] += match i {
                0 | 1 => 0.2,  // D1, D2
                _ => 0.1,      // Imbalance
            };
        }
        simplex.push(point);
    }

    match dir {
        DealDirection::Sell1Buy2 => {
            let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());
            info!("Start optimization UP on events {} - {}", events[0].event_datetime(), events[events.len() - 1].event_datetime());
            let params_up = optimize(events, cost_function, DealDirection::Sell1Buy2, solver);
            SignalParams {
                hold_ms: hold_time,
                up: params_up,
                down: None,
            }
        }
        DealDirection::Buy1Sell2 => {
            let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());
            info!("Start optimization DOWN on events {} - {}", events[0].event_datetime(), events[events.len() - 1].event_datetime());
            let params_down = optimize(events, cost_function, DealDirection::Buy1Sell2, solver);
            SignalParams {
                hold_ms: hold_time,
                up: None,
                down: params_down,
            }
        }
    }
}

fn optimize(events: &[MarketEvent], cost_function: CostFunctionImpl, direction: DealDirection, solver: NelderMead<Array1<f64>, f64>) -> Option<SignalParamsDir> {
    let executor_result = match cost_function {
        CostFunctionImpl::TradingSimulation => Executor::new(TradingProblem {direction, events}, solver)
            .configure(|state| state.max_iters(SOLVER_ITERATIONS))
            .run()
            .map(|result| {
                (result.state().get_best_cost(), result.state().get_best_param().cloned().unwrap_or_default())
            }),
        CostFunctionImpl::SpearmanRanking => Executor::new(SpearmanProblem {direction, events}, solver)
            .configure(|state| state.max_iters(SOLVER_ITERATIONS))
            .run()
            .map(|result| {
                (result.state().get_best_cost(), result.state().get_best_param().cloned().unwrap_or_default())
            }),
        CostFunctionImpl::HuberLoss => Executor::new(HuberProblem {direction, events}, solver)
            .configure(|state| state.max_iters(SOLVER_ITERATIONS))
            .run()
            .map(|result| {
                (result.state().get_best_cost(), result.state().get_best_param().cloned().unwrap_or_default())
            }),
    };
    let mut params = match executor_result {
        Ok((cost, params)) => {
            let params = SignalParamsDir::from_array(&params);
            info!("Best result {} with {params:?}", -cost);
            params
        },
        Err(e) => {
            error!("{e}");
            return None;
        },
    };
    Some(params)
}

#[derive(Clone)]
pub struct SignalPnL {
    pub signal: Vec<f64>,
    pub pnl: Vec<f64>,
    pub commission: Vec<f64>,
    pub contributions: Vec<[f64; 3]>,
}

pub fn collect_signal_to_pnl_values(events: &[MarketEvent], params: &SignalParams) -> SignalPnL {
    let mut signal_values = Vec::new();
    let mut pnl_values = Vec::new();
    let mut commission_values = Vec::new();
    let mut contributions_values = Vec::new();
    let (mut last_order_book1, mut last_order_book2) = (None, None);

    for (i, event) in events.iter().enumerate() {
        match event {
            MarketEvent::OrderBook1(values) => last_order_book1 = Some(*values),
            MarketEvent::OrderBook2(values) => last_order_book2 = Some(*values),
            _ => {},
        }

        if let Some(v1) = &last_order_book1 && let Some(v2) = &last_order_book2 &&
            let Some((signal, c0, c1, c2)) = params.signal_score_contributions(v1, v2) {
            contributions_values.push([c0, c1, c2]);
            let mut deal = if v1.std_derivative > 0.0 {
                Deal::sell1_buy2(v1, v2)
            } else {
                Deal::buy1_sell2(v1, v2)
            };
            if let Some((fv1, fv2)) = find_order_books_on_horizon(
                events, i,
                deal.get_open_time() + TimeDelta::milliseconds(params.hold_ms as i64)
            ) {
                let price = match deal.get_direction() {
                    DealDirection::Sell1Buy2 => fv1.bid,
                    DealDirection::Buy1Sell2 => fv1.ask,
                };
                deal.close_instrument1(price, fv1.time);
                let price = match deal.get_direction() {
                    DealDirection::Sell1Buy2 => fv2.ask,
                    DealDirection::Buy1Sell2 => fv2.bid,
                };
                deal.close_instrument2(price, fv2.time);
                let (_, revenue, cost) = deal.close(false);
                signal_values.push(signal);
                pnl_values.push(revenue - cost);
                commission_values.push(commission(revenue, cost));
            }
        }
    }
    SignalPnL {
        signal: signal_values,
        pnl: pnl_values,
        commission: commission_values,
        contributions: contributions_values,
    }
}

#[derive(Clone)]
struct ThresholdProblem<'a> {
    events: &'a [MarketEvent],
    params: SignalParams,
    dir: DealDirection,
}

impl ThresholdProblem<'_> {
    fn apply_param(&self, param: &Array1<f64>) -> SignalParams {
        let mut threshold_params = self.params;
        let dir_params = match self.dir {
            DealDirection::Sell1Buy2 => threshold_params.up.as_mut().unwrap(),
            DealDirection::Buy1Sell2 => threshold_params.down.as_mut().unwrap(),
        };
        dir_params.threshold = param[0];
        dir_params.c_range = Range::from(param[1] .. param[3]);
        dir_params.a_range = Range::from(param[2] .. param[4]);
        threshold_params
    }
}

impl CostFunction for ThresholdProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let results = run_simulation(self.events, &self.apply_param(param), false);
        Ok(results.income - results.outcome)
    }
}

pub fn calibrate_threshold(events: &[MarketEvent], params_dir: SignalParamsDir, dir: DealDirection) -> Option<SignalParams> {
    let params = SignalParams {
        hold_ms: DEFAULT_HOLD_MS,
        up: if matches!(dir, DealDirection::Sell1Buy2) {Some(params_dir)} else {None},
        down: if matches!(dir, DealDirection::Buy1Sell2) {Some(params_dir)} else {None},
    };

    let initial = Array1::from_vec(
           //   Threshold C_min  A_min  C_max A_max
        vec![0.1,      -0.01, -0.01, 0.01, 0.01]
    );

    let mut simplex = Vec::with_capacity(initial.len() + 1);

    simplex.push(initial.clone());

    for i in 0..initial.len() {
        let mut point = initial.clone();
        point[i] += match i {
            0 => 0.025,      // Threshold
            1 | 2 => 0.0025, // C, A min
            _ => -0.0025,    // C, A max
        };
        simplex.push(point);
    }

    let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());
    let problem = ThresholdProblem {events, params, dir};
    let executor = Executor::new(problem.clone(), solver)
        .configure(|state| state.max_iters(SOLVER_ITERATIONS));
    info!("Calibrate threshold {dir:?} on events {} - {}", events[0].event_datetime(), events[events.len() - 1].event_datetime());
    let (result, param) = executor
        .run()
        .map(|result| {
            (result.state().get_best_cost(), result.state().get_best_param().cloned().unwrap_or_default())
        }).ok()?;
    let dir_params = problem.apply_param(&param);
    info!("Best result {result} with {dir_params:?}");
    Some(dir_params)
}

pub fn find_threshold(events: &[MarketEvent], params_dir: SignalParamsDir, dir: DealDirection) -> Option<f64> {
    let params = SignalParams {
        hold_ms: DEFAULT_HOLD_MS,
        up: if matches!(dir, DealDirection::Sell1Buy2) {Some(params_dir)} else {None},
        down: if matches!(dir, DealDirection::Buy1Sell2) {Some(params_dir)} else {None},
    };
    let values = collect_signal_to_pnl_values(events, &params);
    let mut filtered_signal_values = Vec::new();
    for i in 0..values.signal.len() {
        if values.pnl[i] > -20.0 {
            let signal = values.signal[i];
            filtered_signal_values.push(signal);
        }
    }

    const LEVELS: usize = 16384;

    if filtered_signal_values.len() > 1000 {
        let expected_deals = market_events_time_diapason(events).num_minutes() as u32 / 60;
        info!("Expected deals {expected_deals}");

        filtered_signal_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_signal = filtered_signal_values[5];
        let max_signal = filtered_signal_values[filtered_signal_values.len() - 5];
        info!("Signal min {min_signal} and max {max_signal}");
        let mut best_profit = f64::NAN;
        let mut best_threshold = 0.0;

        for i in 0..LEVELS {
            let threshold = min_signal + (max_signal - min_signal) * (i as f64) / LEVELS as f64;
            let mut test_params_dir = params_dir;
            test_params_dir.threshold = threshold;
            let test_params = SignalParams {
                hold_ms: DEFAULT_HOLD_MS,
                up: if matches!(dir, DealDirection::Sell1Buy2) {Some(test_params_dir)} else {None},
                down: if matches!(dir, DealDirection::Buy1Sell2) {Some(test_params_dir)} else {None},
            };
            let result = run_simulation(events, &test_params, false);
            if result.win + result.loss > expected_deals {
                let net = result.income - result.outcome - result.commission;
                if best_profit.is_nan() || net > best_profit {
                    best_profit = net;
                    best_threshold = threshold;
                }
            }
        }
        if best_profit.is_nan() {
            None
        } else {
            info!("Best profit {best_profit} with threshold {best_threshold}");
            Some(best_threshold)
        }
    } else {
        None
    }
}