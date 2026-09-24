use crate::HOLD_TIME_VARIANTS;
use crate::orderbook_values::OrderBookValues;
use crate::signal_calculator::{CalculatorsPair, SignalCalculator, SignalPerformance};
use crate::simulation::run_simulation;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use log::{debug, error, warn};
use model::common::CommonError;
use ndarray::Array1;

struct TradeSimProblem<'a> {
    values: &'a [OrderBookValues],
    is_up: bool,
    hold_time_ms: u16,
    tau_time: f64,
}

impl CostFunction for TradeSimProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let last_time = self.values[self.values.len() - 1].time;
        let calculator = SignalCalculator::from_optimizer_param(self.hold_time_ms, param);
        let pair = if self.is_up {
            CalculatorsPair::new_up(calculator, last_time)
        } else {
            CalculatorsPair::new_down(calculator, last_time)
        };

        let (_, profit) = run_simulation(self.values, &pair, self.tau_time);
        Ok(-profit)
    }
}

pub fn calibrate_signal_calculator(values: &[OrderBookValues], is_up: bool, decay_time: f64, fixed_hold_time: Option<u16>) -> Option<SignalCalculator> {
    let mut best_profit = 0.0;
    let mut best_calculator = None;
    
    for hold_time_ms in HOLD_TIME_VARIANTS {
        if let Some(fixed_hold) = fixed_hold_time && hold_time_ms != fixed_hold {
            continue;
        }

        let calc = match calibrate_signal_calculator_on_hold_time(values, is_up, hold_time_ms, decay_time) {
            Ok(calc) => calc,
            Err(err) => {
                error!("Failed to optimize {} calculator: {err}", if is_up { "up" } else { "down" });
                None
            }
        };
        if let Some(calc) = calc {
            let profit = calc.get_train_profit();
            if profit > best_profit {
                best_profit = profit;
                best_calculator = Some(calc);
            }
        }
    }
    
    best_calculator
}

fn calibrate_signal_calculator_on_hold_time(
    values: &[OrderBookValues],
    is_up: bool,
    hold_time_ms: u16,
    decay_time: f64,
) -> Result<Option<SignalCalculator>, CommonError> {
    let initial = Array1::from_vec(
        //      A    D1   D2   I3   I5   I10  I20  I50  I3   I5   I10  I20  I50
        vec![3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    let mut simplex = Vec::with_capacity(initial.len() + 1);

    simplex.push(initial.clone());

    for i in 0..initial.len() {
        let mut point = initial.clone();
        point[i] += match i {
            0 => 1.0,      // A
            1 | 2 => 0.2,  // D1, D2
            _ => 0.1,      // Imbalance
        };
        simplex.push(point);
    }

    let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());

    /*let solver = ParticleSwarm::new(
        (Array1::from_vec(vec![0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        Array1::from_vec(vec![10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
        72
    );*/

    let problem = TradeSimProblem { values, is_up, hold_time_ms, tau_time: decay_time };

    debug!("Start optimization {} on events {} - {}",
        if is_up {"UP"} else {"DOWN"}, values[0].time, values[values.len() - 1].time);
    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(4096))
        .run()?;
    let optimized_profit = -result.state().get_best_cost();
    let param = result.state().get_best_param().ok_or("Failed to get best param")?;

    debug!("{} opt profit {optimized_profit} with {param:?}", if is_up {"UP"} else {"DOWN"});

    let mut calc = SignalCalculator::from_optimizer_param(hold_time_ms, param);
    let pair = if is_up {
        CalculatorsPair::new_up(calc, values[values.len() - 1].time)
    } else {
        CalculatorsPair::new_down(calc, values[values.len() - 1].time)
    };

    let (deals, profit) = run_simulation(values, &pair, decay_time);
    if profit != optimized_profit {
        warn!("{} opt profit {optimized_profit} but verified is {profit}", if is_up {"UP"} else {"DOWN"});
    }

    calc.set_performance(SignalPerformance {
        chosen_hold_time: calc.get_hold_time_ms(),
        training_total_pnl: profit,
        training_deals: deals,
        actual_total_pnl: 0.0,
        actual_deals: 0,
        actual_wins: 0,
    });

    //Ok(Some(calc))
    if profit > 1.0 {
        Ok(Some(calc))
    } else {
        Ok(None)
    }
}
