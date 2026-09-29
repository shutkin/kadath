use crate::{ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::Datelike;
use log::info;
use model::events::OrderBookEvent;
use ndarray::Array1;
use model::common::CommonError;
use model::math_util::standard_deviation;
use crate::simulation::SimDeal;

pub struct StrategyOptimizer {
    instrument1_id: i32,
    instrument2_id: i32,
}

impl StrategyOptimizer {
    pub fn new(instrument1_id: i32, instrument2_id: i32) -> Self {
        Self {
            instrument1_id,
            instrument2_id,
        }
    }

    pub fn optimize(&self, events: &[OrderBookEvent]) -> Result<ModelConfig, CommonError> {
        let initial = Array1::from(
            // train_size, decay, stddiv_window
            vec![60.0, 15.0, 1200.0]
        );
        let mut simplex = Vec::with_capacity(initial.len() + 1);

        simplex.push(initial.clone());

        for i in 0..initial.len() {
            let mut point = initial.clone();
            point[i] += match i {
                0 => 10.0, // Train size
                1 => 5.0,  // Decay
                _ => 30.0, // stddiv window
            };
            simplex.push(point);
        }

        let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());
        let problem = GlobalProblem { events, instrument1_id: self.instrument1_id, instrument2_id: self.instrument2_id };
        let result = Executor::new(problem, solver)
            .configure(|state| state.max_iters(128))
            .run()?;
        let optimized_score = -result.state().get_best_cost();
        let param = result.state().get_best_param().ok_or("Failed to get best param")?;
        let config = config_from_param(param);
        info!("Best score {optimized_score} with {config:?}");

        Ok(config)
    }
}

struct GlobalProblem<'a> {
    events: &'a [OrderBookEvent],
    instrument1_id: i32,
    instrument2_id: i32,
}

impl CostFunction for GlobalProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let config = config_from_param(param);
        info!("Check {config:?}");
        let model = WalkForwardModel::new_with_config(
            self.instrument1_id, self.instrument2_id, config,
        );
        let mut window_data = WindowData::new();

        let mut daily_pnl = Vec::new();
        let (mut daily_revenue, mut daily_cost) = (0.0, 0.0);
        let mut active_deal = Option::<SimDeal>::None;

        let mut prev_event_time = self.events[0].order_book.timestamp;

        for event in self.events {
            if event.order_book.timestamp.num_days_from_ce() > prev_event_time.num_days_from_ce() {
                daily_pnl.push(daily_revenue - daily_cost);
                daily_revenue = 0.0;
                daily_cost = 0.0;
            }
            prev_event_time = event.order_book.timestamp;

            model.calibrate(true);
            let signal = model.process(&mut window_data, event);
            if let Some((v1, v2)) = model.get_last_values() {
                if let Some(deal) = active_deal.as_mut() {
                    deal.close1(&v1, &v2);
                    deal.close2(&v1, &v2);
                    if let Some((revenue, cost)) = deal.try_finish() {
                        daily_revenue += revenue;
                        daily_cost += cost;
                        active_deal = None;
                    }
                } else {
                    match signal {
                        TradeSignal::None => {}
                        TradeSignal::Sell1Buy2(hold) => {
                            active_deal = Some(SimDeal::sell1_buy2(&v1, &v2, signal, hold));
                        }
                        TradeSignal::Buy1Sell2(hold) => {
                            active_deal = Some(SimDeal::buy1_sell2(&v1, &v2, signal, hold));
                        }
                    }
                }
            }
        }
        if daily_revenue > 0.0 || daily_cost > 0.0 {
            daily_pnl.push(daily_revenue - daily_cost);
        }

        let max_pnl = daily_pnl.iter().copied().reduce(f64::max).unwrap_or_default();
        let pnl_deviation = standard_deviation(&daily_pnl).unwrap_or_default();
        let score = max_pnl - pnl_deviation;
        info!("Score: {score}");

        Ok(-score)
    }
}

fn config_from_param(param: &Array1<f64>) -> ModelConfig {
    ModelConfig {
        train_data_minutes: param[0].abs().round() as u16,
        decay_time: param[1].abs(),
        std_diapason_s: param[2].abs().round() as u16,
        ..ModelConfig::default()
    }
}