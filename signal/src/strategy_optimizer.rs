use crate::{ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::{DateTime, Datelike, TimeDelta, Utc};
use log::info;
use model::events::OrderBookEvent;
use ndarray::Array1;
use model::common::CommonError;
use model::math_util::{mean, standard_deviation};
use model::{OrderBook, TradeDirection};

pub struct StrategyOptimizer {
    instrument_id: i32,
}

impl StrategyOptimizer {
    pub fn new(instrument_id: i32) -> Self {
        Self {
            instrument_id,
        }
    }

    pub fn optimize(&self, events: &[OrderBookEvent], hold_time: i32, train_time_minutes: u16) -> Result<ModelConfig, CommonError> {
        let initial = Array1::from(
            // decay, stddiv_window
            vec![
                0.0,   // decay
                0.0,   // stddiv window len
                0.0,   // trend tau
                0.0,   // price_change_filter
                0.0035,// prediction_strength
                0.2,   // predicted_change
                -0.65, // prediction_probability_up_to_flat
                0.056, // prediction_probability_up_to_down
            ]
        );
        let mut simplex = Vec::with_capacity(initial.len() + 1);

        simplex.push(initial.clone());

        for i in 0..initial.len() {
            let mut point = initial.clone();
            point[i] += match i {
                0 | 1 | 2 => 0.2,  // decay, stddiv window len, trend tau
                3 => 0.5, // price change filter
                4 => 0.001, // prediction strength
                5 => 0.1, // prediction change
                6 => 0.05, // prediction probability up to flat
                _ => 0.01, // prediction probability up to down
            };
            simplex.push(point);
        }

        let solver = NelderMead::<Array1<f64>, f64>::new(simplex.clone());
        let problem = GlobalProblem { events, instrument_id: self.instrument_id, hold_time, train_time_minutes };
        let result = Executor::new(problem, solver)
            .configure(|state| state.max_iters(128))
            .run()?;
        let optimized_score = -result.state().get_best_cost();
        let param = result.state().get_best_param().ok_or("Failed to get best param")?;
        let config = config_from_param(param, hold_time, train_time_minutes);
        info!("Best score {optimized_score} with {config:?}");

        Ok(config)
    }
}

struct GlobalProblem<'a> {
    events: &'a [OrderBookEvent],
    instrument_id: i32,
    hold_time: i32,
    train_time_minutes: u16,
}

struct SimDeal {
    close_time: DateTime<Utc>,
    direction: TradeDirection,
    open_price: f64,
}

impl CostFunction for GlobalProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let config = config_from_param(param, self.hold_time, self.train_time_minutes);
        info!("Check {config:?}");
        let model = WalkForwardModel::new_buying(
            self.instrument_id, config,
        );
        let mut window_data = WindowData::new();

        let mut daily_pnl = Vec::new();
        let (mut daily_revenue, mut daily_cost, mut daily_commission) = (0.0, 0.0, 0.0);

        let mut prev_event_time = self.events[0].order_book.timestamp;

        let mut active_deal = Option::<SimDeal>::None;

        for event in self.events {
            if event.order_book.timestamp.num_days_from_ce() > prev_event_time.num_days_from_ce() {
                info!("Daily PnL {} and commission {daily_commission}", daily_revenue - daily_cost);
                daily_pnl.push(daily_revenue - daily_cost - daily_commission);
                daily_revenue = 0.0;
                daily_cost = 0.0;
                daily_commission = 0.0;
            }
            prev_event_time = event.order_book.timestamp;

            let cur_ob = event.order_book.clone();

            model.calibrate(true);
            let signal = model.process(&mut window_data, event);
            if let Some(deal) = active_deal.as_ref() {
                if deal.close_time <= cur_ob.timestamp {
                    let close_price = get_mid_price(&cur_ob);
                    let (revenue, cost) = match deal.direction {
                        TradeDirection::Buy => (close_price, deal.open_price),
                        TradeDirection::Sell => (deal.open_price, close_price),
                    };
                    daily_revenue += revenue;
                    daily_cost += cost;
                    daily_commission += 0.5;
                    active_deal = None;
                }
            } else {
                match signal {
                    TradeSignal::None => {}
                    TradeSignal::Buy(hold) => {
                        let close_time = cur_ob.timestamp + TimeDelta::milliseconds(hold as i64);
                        active_deal = Some(SimDeal { close_time, direction: TradeDirection::Buy, open_price: get_mid_price(&cur_ob) });
                    }
                    TradeSignal::Sell(hold) => {
                        let close_time = cur_ob.timestamp + TimeDelta::milliseconds(hold as i64);
                        active_deal = Some(SimDeal { close_time, direction: TradeDirection::Sell, open_price: get_mid_price(&cur_ob) });
                    }
                }
            }
        }
        if daily_revenue > 0.0 || daily_cost > 0.0 {
            info!("Daily PnL {} and commission {daily_commission}", daily_revenue - daily_cost);
            daily_pnl.push(daily_revenue - daily_cost - daily_commission);
        }

        let mean_pnl = mean(&daily_pnl);
        let pnl_deviation = standard_deviation(&daily_pnl).unwrap_or_default();
        let score = mean_pnl - pnl_deviation;
        info!("Score: {score}");

        Ok(-score)
    }
}

fn config_from_param(param: &Array1<f64>, hold_time_ms: i32, train_time: u16) -> ModelConfig {
    ModelConfig {
        decay_time: (param[0] * 1.7006 + 6.3875).exp(), // x=-1: y=108.5, x=0: y=594.5, x=1: y=3250
        derivatives_normalize_diapason_s: (param[1] * 1.4979 + 4.8619).exp(), // x=-1: y=28.9, x=0: y=129.3, x=1: y=578.7
        trend_tau_s: (param[2] * 1.4979 + 4.8619).exp(), // x=-1: y=28.9, x=0: y=129.3, x=1: y=578.7

        price_change_filter: param[3],
        prediction_strength: param[4],
        predicted_change: param[5],
        prediction_probability_up_to_flat: param[6],
        prediction_probability_up_to_down: param[7],

        train_interval_minutes: 1,
        train_data_minutes: train_time,
        hold_time_ms,
    }
}

fn get_mid_price(order_book: &OrderBook) -> f64 {
    let best_ask = order_book.asks.iter().map(|data| data.price).min().unwrap().as_f64();
    let best_bid = order_book.bids.iter().map(|data| data.price).max().unwrap().as_f64();
    (best_ask + best_bid) * 0.5
}
