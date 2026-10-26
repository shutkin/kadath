use crate::{DealData, ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use argmin::core::{CostFunction, Error};
use chrono::{DateTime, Datelike, Utc};
use log::{debug, info, warn};
use model::common::CommonError;
use model::events::OrderBookEvent;
use model::math_util::{mean, standard_deviation, ExponentFunction};
use model::{OrderBook, TradeDirection};
use rayon::prelude::*;
use std::cmp::Ordering;

pub struct StrategyOptimizer {
    instrument_id: i32,
}

const POINTS: usize = 8;
const FULL_CYCLES: usize = 6;

const PARAM_NAME: [&str; 8] = [
    "Decay",
    "StdDiv Window",
    "Trend Tau",
    "Price Change Filter",
    "Prediction Strength",
    "Predicted Price Change",
    "Prediction Probability Up to Flat",
    "Prediction Probability Up to Down",
];

impl StrategyOptimizer {
    pub fn new(instrument_id: i32) -> Self {
        Self {
            instrument_id,
        }
    }

    pub fn optimize(&self, events: &[OrderBookEvent], hold_time: i32, train_time_minutes: u16) -> Result<ModelConfig, CommonError> {
        let param_functions = [
            Some(ExponentFunction::find(70.0, 300.0, 1300.0)),
            Some(ExponentFunction::find(60.0, 250.0, 1100.0)),
            Some(ExponentFunction::find(20.0, 80.0, 400.0)),
            None,
            None,
            None,
            None,
            None,
        ];
        let mut diapasons = [
            -1.0..1.0, // decay
            -1.0..1.0, // stddiv window len
            -1.0..1.0, // trend tau
            0.15 - 0.6..0.15 + 0.6, // price change filter
            0.0015 - 0.0014..0.0015 + 0.0015, // prediction strength
            0.3 - 0.4..0.3 + 0.4, // predicted change
            -0.75 - 0.5..-0.75 + 0.5, // prediction probability up to flat
            0.08 - 0.1..0.08 + 0.1, // prediction probability up to down
        ];

        let problem = GlobalProblem { events, param_functions: &param_functions, instrument_id: self.instrument_id, hold_time, train_time_minutes };

        let mut best_score = f64::NEG_INFINITY;
        let mut best_config = None;

        let mut cycle = 0;
        let mut param_index = 0;
        let mut best_param_points = [0.0; PARAM_NAME.len()];
        loop {
            let full_range = diapasons[param_index].end - diapasons[param_index].start;

            let point_params_score = (0..POINTS).into_par_iter().map(|point_index| {
                let point = diapasons[param_index].start + point_index as f64 * full_range / (POINTS - 1) as f64;

                let params = diapasons.iter().enumerate()
                    .map(|(i, d)| {
                        if i == param_index {
                            point
                        } else {
                            (d.start + d.end) * 0.5
                        }
                    }).collect::<Vec<_>>();
                problem.cost(&params).ok().map(|score| {
                    (point, params, score)
                })
            }).filter_map(|opt| opt).collect::<Vec<_>>();

            let min_score = point_params_score.iter().map(|(_, _, score)| *score).reduce(f64::min).unwrap_or_default();
            let (mut sum, mut sum_score) = (0.0, 0.0);
            for (point, params, score) in &point_params_score {
                if *score > best_score {
                    best_score = *score;
                    let config = config_from_param(&params, &param_functions, hold_time, train_time_minutes);
                    info!("New best {best_score} with {config:?}");
                    best_config = Some(config);
                }
                let score_from_min = score - min_score;
                let score_squared = score_from_min * score_from_min;
                sum += point * score_squared;
                sum_score += score_squared;
            }
            if sum_score < 0.1 {
                warn!("Scores are equal!");
            } else {
                if let Some((best_param_point, _, _)) = point_params_score.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal)) {
                    best_param_points[param_index] = *best_param_point;
                }

                let next_center = sum / sum_score;
                let next_range = full_range * 0.6;
                diapasons[param_index] = next_center - next_range * 0.5 .. next_center + next_range * 0.5;
                if let Some(function) = &param_functions[param_index] {
                    info!(
                        "{} center {}, diapason {}..{}",
                        PARAM_NAME[param_index],
                        function.calculate(next_center),
                        function.calculate(diapasons[param_index].start),
                        function.calculate(diapasons[param_index].end),
                    );
                } else {
                    info!(
                        "{} center {}, diapason {}..{}",
                        PARAM_NAME[param_index],
                        next_center,
                        diapasons[param_index].start,
                        diapasons[param_index].end,
                    );
                }
            }

            param_index += 1;
            if param_index == diapasons.len() {
                param_index = 0;
                cycle += 1;
                let config = config_from_param(&best_param_points, &param_functions, hold_time, train_time_minutes);
                let optimal_score = problem.cost(&best_param_points.to_vec()).unwrap_or_default();
                info!("Cycle {cycle} optimal score {optimal_score} with {config:?}");
                if cycle == FULL_CYCLES {
                    break;
                }
            }
        }

        if let Ok(optimal_score) = problem.cost(&best_param_points.to_vec()) && let Some(best_config) = best_config {
            let optimal_config = config_from_param(&best_param_points, &param_functions, hold_time, train_time_minutes);
            info!("Optimal {optimal_config:?} score {optimal_score}");
            info!("Best {best_config:?} score {best_score}");
            if optimal_score > best_score {
                Ok(optimal_config)
            } else {
                Ok(best_config)
            }
        } else {
            Err("failed to optimize model config".into())
        }
    }
}

struct GlobalProblem<'a> {
    param_functions: &'a [Option<ExponentFunction>],
    events: &'a [OrderBookEvent],
    instrument_id: i32,
    hold_time: i32,
    train_time_minutes: u16,
}

struct SimDeal {
    open_at: DateTime<Utc>,
    direction: TradeDirection,
    open_price: f64,
    deal_data: DealData,
}

impl CostFunction for GlobalProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let config = config_from_param(param, self.param_functions, self.hold_time, self.train_time_minutes);
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
                debug!("Daily PnL {} and commission {daily_commission}", daily_revenue - daily_cost);
                daily_pnl.push(daily_revenue - daily_cost - daily_commission);
                daily_revenue = 0.0;
                daily_cost = 0.0;
                daily_commission = 0.0;
            }
            prev_event_time = event.order_book.timestamp;

            let cur_ob = event.order_book.clone();

            model.calibrate(true);
            let signal = model.process(&mut window_data, event);
            if let Some(deal) = active_deal.as_mut() {
                let close_price = match deal.direction {
                    TradeDirection::Buy => get_bid_price(&cur_ob),
                    TradeDirection::Sell => get_ask_price(&cur_ob),
                };

                if model.is_it_time_to_close_deal(&cur_ob, &mut deal.deal_data) {
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
                        let deal_data = DealData::new(&cur_ob, TradeDirection::Buy);
                        active_deal = Some(SimDeal { open_at: cur_ob.timestamp, deal_data, direction: TradeDirection::Buy, open_price: get_ask_price(&cur_ob) });
                    }
                    TradeSignal::Sell(hold) => {
                        let deal_data = DealData::new(&cur_ob, TradeDirection::Buy);
                        active_deal = Some(SimDeal { open_at: cur_ob.timestamp, deal_data, direction: TradeDirection::Sell, open_price: get_bid_price(&cur_ob) });
                    }
                }
            }
        }
        if daily_revenue > 0.0 || daily_cost > 0.0 {
            debug!("Daily PnL {} and commission {daily_commission}", daily_revenue - daily_cost);
            daily_pnl.push(daily_revenue - daily_cost - daily_commission);
        }

        let score = if daily_pnl.len() == 1 {
            daily_pnl[0]
        } else {
            let mean_pnl = mean(&daily_pnl);
            let pnl_deviation = standard_deviation(&daily_pnl).unwrap_or_default();
            mean_pnl - pnl_deviation
        };
        info!("Score: {score}");

        // Must be negative for Nelder-Mead!!
        Ok(score)
    }
}

fn config_from_param(param: &[f64], functions: &[Option<ExponentFunction>], hold_time_ms: i32, train_time: u16) -> ModelConfig {

    let calc = move |i: usize| {
        functions[i].as_ref().map(|func| func.calculate(param[i])).unwrap_or(param[i])
    };

    ModelConfig {
        decay_time: calc(0),
        derivatives_normalize_diapason_s: calc(1),
        trend_tau_s: calc(2),

        price_change_filter: calc(3),
        prediction_strength: calc(4),
        predicted_change: calc(5),
        prediction_probability_up_to_flat: calc(6),
        prediction_probability_up_to_down: calc(7),

        train_interval_minutes: 1,
        train_data_minutes: train_time,
        hold_time_ms,
    }
}

fn get_ask_price(book: &OrderBook) -> f64 {
    book.asks.iter().map(|data| data.price).min().unwrap_or_default().as_f64()
}

fn get_bid_price(book: &OrderBook) -> f64 {
    book.bids.iter().map(|data| data.price).max().unwrap_or_default().as_f64()
}

fn get_mid_price(order_book: &OrderBook) -> f64 {
    let best_ask = order_book.asks.iter().map(|data| data.price).min().unwrap().as_f64();
    let best_bid = order_book.bids.iter().map(|data| data.price).max().unwrap().as_f64();
    (best_ask + best_bid) * 0.5
}
