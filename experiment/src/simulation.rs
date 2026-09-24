use crate::deal::{Deal, DealDirection};
use crate::signal_optimization::{Signal, SignalParams};
use crate::{MarketEvent, OrderBookValues, commission};
use chrono::{DateTime, TimeDelta, Utc};

pub struct SimulationResult {
    pub win: u32,
    pub loss: u32,
    pub income: f64,
    pub outcome: f64,
    pub commission: f64,
}


fn get_close_price_from_values1(deal: &Deal, values: &OrderBookValues) -> f64 {
    match deal.get_direction() {
        DealDirection::Sell1Buy2 => values.bid,
        DealDirection::Buy1Sell2 => values.ask,
    }
}

fn get_close_price_from_values2(deal: &Deal, values: &OrderBookValues) -> f64 {
    match deal.get_direction() {
        DealDirection::Sell1Buy2 => values.ask,
        DealDirection::Buy1Sell2 => values.bid,
    }
}

pub fn find_order_books_on_horizon(
    events: &[MarketEvent],
    mut index: usize,
    target_time: DateTime<Utc>,
) -> Option<(OrderBookValues, OrderBookValues)> {
    let (mut v1, mut v2) = (None, None);
    while index < events.len() {
        match events[index] {
            MarketEvent::OrderBook1(v) => {
                if v.time >= target_time {
                    v1 = Some(v);
                }
            }
            MarketEvent::OrderBook2(v) => {
                if v.time >= target_time {
                    v2 = Some(v);
                }
            }
            _ => {}
        }
        if let Some(v1) = v1 && let Some(v2) = v2 {
            return Some((v1, v2));
        }
        index += 1;
    }
    None
}

pub trait DealHandler {
    fn handle_deal(&mut self, events: &[MarketEvent], event_index: usize, deal: Deal);
    fn get_stats(&self) -> Vec<(String, String)>;
    fn init(&mut self, variant: u8);
}

pub fn run_stats(events: &[MarketEvent], params: &SignalParams, deal_handler: &mut dyn DealHandler) -> Vec<(String, String)> {
    let (mut last_order_book1, mut last_order_book2) = (None, None);
    let mut prev_signal = Signal::None;

    for (index, event) in events.iter().enumerate() {
        match event {
            MarketEvent::OrderBook1(values) => last_order_book1 = Some(*values),
            MarketEvent::OrderBook2(values) => last_order_book2 = Some(*values),
        }

        if let Some(values1) = &last_order_book1 && let Some(values2) = &last_order_book2 {
            let (cur_signal, _) = params.signal(values1, values2);

            if cur_signal != prev_signal &&
                let Some(deal) = match cur_signal {
                    Signal::Buy1Sell2 => Some(Deal::buy1_sell2(values1, values2)),
                    Signal::Sell1Buy2 => Some(Deal::sell1_buy2(values1, values2)),
                    _ => None,
                } { deal_handler.handle_deal(events, index, deal); }

            prev_signal = cur_signal;
        }
    }
    deal_handler.get_stats()
}

pub fn run_simulation(events: &[MarketEvent], params: &SignalParams, log: bool) -> SimulationResult {
    let (mut win, mut loss) = (0, 0);
    let (mut total_revenue, mut total_cost, mut total_commission) = (0.0, 0.0, 0.0);
    let mut cur_deal = Option::<Deal>::None;
    let (mut last_order_book1, mut last_order_book2) = (None, None);
    let mut prev_signal = Signal::None;
    //let mut scores = Vec::new();

    for event in events {
        match event {
            MarketEvent::OrderBook1(values) => last_order_book1 = Some(*values),
            MarketEvent::OrderBook2(values) => last_order_book2 = Some(*values),
            _ => {},
        }

        if let Some(deal) = cur_deal.as_mut() {
            if let Some(v) = last_order_book1 &&
                v.time > deal.get_open_time() + TimeDelta::milliseconds(params.hold_ms as i64) {
                let price = match deal.get_direction() {
                    DealDirection::Sell1Buy2 => v.ask,
                    DealDirection::Buy1Sell2 => v.bid,
                };
                deal.close_instrument1(price, v.time)
            }
            if let Some(v) = last_order_book2 &&
                v.time > deal.get_open_time() + TimeDelta::milliseconds(params.hold_ms as i64) {
                let price = match deal.get_direction() {
                    DealDirection::Sell1Buy2 => v.bid,
                    DealDirection::Buy1Sell2 => v.ask,
                };
                deal.close_instrument2(price, v.time)
            }
            if deal.is_completed() {
                let (quantity, revenue, cost) = deal.close(log);
                if revenue > cost {win += 1} else {loss += 1};
                total_revenue += revenue;// * quantity as f64;
                total_cost += cost;// * quantity as f64;
                total_commission += commission(revenue, cost); // * quantity as f64;
                cur_deal = None;
            }
        }

        if let Some(values1) = &last_order_book1 && let Some(values2) = &last_order_book2 {
            let (cur_signal, score) = params.signal(values1, values2);

            if cur_deal.is_none() &&
                //(!apply_filters || TrendFilter::filter(values1, values2) && filter_by_scores(score, &scores)) &&
                cur_signal != prev_signal {
                cur_deal = match cur_signal {
                    Signal::None => None,
                    Signal::Buy1Sell2 => {
                        Some(Deal::buy1_sell2(values1, values2))
                    },
                    Signal::Sell1Buy2 => {
                        Some(Deal::sell1_buy2(values1, values2))
                    },
                };
            }

            prev_signal = cur_signal;
            
            /*if apply_filters {
                scores.push(score);
                if scores.len() > 32768 {
                    scores.remove(0);
                }
            }*/
        }
    }
    SimulationResult {
        win,
        loss,
        income: total_revenue,
        outcome: total_cost,
        commission: total_commission,
    }
}

fn filter_by_scores(score: f64, scores: &[f64]) -> bool {
    let offset_from_top = (scores.len() as f64 * 0.0075 / 100.0).round() as usize;
    if offset_from_top < 2 {
        return false;
    }

    let mut scores = scores.to_vec();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = scores[scores.len() - 1 - offset_from_top];
    score >= threshold
}
