use chrono::{DateTime, TimeDelta, Utc};
use crate::orderbook_values::{Leg, OrderBookValues};
use crate::signal_calculator::CalculatorsPair;
use crate::TradeSignal;

#[derive(Copy, Clone)]
pub struct SimDeal {
    signal: TradeSignal,
    open_time: DateTime<Utc>,
    close_time: DateTime<Utc>,
    open_price1: f64,
    open_price2: f64,
    close_price1: Option<f64>,
    close_price2: Option<f64>,
}

impl SimDeal {
    pub fn sell1_buy2(v1: &OrderBookValues, v2: &OrderBookValues, signal: TradeSignal, hold: u16) -> Self {
        let cur_time = v1.time.max(v2.time);
        SimDeal {
            signal,
            open_time: cur_time,
            close_time: cur_time + TimeDelta::milliseconds(hold as i64),
            open_price1: v1.bid,
            open_price2: v2.ask,
            close_price1: None,
            close_price2: None,
        }
    }
    
    pub fn buy1_sell2(v1: &OrderBookValues, v2: &OrderBookValues, signal: TradeSignal, hold: u16) -> Self {
        let cur_time = v1.time.max(v2.time);
        SimDeal {
            signal,
            open_time: cur_time,
            close_time: cur_time + TimeDelta::milliseconds(hold as i64),
            open_price1: v1.ask,
            open_price2: v2.bid,
            close_price1: None,
            close_price2: None,
        }
    }

    pub fn close1(&mut self, v1: &OrderBookValues, v2: &OrderBookValues) {
        if self.close_price1.is_none() && v1.time >= self.close_time {
            self.close_price1 = Some(match self.signal {
                TradeSignal::Sell1Buy2(_) => v1.ask,
                TradeSignal::Buy1Sell2(_) => v1.bid,
                TradeSignal::None => unreachable!(),
            });
        }
    }

    pub fn close2(&mut self, v1: &OrderBookValues, v2: &OrderBookValues) {
        if self.close_price2.is_none() && v2.time >= self.close_time {
            self.close_price2 = Some(match self.signal {
                TradeSignal::Sell1Buy2(_) => v2.bid,
                TradeSignal::Buy1Sell2(_) => v2.ask,
                TradeSignal::None => unreachable!(),
            });
        }
    }

    pub fn try_finish(&self) -> Option<(f64, f64)> {
        if let Some(close_price1) = self.close_price1 && let Some(close_price2) = self.close_price2 {
            let revenue = match self.signal {
                TradeSignal::Sell1Buy2(_) => self.open_price1 + close_price2,
                TradeSignal::Buy1Sell2(_) => self.open_price2 + close_price1,
                TradeSignal::None => unreachable!(),
            };
            let cost = match self.signal {
                TradeSignal::Sell1Buy2(_) => self.open_price2 + close_price1,
                TradeSignal::Buy1Sell2(_) => self.open_price1 + close_price2,
                TradeSignal::None => unreachable!(),
            };
            Some((revenue, cost))
        } else {
            None
        }
    }
}

pub fn run_simulation(
    values: &[OrderBookValues],
    calculator: &CalculatorsPair,
    decay_time: f64,
) -> (u32, f64) {
    let (mut v1, mut v2) = (None, None);
    let mut active_deal = Option::<SimDeal>::None;
    let (mut total_revenue, mut total_cost) = (0.0, 0.0);
    let mut deals_count = 0;
    let end_time = values[values.len() - 1].time;

    for value in values {
        match value.leg {
            Leg::First => v1 = Some(value),
            Leg::Second => v2 = Some(value),
        }

        if let Some(v1) = v1 && let Some(v2) = v2 {
            if let Some(deal) = active_deal.as_mut() {
                deal.close1(v1, v2);
                deal.close2(v1, v2);

                if let Some((revenue, cost)) = deal.try_finish() {
                    let weight = (-(end_time - deal.open_time).as_seconds_f64() / decay_time).exp();
                    active_deal = None;

                    total_revenue += weight * revenue;
                    total_cost += weight * cost;
                    deals_count += 1;
                }
            } else {
                let signal = calculator.calculate(v1, v2);
                active_deal = match signal {
                    TradeSignal::Sell1Buy2(hold) => Some(SimDeal::sell1_buy2(v1, v2, signal, hold)),
                    TradeSignal::Buy1Sell2(hold) => Some(SimDeal::buy1_sell2(v1, v2, signal, hold)),
                    TradeSignal::None => None,
                };
            }
        }
    }

    (deals_count, total_revenue - total_cost)
}
