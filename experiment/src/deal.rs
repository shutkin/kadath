use chrono::{DateTime, Utc};
use crate::OrderBookValues;

#[derive(Copy, Clone, Debug)]
pub enum DealDirection {
    Sell1Buy2, Buy1Sell2,
}

#[derive(Copy, Clone)]
pub struct Deal {
    direction: DealDirection,
    open_time1: DateTime<Utc>,
    open_time2: DateTime<Utc>,
    open_time: DateTime<Utc>,
    quantity: u16,
    entry_price1: f64,
    entry_price2: f64,
    close_price1: Option<f64>,
    close_price2: Option<f64>,
    close_time1: Option<DateTime<Utc>>,
    close_time2: Option<DateTime<Utc>>,
}

impl Deal {
    pub fn sell1_buy2(v1: &OrderBookValues, v2: &OrderBookValues) -> Self {
        Self {
            open_time1: v1.time,
            open_time2: v2.time,
            open_time: v1.time.max(v2.time),
            direction: DealDirection::Sell1Buy2,

            // Sell GLU6 -> bid
            entry_price1: v1.bid,

            // Buy GLZ6 -> ask
            entry_price2: v2.ask,

            quantity: v1.quantity_bid.min(v2.quantity_ask),

            close_price1: None,
            close_price2: None,
            close_time1: None,
            close_time2: None,
        }
    }

    pub fn buy1_sell2(v1: &OrderBookValues, v2: &OrderBookValues) -> Self {
        Self {
            open_time1: v1.time,
            open_time2: v2.time,
            open_time: v1.time.max(v2.time),
            direction: DealDirection::Buy1Sell2,

            // Buy GLU6 -> ask
            entry_price1: v1.ask,

            // Sell GLZ6 -> bid
            entry_price2: v2.bid,

            quantity: v1.quantity_ask.min(v2.quantity_bid),

            close_price1: None,
            close_price2: None,
            close_time1: None,
            close_time2: None,
        }
    }

    pub fn get_direction(&self) -> DealDirection {
        self.direction
    }

    pub fn get_open_time(&self) -> DateTime<Utc> {
        self.open_time
    }

    pub fn get_close_time(&self) -> DateTime<Utc> {
        self.close_time1.unwrap().max(self.close_time2.unwrap())
    }

    pub fn close_instrument1(&mut self, price: f64, time: DateTime<Utc>) {
        if self.close_price1.is_none() {
            self.close_price1 = Some(price);
            self.close_time1 = Some(time);
        }
    }

    pub fn close_instrument2(&mut self, price: f64, time: DateTime<Utc>) {
        if self.close_price2.is_none() {
            self.close_price2 = Some(price);
            self.close_time2 = Some(time);
        }
    }

    pub fn is_completed(&self) -> bool {
        self.close_price1.is_some() && self.close_price2.is_some()
    }

    pub fn legs_profits(&self) -> (f64, f64) {
        match self.direction {
            DealDirection::Sell1Buy2 => {
                let leg1 = self.entry_price1 - self.close_price1.unwrap();
                let leg2 = self.close_price2.unwrap() - self.entry_price2;
                (leg1, leg2)
            },
            DealDirection::Buy1Sell2 => {
                let leg1 = self.close_price1.unwrap() - self.entry_price1;
                let leg2 = self.entry_price2 - self.close_price2.unwrap();
                (leg1, leg2)
            }
        }
    }

    pub fn close(&self, log: bool) -> (u16, f64, f64) {
        if let Some(p1) = self.close_price1 && let Some (p2) = self.close_price2 &&
            let Some(close_time1) = self.close_time1 && let Some(close_time2) = self.close_time2 {
            match self.direction {
                DealDirection::Sell1Buy2 => {
                    let revenue = self.entry_price1 + p2; // bid
                    let cost = self.entry_price2 + p1; // ask

                    if log {
                        println!(
                            "sell 1 @ {} at {} buy 2 @ {} at {} -> buy 1 @ {} at {} sell 2 {} at {}, revenue {}, cost {}",
                            self.entry_price1, self.open_time1, self.entry_price2, self.open_time2,
                            p1, close_time1, p2, close_time2,
                            revenue, cost,
                        );
                        println!("Profit {}", revenue - cost);
                    }

                    (self.quantity, revenue, cost)
                }

                DealDirection::Buy1Sell2 => {
                    let revenue = self.entry_price2 + p1; // bid
                    let cost = self.entry_price1 + p2; // ask

                    if log {
                        println!(
                            "Buy 1 @ {} at {} sell 2 @ {} at {} -> sell 1 @ {} at {} buy 2 @ {} at {}, revenue {}, cost {}",
                            self.entry_price1, self.open_time1, self.entry_price2, self.open_time2,
                            p1, close_time1, p2, close_time2,
                            revenue, cost,
                        );
                        println!("Profit {}", revenue - cost);
                    }

                    (self.quantity, revenue, cost)
                }
            }
        } else {
            panic!("Close unfinished deal");
        }
    }
}
