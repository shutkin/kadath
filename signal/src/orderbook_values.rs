use chrono::{DateTime, TimeDelta, Utc};
use model::math_util::standard_deviation;
use model::OrderBook;
use crate::MAX_WINDOW_LENGTH_S;

pub const IMBALANCE_LEVELS: [usize; 5] = [3, 5, 10, 20, 50];

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Leg {
    First, Second,
}

#[derive(Copy, Clone)]
pub struct OrderBookValues {
    pub leg: Leg,
    pub time: DateTime<Utc>,
    pub ask: f64,
    pub bid: f64,
    pub imbalances: [f64; IMBALANCE_LEVELS.len()],
    pub normal_derivative: f64,
}

impl OrderBookValues {
    pub fn mid(&self) -> f64 {
        (self.ask + self.bid) * 0.5
    }
}

pub struct WindowValuesCalculator {
    leg: Leg,
    window_mids: Vec<f64>,
    window_derivatives: Vec<f64>,
    window_times: Vec<DateTime<Utc>>,
}

impl WindowValuesCalculator {
    pub fn new(leg: Leg) -> Self {
        Self {
            leg,
            window_mids: Vec::new(),
            window_derivatives: Vec::new(),
            window_times: Vec::new(),
        }
    }

    pub fn calculate(&mut self, order_book: &OrderBook, std_diapason_s: u16) -> OrderBookValues {
        let mut asks = order_book.asks.clone();
        asks.sort_by(|a, b| {a.price.partial_cmp(&b.price).unwrap()});

        let mut bids = order_book.bids.clone();
        bids.sort_by(|a, b| {b.price.partial_cmp(&a.price).unwrap()});

        let best_bid = bids[0].price.as_f64();
        let best_ask = asks[0].price.as_f64();

        let mut imbalances = [0.0; IMBALANCE_LEVELS.len()];
        for i in 0..IMBALANCE_LEVELS.len() {
            let levels = IMBALANCE_LEVELS[i];
            let (mut bids_sum, mut asks_sum) = (0.0, 0.0);
            for level in 0..levels {
                if let Some(bid) = bids.get(level) {
                    bids_sum += bid.size.as_f64();
                }
                if let Some(ask) = asks.get(level) {
                    asks_sum += ask.size.as_f64();
                }
            }
            imbalances[i] = (bids_sum - asks_sum) / (bids_sum + asks_sum);
        }

        let normal_derivative = self.calculate_derivative(
            order_book.timestamp,
            (best_bid + best_ask) * 0.5,
            std_diapason_s,
        );

        OrderBookValues {
            leg: self.leg,
            time: order_book.timestamp,
            ask: best_ask,
            bid: best_bid,
            imbalances,
            normal_derivative,
        }
    }

    fn calculate_derivative(&mut self, time: DateTime<Utc>, mid: f64, diapason_s: u16) -> f64 {
        let window_start = time - TimeDelta::seconds(MAX_WINDOW_LENGTH_S);
        while !self.window_times.is_empty() && self.window_times[0] < window_start {
            self.window_times.remove(0);
            self.window_derivatives.remove(0);
            self.window_mids.remove(0);
        }

        let data_start = time - TimeDelta::seconds(diapason_s as i64);
        let mut start_index = 0;
        while start_index < self.window_times.len() && self.window_times[start_index] < data_start {
            start_index += 1;
        }

        let mut prev_index = self.window_mids.len() as i32 - 1;
        while prev_index >= 0 && self.window_times[prev_index as usize] == time {
            prev_index -= 1;
        }

        let derivative: f64 = if prev_index >= 0 {
            let dt = (time.timestamp_millis() - self.window_times[prev_index as usize].timestamp_millis()) as f64;
            (mid - self.window_mids[prev_index as usize]) / dt
        } else { 0.0 };

        let mut result = 0.0;
        if derivative.abs() > f64::MIN_POSITIVE && let Some(std) = standard_deviation(&self.window_derivatives[start_index..]) {
            if std > 0.0000001 {
                result = derivative / std
            } else {
                result = if derivative > 0.0 { 100.0 } else { -100.0 }
            }
        }

        self.window_mids.push(mid);
        self.window_derivatives.push(derivative);
        self.window_times.push(time);

        result
    }
}
