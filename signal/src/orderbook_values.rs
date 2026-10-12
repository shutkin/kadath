use chrono::{DateTime, TimeDelta, Utc};
use model::math_util::standard_deviation;
use model::OrderBook;
use crate::MAX_WINDOW_LENGTH_S;

pub const IMBALANCE_LEVELS: [usize; 5] = [3, 5, 10, 20, 50];

#[derive(Copy, Clone)]
pub struct OrderBookValues {
    pub time: DateTime<Utc>,
    pub ask: f64,
    pub bid: f64,
    pub mid: f64,
    pub imbalances: [f64; IMBALANCE_LEVELS.len()],
    pub imbalances_derivative: [f64; IMBALANCE_LEVELS.len()],
    pub trend: f64,
}

pub struct WindowValuesCalculator {
    window_imbalances: Vec<Vec<f64>>,
    window_derivatives: Vec<Vec<f64>>,
    window_times: Vec<DateTime<Utc>>,
    trend: f64,
}

impl WindowValuesCalculator {
    pub fn new() -> Self {
        Self {
            window_imbalances: vec![Vec::new(); IMBALANCE_LEVELS.len()],
            window_derivatives: vec![Vec::new(); IMBALANCE_LEVELS.len()],
            window_times: Vec::new(),
            trend: 0.0,
        }
    }

    pub fn calculate(&mut self, order_book: &OrderBook, derivative_diapason_s: f64, trend_tau_s: f64) -> OrderBookValues {
        let mut asks = order_book.asks.clone();
        asks.sort_by(|a, b| {a.price.partial_cmp(&b.price).unwrap()});

        let mut bids = order_book.bids.clone();
        bids.sort_by(|a, b| {b.price.partial_cmp(&a.price).unwrap()});

        let best_bid = bids[0].price.as_f64();
        let best_ask = asks[0].price.as_f64();
        let mid = (best_ask + best_bid) * 0.5;

        let mut imbalances = [0.0; IMBALANCE_LEVELS.len()];
        for i in 0..IMBALANCE_LEVELS.len() {
            let levels = IMBALANCE_LEVELS[i];
            let (mut bids_sum, mut asks_sum) = (0.0, 0.0);
            for level in 0..levels {
                if let Some(bid) = bids.get(level) {
                    bids_sum += bid.size as f64;
                }
                if let Some(ask) = asks.get(level) {
                    asks_sum += ask.size as f64;
                }
            }
            imbalances[i] = (bids_sum - asks_sum) / (bids_sum + asks_sum);
        }

        let trend = self.calculate_trend(order_book.timestamp, mid, trend_tau_s);
        let normalized_derivatives = self.calculate_derivatives_and_move(
            order_book.timestamp,
            imbalances,
            derivative_diapason_s,
        );
        OrderBookValues {
            time: order_book.timestamp,
            ask: best_ask,
            bid: best_bid,
            mid,
            imbalances,
            imbalances_derivative: normalized_derivatives,
            trend,
        }
    }

    fn calculate_derivatives_and_move(&mut self, time: DateTime<Utc>, imbalances: [f64; IMBALANCE_LEVELS.len()], diapason_s: f64) -> [f64; IMBALANCE_LEVELS.len()] {
        let window_start = time - TimeDelta::seconds(MAX_WINDOW_LENGTH_S);
        while !self.window_times.is_empty() && self.window_times[0] < window_start {
            self.window_times.remove(0);
            for imbalance_index in 0..IMBALANCE_LEVELS.len() {
                self.window_imbalances[imbalance_index].remove(0);
                self.window_derivatives[imbalance_index].remove(0);
            }
        }

        let mut derivatives = [0.0; IMBALANCE_LEVELS.len()];
        let prev_index = self.window_times.len() as i32 - 1;
        if prev_index >= 0 {
            let dt = (time.timestamp_millis() - self.window_times[prev_index as usize].timestamp_millis()) as f64;
            if dt > f64::MIN_POSITIVE {
                for imbalance_index in 0..IMBALANCE_LEVELS.len() {
                    derivatives[imbalance_index] = (imbalances[imbalance_index] - self.window_imbalances[imbalance_index][prev_index as usize]) / dt;
                }
            }
        }

        let data_start = time - TimeDelta::milliseconds((diapason_s * 1000.0) as i64);
        let mut start_index = 0;
        while start_index < self.window_times.len() && self.window_times[start_index] < data_start {
            start_index += 1;
        }

        let mut derivatives_normalized = [0.0; IMBALANCE_LEVELS.len()];
        for imbalance_index in 0..IMBALANCE_LEVELS.len() {
            let deviation = standard_deviation(&self.window_derivatives[imbalance_index][start_index..]).unwrap_or(0.0);
            if deviation > f64::MIN_POSITIVE {
                derivatives_normalized[imbalance_index] = derivatives[imbalance_index] / deviation;
            }
        }

        for imbalance_index in 0..IMBALANCE_LEVELS.len() {
            self.window_imbalances[imbalance_index].push(imbalances[imbalance_index]);
            self.window_derivatives[imbalance_index].push(derivatives[imbalance_index]);
        }
        self.window_times.push(time);

        derivatives_normalized
    }

    fn calculate_trend(&mut self, time: DateTime<Utc>, mid: f64, tau_s: f64) -> f64 {
        let mut prev_time_index = self.window_times.len() as i32 - 1;
        while prev_time_index >= 0 && self.window_times[prev_time_index as usize] >= time {
            prev_time_index -= 1;
        }

        if prev_time_index >= 0 && let Some(prev_time) = self.window_times.get(prev_time_index as usize) {
            let dt = (time - prev_time).as_seconds_f64();
            let alpha = 1.0 - (-dt / tau_s).exp();
            self.trend += alpha * (mid - self.trend);
        } else {
            self.trend = mid;
        }

        self.trend
    }
}
