use std::vec;
use chrono::{DateTime, TimeDelta, Utc};
use model::math_util::standard_deviation;
use model::OrderBook;

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

impl OrderBookValues {
    pub fn calculate(order_book: &OrderBook) -> Option<OrderBookValues> {
        if order_book.asks.len() < 20 || order_book.bids.len() < 20 {
            return None;
        }

        let mut asks = order_book.asks.clone();
        asks.sort_by(|a, b| {a.price.partial_cmp(&b.price).unwrap()});

        let mut bids = order_book.bids.clone();
        bids.sort_by(|a, b| {b.price.partial_cmp(&a.price).unwrap()});

        let best_bid = bids[0].price.as_f64();
        let best_ask = asks[0].price.as_f64();
        let mid = (best_bid + best_ask) / 2.0;

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

        Some(OrderBookValues {
            time: order_book.timestamp,
            bid: best_bid,
            ask: best_ask,
            mid,
            imbalances,
            imbalances_derivative: [0.0; IMBALANCE_LEVELS.len()],
            trend: mid,
        })
    }

    pub fn calculate_window_values(values: &mut [OrderBookValues], normalize_seconds: f64, trend_tau_seconds: f64) {
        // Calculate absolute derivatives
        let mut derivatives = vec![Vec::with_capacity(values.len()); IMBALANCE_LEVELS.len()];
        for i in 0..values.len() {
            let mut id = [0.0; IMBALANCE_LEVELS.len()];
            if i > 0 {
                let delta_time = values[i].time.timestamp_millis() - values[i - 1].time.timestamp_millis();
                if delta_time > 0 {
                    for imbalance_index in 0..IMBALANCE_LEVELS.len() {
                        let d = values[i].imbalances[imbalance_index] - values[i - 1].imbalances[imbalance_index];
                        id[imbalance_index] = d / delta_time as f64;
                    }
                }
            }
            for imbalance_index in 0..IMBALANCE_LEVELS.len() {
                derivatives[imbalance_index].push(id[imbalance_index]);
            }
        }

        // Normalize derivatives
        for i in 1..values.len() {
            if normalize_seconds > 0.1 {
                let start_time = values[i].time - TimeDelta::milliseconds((normalize_seconds * 1000.0) as i64);
                let mut window_start = i - 1;
                while window_start > 0 && values[window_start].time > start_time {
                    window_start -= 1;
                }
                for imbalance_index in 0..IMBALANCE_LEVELS.len() {
                    if let Some(std) = standard_deviation(&derivatives[imbalance_index][window_start .. i]) && std > f64::MIN_POSITIVE {
                        values[i].imbalances_derivative[imbalance_index] = derivatives[imbalance_index][i] / std;
                    } else {
                        values[i].imbalances_derivative[imbalance_index] = derivatives[imbalance_index][i];
                    }
                }
            } else {
                for imbalance_index in 0..IMBALANCE_LEVELS.len() {
                    values[i].imbalances_derivative[imbalance_index] = derivatives[imbalance_index][i];
                }
            }
        }

        // Calculate trend
        let mut trend = 0.0;
        let mut prev_time = Option::<DateTime<Utc>>::None;

        for value in values {
            if let Some(cur_prev_time) = prev_time {
                let dt = (value.time - cur_prev_time).as_seconds_f64();
                let alpha = 1.0 - (-dt / trend_tau_seconds).exp();
                trend += alpha * (value.mid - trend);
                value.trend = trend;
                prev_time = Some(value.time);
            } else {
                trend = value.mid;
                prev_time = Some(value.time);
            }
        }
    }
}
