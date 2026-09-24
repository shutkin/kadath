use chrono::{DateTime, TimeDelta, Utc};
use model::OrderBook;

pub const IMBALANCE_LEVELS: [usize; 4] = [3, 5, 10, 20];

#[derive(Copy, Clone)]
pub struct OrderBookValues {
    pub time: DateTime<Utc>,
    pub ask: f64,
    pub bid: f64,
    pub mid: f64,
    pub quantity_ask: u16,
    pub quantity_bid: u16,
    pub imbalances: [f64; IMBALANCE_LEVELS.len()],
    pub std_derivative: f64,
    pub trend_deviation: f64,
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
        let best_bid_volume = bids[0].size.as_f64();
        let best_ask_volume = asks[0].size.as_f64();
        let mid = (best_bid + best_ask) / 2.0;
        let micro = (best_ask * best_bid_volume + best_bid * best_ask_volume) / (best_bid_volume + best_ask_volume);

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

        Some(OrderBookValues {
            time: order_book.timestamp,
            bid: best_bid,
            ask: best_ask,
            mid,
            quantity_bid: bids[0].size.as_i128() as u16,
            quantity_ask: asks[0].size.as_i128() as u16,
            imbalances,
            std_derivative: 0.0,
            trend_deviation: 0.0,
        })
    }

    pub fn calculate_window_values(values: &mut [OrderBookValues], deviation_diapason: i64, trend_tau: f64) {
        let mut window_mids = Vec::new();
        let mut window_derivatives = Vec::new();
        let mut window_times = Vec::<DateTime<Utc>>::new();

        let mut trend = 0.0;
        let mut prev_time = Option::<DateTime<Utc>>::None;

        for v in values {
            if let Some(cur_prev_time) = prev_time {
                let dt = (v.time - cur_prev_time).as_seconds_f64();
                let alpha = 1.0 - (-dt / trend_tau).exp();
                trend += alpha * (v.mid - trend);
                v.trend_deviation = (v.mid - trend) / trend;
                prev_time = Some(v.time);
            } else {
                trend = v.mid;
                prev_time = Some(v.time);
            }

            let window_start = v.time - TimeDelta::seconds(deviation_diapason);
            while !window_times.is_empty() && window_times[0] < window_start {
                window_times.remove(0);
                window_derivatives.remove(0);
                window_mids.remove(0);
            }

            let mut prev_index = window_mids.len() as i32 - 1;
            while prev_index >= 0 && window_times[prev_index as usize] == v.time {
                prev_index -= 1;
            }

            let derivative: f64 = if prev_index >= 0 {
                (v.mid - window_mids[prev_index as usize]) / (v.time.timestamp_millis() - window_times[prev_index as usize].timestamp_millis()) as f64
            } else { 0.0 };

            if derivative.abs() > f64::MIN_POSITIVE {
                let d = if let Some(std) = standard_deviation(&window_derivatives) {
                    if std > 0.0000001 {
                        derivative / std
                    } else {
                        if derivative > 0.0 { 100.0 } else { -100.0 }
                    }
                } else {
                    0.0
                };
                if !d.is_nan() {
                    v.std_derivative = d;
                }
            }

            window_mids.push(v.mid);
            window_derivatives.push(derivative);
            window_times.push(v.time);
        }
    }

    pub fn to_csv(&self) -> String {
        format!(
            "{:.6},{:.6},{:.4},{:.4},{:.4},{:.4}",
            self.std_derivative,
            self.trend_deviation * 1000.0,
            self.imbalances[0], self.imbalances[1], self.imbalances[2], self.imbalances[3],
        )
    }
}

pub fn standard_deviation(data: &[f64]) -> Option<f64> {
    let count = data.len();
    if count < 2 {
        return None;
    }

    let mean = data.iter().sum::<f64>() / count as f64;
    let variance = data.iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>() / count as f64;

    Some(variance.sqrt())
}
