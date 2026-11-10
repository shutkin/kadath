use chrono::{DateTime, TimeDelta, Utc};
use model::OrderBook;
use model::math_util::{standard_deviation, TimedFeatures};

pub const IMBALANCE_LEVELS: [usize; 5] = [3, 5, 10, 20, 50];

#[derive(Copy, Clone)]
pub struct OrderBookValues {
    pub time: DateTime<Utc>,
    pub ask: f64,
    pub bid: f64,
    pub mid: f64,
    pub imbalances: [f64; IMBALANCE_LEVELS.len()],
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
        })
    }

    pub fn to_features(&self) -> TimedFeatures {
        TimedFeatures {
            time: self.time,
            features: vec![self.mid, self.ask - self.bid, self.imbalances[0], self.imbalances[1], self.imbalances[2], self.imbalances[3], self.imbalances[4]],
        }
    }
}
