use chrono::{DateTime, Utc};
use model::OrderBook;
use model::math_util::TimedFeatures;

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
    pub fn from_order_book(book: &OrderBook) -> Self {
        let mut asks = book.asks.clone();
        asks.sort_by(|a, b| {a.price.partial_cmp(&b.price).unwrap()});

        let mut bids = book.bids.clone();
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
        
        Self {
            time: book.timestamp,
            ask: best_ask,
            bid: best_bid,
            mid,
            imbalances,
        }
    }

    pub fn to_features(&self) -> TimedFeatures {
        TimedFeatures {
            time: self.time,
            features: vec![self.mid, self.imbalances[0], self.imbalances[1], self.imbalances[2], self.imbalances[3], self.imbalances[4]],
        }
    }
    
    pub fn calculate_ask_bid_mid(book: &OrderBook) -> (f64, f64, f64) {
        let ask = book.asks.iter().map(|data| data.price).min().unwrap_or_default().as_f64();
        let bid = book.bids.iter().map(|data| data.price).max().unwrap_or_default().as_f64();
        let mid = (ask + bid) * 0.5;
        (ask, bid, mid)
    }
}
