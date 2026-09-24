pub mod utils;
pub mod events;
pub mod common;
pub mod math_util;
pub mod order_book_cache;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct Instrument {
    pub id: Option<i32>,
    pub ticker: String,
    pub name: String,
    pub lot_size: i16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderBook {
    pub id: Option<i64>,
    pub timestamp: DateTime<Utc>,
    pub asks: Vec<OrderData>,
    pub bids: Vec<OrderData>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct OrderData {
    pub price: Decimal,
    pub size: i32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OrderDataDelta {
    pub price_id: i32,
    pub size: i32,
}

#[derive(Clone, Debug)]
pub struct OrderDataSlice {
    pub timestamp: DateTime<Utc>,
    pub order_data: Vec<OrderData>,
}
