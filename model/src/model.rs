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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TradeDirection {
    Buy, Sell,
}

impl TradeDirection {
    pub fn reverse(&self) -> TradeDirection {
        match self {
            TradeDirection::Buy => TradeDirection::Sell,
            TradeDirection::Sell => TradeDirection::Buy,
        }
    }
    
    pub fn to_short_string(&self) -> String {
        match self {
            TradeDirection::Buy => "b".to_string(),
            TradeDirection::Sell => "s".to_string(),
        }
    }

    pub fn to_full_string(&self) -> String {
        match self {
            TradeDirection::Buy => "buy".to_string(),
            TradeDirection::Sell => "sell".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Deal {
    pub id: i32,
    pub instrument_id: i32,
    pub started: DateTime<Utc>,
    pub direction: TradeDirection,
    pub open_order_id: Option<String>,
    pub close_order_id: Option<String>,
}

#[derive(Clone)]
pub struct Trade {
    pub id: Option<i32>,
    pub order_id: String,
    pub executed: DateTime<Utc>,
    pub direction: TradeDirection,
    pub price: Decimal,
    pub quantity: i16,
    pub json: String,
}