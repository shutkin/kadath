pub mod events;
pub mod common;
pub mod math_util;
pub mod order_book_cache;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum DataSource {
    Alor, TInvest,
}

impl DataSource {
    pub fn get_code(&self) -> &str {
        match self {
            DataSource::Alor => "A",
            DataSource::TInvest => "T",
        }
    }

    pub fn from_code (code: &str) -> DataSource {
        if code == "T" {
            DataSource::TInvest
        } else {
            DataSource::Alor
        }
    }
}

#[derive(Clone, Debug)]
pub struct Instrument {
    pub id: Option<i32>,
    pub source: DataSource,
    pub ticker: String,
    pub name: String,
    pub lot_size: i16,
    pub external_id: Option<String>,
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
pub struct OrderDataCompact {
    pub price_id: i32,
    pub size: i32,
}

#[derive(Clone, Debug)]
pub struct OrderDataSlice {
    pub timestamp: DateTime<Utc>,
    pub order_data: Vec<OrderData>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OrderStatus {
    NotStarted, Started, Executed, Failed,
}

impl OrderStatus {
    pub fn to_short_string(&self) -> String {
        match self {
            OrderStatus::NotStarted => "n".to_string(),
            OrderStatus::Started => "s".to_string(),
            OrderStatus::Executed => "e".to_string(),
            OrderStatus::Failed => "f".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Deal {
    pub id: i32,
    pub instrument_id: i32,
    pub lots: u16,
    pub direction: TradeDirection,
    pub max_hold: i32,

    pub open_time: DateTime<Utc>,
    pub expected_open_price: Decimal,
    pub open_order_id: Option<String>,
    pub open_status: OrderStatus,

    pub close_time: Option<DateTime<Utc>>,
    pub expected_close_price: Option<Decimal>,
    pub close_order_id: Option<String>,
    pub close_status: OrderStatus,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TradeStatus {
    Executed, Failed,
}

#[derive(Clone, Debug)]
pub struct Trade {
    pub id: Option<i32>,
    pub order_id: String,
    pub executed: DateTime<Utc>,
    pub direction: TradeDirection,
    pub price: Decimal,
    pub quantity: i16,
    pub status: TradeStatus,
    pub json: String,
}

#[derive(Clone)]
pub struct SignalCalculatorLog {
    pub trained_on : DateTime<Utc>,
    pub up_score: Option<f32>,
    pub down_score: Option<f32>,
    pub up_params: Option<String>,
    pub down_params: Option<String>,
    pub train_time_ms: i32,
}