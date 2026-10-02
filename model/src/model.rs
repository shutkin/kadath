pub mod utils;
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DealStatus {
    Opening,
    OpenConfirmed,
    Closing,
    CloseConfirmed,
    OpenFailed,
    CloseFailed,
}

impl DealStatus {
    pub fn get_code(&self) -> &str {
        match self {
            DealStatus::Opening => "0",
            DealStatus::OpenConfirmed => "1",
            DealStatus::Closing => "2",
            DealStatus::CloseConfirmed => "3",
            DealStatus::OpenFailed => "8",
            DealStatus::CloseFailed => "9",
        }
    }
    
    pub fn from_code (code: &str) -> DealStatus {
        match code { 
            "0" => DealStatus::Opening,
            "1" => DealStatus::OpenConfirmed,
            "2" => DealStatus::Closing,
            "3" => DealStatus::CloseConfirmed,
            "4" => DealStatus::OpenFailed,
            _ => DealStatus::CloseFailed,
        }
    }
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
pub struct SpreadDeal {
    pub id: i32,
    pub instrument_id: i32,
    pub started: DateTime<Utc>,
    pub direction: TradeDirection,
    pub hold: u16,
    pub open_price_expected: Option<Decimal>,
    pub close_price_expected: Option<Decimal>,
    pub open_order_id: Option<String>,
    pub close_order_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Deal {
    pub id: i32,
    pub started: DateTime<Utc>,
    pub leg1_instrument_id: i32,
    pub leg2_instrument_id: i32,
    pub direction: TradeDirection,
    pub hold: u16,
    pub leg1_status: DealStatus,
    pub leg2_status: DealStatus,
    pub leg1_open_price_expected: Decimal,
    pub leg2_open_price_expected: Decimal,
    pub leg1_close_price_expected: Option<Decimal>,
    pub leg2_close_price_expected: Option<Decimal>,
    pub leg1_open_order_id: Option<String>,
    pub leg2_open_order_id: Option<String>,
    pub leg1_close_order_id: Option<String>,
    pub leg2_close_order_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Trade {
    pub id: Option<i32>,
    pub order_id: String,
    pub executed: DateTime<Utc>,
    pub direction: TradeDirection,
    pub price: Decimal,
    pub quantity: i16,
    pub json: String,
}

#[derive(Clone)]
pub struct SignalCalculatorLog {
    pub trained_on : DateTime<Utc>,
    pub up_score: Option<f32>,
    pub down_score: Option<f32>,
    pub up_params: Option<String>,
    pub down_params: Option<String>,
}