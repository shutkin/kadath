use chrono::{DateTime, Utc};
use crate::{OrderBook, OrderData};

#[derive(Debug)]
pub struct OrderBookEvent {
    pub instrument_id: i32,
    pub order_book: OrderBook,
}

#[derive(Debug)]
pub struct OrderBookDeltaEvent {
    pub instrument_id: i32,
    pub timestamp: DateTime<Utc>,
    pub asks: Vec<OrderData>,
    pub bids: Vec<OrderData>,
}