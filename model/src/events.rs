use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use crate::{OrderBook, OrderData, TradeDirection};

#[derive(Debug, Clone)]
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

#[derive(Clone, Debug)]
pub struct OrderEvent {
    pub deal_id: i32,
    pub instrument_id: i32,
    pub ticker: String,
    pub direction: TradeDirection,
    pub quantity: i32,
    pub expected_price: Decimal,
}

#[derive(Clone, Debug)]
pub struct OrderResponseEvent {
    pub deal_id: i32,
    pub direction: TradeDirection,
    pub order_id: String,
}

pub fn merge_events(ids: &[i32], data: &[&[OrderBook]]) -> Vec<OrderBookEvent> {
    let total_size = data.iter().map(|d| d.len()).sum();
    let mut events = Vec::with_capacity(total_size);
    let mut indices = vec![0; data.len()];
    loop {
        let mut cur_event = None;
        let mut cur_event_time = None;
        let mut cur_event_stream_index = 0;

        for (i, order_books) in data.iter().enumerate() {
            if let Some(ob) = order_books.get(indices[i]) {
                let use_it = if let Some(event_time) = cur_event_time {
                    ob.timestamp < event_time
                } else { true };
                if use_it {
                    cur_event_time = Some(ob.timestamp);
                    cur_event = Some(OrderBookEvent { instrument_id: ids[i], order_book: ob.clone() });
                    cur_event_stream_index = i;
                }
            }
        }

        if let Some(event) = cur_event {
            indices[cur_event_stream_index] += 1;
            events.push(event);
        } else {
            break;
        }
    }
    events
}
