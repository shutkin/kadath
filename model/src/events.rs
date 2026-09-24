use chrono::{DateTime, Utc};
use crate::{OrderBook, OrderData};

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

pub fn merge_events(id1: i32, id2: i32, values1: &[OrderBook], values2: &[OrderBook]) -> Vec<OrderBookEvent> {
    let mut events = Vec::with_capacity(values1.len() + values2.len());
    let (mut v_index1, mut v_index2) = (0, 0);
    while v_index1 < values1.len() || v_index2 < values2.len() {
        let mut cur_event = None;
        let mut cur_event_time = None;

        if let Some(v) = values1.get(v_index1) {
            let use_it = if let Some(event_time) = cur_event_time {
                v.timestamp < event_time
            } else { true };
            if use_it {
                cur_event_time = Some(v.timestamp);
                cur_event = Some(OrderBookEvent { instrument_id: id1, order_book: v.clone() });
            }
        }

        if let Some(v) = values2.get(v_index2) {
            let use_it = if let Some(event_time) = cur_event_time {
                v.timestamp < event_time
            } else { true };
            if use_it {
                cur_event_time = Some(v.timestamp);
                cur_event = Some(OrderBookEvent { instrument_id: id2, order_book: v.clone() });
            }
        }

        if let Some(event) = cur_event {
            if event.instrument_id == id1 {
                v_index1 += 1;
            } else {
                v_index2 += 1;
            }
            events.push(event);
        } else {
            break;
        }
    }
    events
}
