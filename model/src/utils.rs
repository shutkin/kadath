use std::fs::File;
use std::io::Write;
use chrono::{DateTime, Timelike, Utc};
use rust_decimal::Decimal;
use crate::{OrderBook, OrderData};

pub fn calculate_delta_sell(order_book1: &OrderBook, order_book2: &OrderBook, min_volume: Decimal) -> Option<OrderData> {
    let max_bid = order_book1.bids.iter()
        .filter(|bid| bid.price * bid.size > min_volume)
        .max_by_key(|bid| bid.price);
    let min_ask = order_book2.asks.iter()
        .filter(|ask| ask.price * ask.size > min_volume)
        .min_by_key(|ask| ask.price);
    map_delta(max_bid, min_ask)
}

pub fn calculate_delta_buy(order_book1: &OrderBook, order_book2: &OrderBook, min_volume: Decimal) -> Option<OrderData> {
    let min_ask = order_book1.asks.iter()
        .filter(|ask| ask.price * ask.size > min_volume)
        .min_by_key(|ask| ask.price);
    let max_bid = order_book2.bids.iter()
        .filter(|bid| bid.price * bid.size > min_volume)
        .max_by_key(|bid| bid.price);
    map_delta(min_ask, max_bid)
}

fn map_delta(data1: Option<&OrderData>, data2: Option<&OrderData>) -> Option<OrderData> {
    if let (Some(data1), Some(data2)) = (data1, data2) {
        Some(OrderData {
            price: data2.price - data1.price,
            size: data1.size.min(data2.size),
        })
    } else {
        None
    }
}

pub fn handle_delta(deltas: &[OrderData], order_book_data: &mut Vec<OrderData>) {
    for delta_data in deltas {
        if delta_data.size.is_zero() {
            order_book_data.retain(|data| data.price != delta_data.price);
        } else {
            let mut found = false;
            for data in order_book_data.as_mut_slice() {
                if data.price == delta_data.price {
                    data.size = delta_data.size;
                    found = true;
                    break;
                }
            }
            if !found {
                order_book_data.push(delta_data.clone());
            }
        }
    }
}
