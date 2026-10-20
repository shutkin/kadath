use chrono::Timelike;
use crate::orderbook_values::OrderBookValues;
use crate::Regime;

pub fn get_regime(values: &[OrderBookValues]) -> Option<Regime> {
    if values.len() < 3 || (values[values.len() - 1].time - values[0].time).num_minutes() < 30 {
        return None;
    }

    let first_price = values[0].mid;
    let last_price = values[values.len() - 1].mid;
    let change = (last_price / first_price).ln();

    let mut prev_minute_price = first_price;
    let mut minutes_sum = 0.0;
    for i in 1..values.len() {
        if values[i - 1].time.minute() < values[i].time.minute() {
            let minute_delta = values[i].mid - prev_minute_price;
            minutes_sum += minute_delta.abs();
            prev_minute_price = values[i].mid;
        }
    }
    minutes_sum += (last_price - prev_minute_price).abs();
    let er = (last_price - first_price).abs() / minutes_sum;

    Some(Regime {
        change,
        er,
    })
}