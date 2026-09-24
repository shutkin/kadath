use std::fs;
use crate::common::{CommonError, EmptyResult, TimeDiapason};
use crate::OrderBook;

pub fn write(instrument: &str, diapason: TimeDiapason, order_books: &[OrderBook], is_full: bool) -> EmptyResult {
    let data = bincode::serde::encode_to_vec(order_books, bincode::config::standard())?;
    let filename = format!(".cache{}/{instrument}_{}_{}.bin",
                           if is_full {"_full"} else {""}, diapason.from.timestamp_millis(), diapason.to.timestamp_millis());
    fs::write(&filename, data)?;
    Ok(())
}

pub fn read(instrument: &str, diapason: TimeDiapason, is_full: bool) -> Result<Option<Vec<OrderBook>>, CommonError> {
    let filename = format!(".cache{}/{instrument}_{}_{}.bin",
                           if is_full {"_full"} else {""}, diapason.from.timestamp_millis(), diapason.to.timestamp_millis());
    if fs::exists(&filename)? {
        let data = fs::read(&filename)?;
        let (result, _): (Vec<OrderBook>, usize) = bincode::serde::decode_from_slice(&data, bincode::config::standard())?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}