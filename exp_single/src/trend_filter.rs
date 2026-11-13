use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use model::math_util::mean_f32;
use model::TradeDirection;

pub struct TrendFilter {
    direction: TradeDirection,
    length: TimeDelta,
    buffer_prices: Vec<f32>,
    buffer_times: Vec<DateTime<Utc>>,
}

impl TrendFilter {
    pub fn new(direction: TradeDirection, length_seconds: u16) -> Self {
        Self {
            direction,
            length: TimeDelta::seconds(length_seconds as i64),
            buffer_prices: Vec::new(),
            buffer_times: Vec::new(),
        }
    }
    
    pub fn process(&mut self, book: &OrderBookValues) -> bool {
        let cur_price = book.mid as f32;
        let mean = if self.buffer_prices.len() > 5 {
            mean_f32(&self.buffer_prices)
        } else {
            0.0
        };
        
        let result = if mean > 0.0 {
            match self.direction {
                TradeDirection::Buy => cur_price > mean,
                TradeDirection::Sell => cur_price < mean,
            }
        } else {
            true
        };
        
        self.buffer_prices.push(cur_price);
        self.buffer_times.push(book.time);

        let threshold = book.time - self.length;
        while let Some(first_time) = self.buffer_times.first() && *first_time < threshold {
            self.buffer_prices.remove(0);
            self.buffer_times.remove(0);
        }
        
        result
    }
    
    pub fn test(&mut self, test_values: &[OrderBookValues]) -> Vec<bool> {
        let mut result = Vec::with_capacity(test_values.len());
        
        for v in test_values {
            result.push(self.process(v));
        }
        
        result
    }
}