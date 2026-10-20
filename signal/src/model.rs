extern crate core;

pub mod orderbook_values;
pub mod signal_calculator;
mod optimizer;
pub mod strategy_optimizer;
pub mod kernel_prediction;
mod regime;

use crate::optimizer::calibrate_signal_calculator;
use crate::orderbook_values::{OrderBookValues, WindowValuesCalculator};
use crate::signal_calculator::{SignalCalculator, SignalPerformance};
use chrono::{DateTime, TimeDelta, Utc};
use model::events::OrderBookEvent;
use std::sync::{Arc, Mutex, RwLock};
use model::{OrderBook, TradeDirection};
use crate::kernel_prediction::{predict_kernel, KernelPrediction, KernelConfig};
use crate::regime::get_regime;

const MAX_WINDOW_LENGTH_S: i64 = 2000;

pub const HOLD_TIME_VARIANTS: [u16; 6] = [200, 300, 500, 750, 1000, 1500];

#[derive(Copy, Clone)]
pub enum TradeSignal {
    None,
    Sell(i32),
    Buy(i32),
}

impl Default for TradeSignal {
    fn default() -> Self {
        TradeSignal::None
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Regime {
    pub change: f64,
    pub er: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct ModelConfig {
    pub derivatives_normalize_diapason_s: f64,
    pub trend_tau_s: f64,
    pub train_data_minutes: u16,
    pub train_interval_minutes: u16,
    pub decay_time: f64,
    pub hold_time_ms: i32,
    
    pub price_change_filter: f64,
    pub prediction_strength: f64,
    pub predicted_change: f64,
    pub prediction_probability_up_to_flat: f64,
    pub prediction_probability_up_to_down: f64,
}

pub struct WindowData {
    values_calculator: WindowValuesCalculator,
}

impl WindowData {
    pub fn new() -> Self {
        Self {
            values_calculator: WindowValuesCalculator::new(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DealData {
    direction: TradeDirection,
    open_time: DateTime<Utc>,
    open_price: f64,
    trend: f64,
    max_trend: f64,
    prev_price_time: DateTime<Utc>,
}

impl DealData {
    pub fn new(book: &OrderBook, direction: TradeDirection) -> Self {
        let (_ask, _bid, mid) = OrderBookValues::calculate_ask_bid_mid(book);
        //let (open_price, close_price) = match direction {
        //    TradeDirection::Buy => (ask, bid),
        //    TradeDirection::Sell => (bid, ask),
        //};
        Self {
            direction,
            open_time: book.timestamp,
            open_price: mid,
            trend: mid,
            max_trend: mid,
            prev_price_time: book.timestamp,
        }
    }
}

#[derive(Clone)]
pub struct WalkForwardModel {
    instrument_id: i32,
    config: ModelConfig,
    is_buying: bool,
    calculator: Arc<RwLock<Option<SignalCalculator>>>,
    buffer: Arc<Mutex<Vec<OrderBookValues>>>,
}

#[derive(Copy, Clone, Default)]
pub struct CalibrationResponse {
    pub prev_cycle_performances: Option<SignalPerformance>,
    pub calculator: Option<SignalCalculator>,
}

impl WalkForwardModel {
    pub fn new_buying(instrument_id: i32, config: ModelConfig) -> Self {
        Self {
            instrument_id,
            config,
            is_buying: true,
            calculator: Arc::new(RwLock::new(None)),
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn new_selling(instrument_id: i32, config: ModelConfig) -> Self {
        Self {
            instrument_id,
            config,
            is_buying: false,
            calculator: Arc::new(RwLock::new(None)),
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_last_value(&self) -> Option<OrderBookValues> {
        let mut v = None;
        if let Ok(lock) = self.buffer.lock() {
            v = lock.last().copied();
        }
        v
    }

    pub fn get_price_change_from_past(&self, time_ms: i64) -> Option<f64> {
        if let Ok(buf) = self.buffer.lock() && buf.len() > 3 {
            let mut i = buf.len() - 1;
            let last_value = &buf[i];
            let target_time = last_value.time - TimeDelta::milliseconds(time_ms);
            while i > 0 && buf[i].time > target_time {
                i -= 1;
            }
            if i > 0 {
                Some(last_value.mid - buf[i].mid)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn predict(&self, time_ms: i32) -> Option<KernelPrediction> {
        let buf: Vec<OrderBookValues> = self.buffer.lock()
            .map(|vec| vec.to_vec())
            .unwrap_or_default();
        if let Some(v) = self.get_last_value() {
            let kernel_config = KernelConfig {
                bandwidth: 0.2,
                epsilon: 1.5,
                min_effective_samples: 1024.0,
            };
            predict_kernel(&buf, &v, kernel_config, time_ms)
        } else {
            None
        }
    }

    pub fn get_regime(&self) -> Option<Regime> {
        get_regime(self.buffer.lock().unwrap().as_slice())
    }

    pub fn is_it_time_to_close_deal(&self, book: &OrderBook, deal_data: &mut DealData) -> bool {
        let (_ask, _bid, close_price) = OrderBookValues::calculate_ask_bid_mid(book);
        //let close_price = match deal_data.direction {
        //    TradeDirection::Buy => bid,
        //    TradeDirection::Sell => ask,
        //};

        let dt = (book.timestamp - deal_data.prev_price_time).as_seconds_f64();
        let alpha = 1.0 - (-dt / 1.5).exp();
        deal_data.trend += alpha * (close_price - deal_data.trend);

        if deal_data.max_trend < deal_data.trend {
            deal_data.max_trend = deal_data.trend;
        }

        deal_data.prev_price_time = book.timestamp;

        match deal_data.direction {
            TradeDirection::Buy => {
                if close_price < deal_data.open_price - 0.5 && book.timestamp > deal_data.open_time + TimeDelta::milliseconds(300)
                    || deal_data.trend < deal_data.max_trend - 0.5 {
                    return true;
                }
            }
            TradeDirection::Sell => {
                if close_price > deal_data.open_price + 0.5 && book.timestamp > deal_data.open_time + TimeDelta::milliseconds(300)
                    || deal_data.trend > deal_data.max_trend + 0.5 {
                    return true;
                }
            }
        }

        false
    }

    pub fn process(&self, window_data: &mut WindowData, event: &OrderBookEvent) -> TradeSignal {
        let v = if self.instrument_id == event.instrument_id {
            window_data.values_calculator.calculate(
                &event.order_book,
                self.config.derivatives_normalize_diapason_s,
                self.config.trend_tau_s)
        } else {
            return TradeSignal::None;
        };
        if let Ok(mut lock) = self.buffer.lock() {
            lock.push(v);
        }

        if let Ok(read) = self.calculator.read()
            && let Some(calculator) = read.as_ref()
            && let Some(v) = self.get_last_value() {
            let signal_score = calculator.calculate(&v);
            if signal_score.abs() > f64::MIN_POSITIVE {
                if self.is_buying && signal_score > 0.0 {
                    if let Some(price_change) = self.get_price_change_from_past(5000)
                        && price_change > self.config.price_change_filter {
                        if let Some(prediction) = self.predict(3000) {
                            if prediction.strength > self.config.prediction_strength
                                && prediction.predicted_change > self.config.predicted_change
                                && prediction.probability_up - prediction.probability_flat > self.config.prediction_probability_up_to_flat
                                && prediction.probability_up - prediction.probability_down > self.config.prediction_probability_up_to_down {
                                return TradeSignal::Buy(calculator.get_hold_time_ms())
                            }
                        }
                    }
                } /*else if !self.is_buying && signal_score < 0.0 {
                    return TradeSignal::Sell(calculator.get_hold_time_ms())
                }*/
            }
        }
        TradeSignal::None
    }
    
    pub fn inform_deal_result(&self, profit: f64) {
        if let Ok(mut write) = self.calculator.write()
            && let Some(calc) = write.as_mut() {
            calc.deal_performance(profit);
        }
    }
    
    pub fn calibrate(&self, check_time: bool) -> CalibrationResponse {
        let buf_minutes = if let Ok(buffer) = self.buffer.lock() {
            if buffer.len() >= 2 {
                let delta = buffer[buffer.len() - 1].time - buffer[0].time;
                delta.num_minutes() as u16
            } else { 0 }
        } else { 0 };
        if buf_minutes < self.config.train_data_minutes - 1 {
            return CalibrationResponse::default();
        }

        if check_time
            && let Ok(read) = self.calculator.read()
            && let Some(calc) = read.as_ref()
            && let Ok(buffer) = self.buffer.lock() {
            let last_time = buffer[buffer.len() - 1].time;
            if last_time - calc.get_created_on() < TimeDelta::minutes(self.config.train_interval_minutes as i64) {
                return CalibrationResponse::default();
            }
        }

        self.shrink_buffer();

        let prev_calc = self.calculator.read().ok().map(|c| c.clone()).flatten();
        let mut response = CalibrationResponse {
            prev_cycle_performances: prev_calc.map(|c| c.get_performances()),
            ..Default::default()
        };

        let buf_clone = self.buffer.lock().map(|b| b.clone()).unwrap_or_default();
        let calibration_result = calibrate_signal_calculator(
            &prev_calc, &buf_clone, self.is_buying, self.config.decay_time, self.config.hold_time_ms
        );
        self.set_calculator(calibration_result);

        response.calculator = calibration_result;
        response
    }
    
    pub fn set_calculator(&self, calculator: Option<SignalCalculator>) {
        if let Ok(mut write) = self.calculator.write() {
            *write = calculator;
        }
    }

    fn shrink_buffer(&self) {
        if let Ok(mut buffer) = self.buffer.lock() {
            let threshold = buffer[buffer.len() - 1].time - TimeDelta::minutes(self.config.train_data_minutes as i64);
            let mut to_delete = 0;
            while to_delete < buffer.len() && buffer[to_delete].time < threshold {
                to_delete += 1;
            }

            if to_delete > 0 {
                buffer.drain(0..to_delete);
            }
        }
    }
}