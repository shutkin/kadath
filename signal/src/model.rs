extern crate core;

mod orderbook_values;
pub mod signal_calculator;
mod optimizer;
pub mod strategy_optimizer;

use crate::optimizer::calibrate_signal_calculator;
use crate::orderbook_values::{OrderBookValues, WindowValuesCalculator};
use crate::signal_calculator::{SignalCalculator, SignalPerformance};
use chrono::TimeDelta;
use model::events::OrderBookEvent;
use std::sync::{Arc, Mutex, RwLock};

const MAX_WINDOW_LENGTH_S: i64 = 2000;
const TRAIN_DATA_MINUTES: u16 = 45;
const TRAIN_INTERVAL_MINUTES: u16 = 1;
const DEFAULT_DECAY_TIME: f64 = 15.0;
const STD_DIAPASON_SECONDS: u16 = 1200;

pub const HOLD_TIME_VARIANTS: [u16; 6] = [200, 300, 500, 750, 1000, 1500];

#[derive(Copy, Clone)]
pub enum TradeSignal {
    None,
    Sell(u16),
    Buy(u16),
}

#[derive(Copy, Clone, Debug)]
pub struct ModelConfig {
    pub derivatives_normalize_diapason_s: f64,
    pub trend_tau_s: f64,
    pub train_data_minutes: u16,
    pub train_interval_minutes: u16,
    pub decay_time: f64,
    pub hold_time_ms: u16,
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
                    return TradeSignal::Buy(calculator.get_hold_time_ms())
                } else if !self.is_buying && signal_score < 0.0 {
                    return TradeSignal::Sell(calculator.get_hold_time_ms())
                }
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

        let performances = if let Ok(read) = self.calculator.read() {
            read.map(|calc| calc.get_performances())
        } else {
            None
        };
        let mut response = CalibrationResponse {
            prev_cycle_performances: performances,
            ..Default::default()
        };

        let buf_clone = self.buffer.lock().map(|b| b.clone()).unwrap_or_default();
        let calibration_result = calibrate_signal_calculator(&buf_clone, self.is_buying, self.config.decay_time, self.config.hold_time_ms);
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