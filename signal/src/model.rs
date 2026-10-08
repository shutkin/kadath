mod orderbook_values;
pub mod signal_calculator;
mod optimizer;
mod simulation;
pub mod strategy_optimizer;

use crate::optimizer::calibrate_signal_calculator;
use crate::orderbook_values::{Leg, OrderBookValues, WindowValuesCalculator};
use crate::signal_calculator::{CalculatorsPair, PairPerformance, SignalCalculator};
use chrono::{TimeDelta, Utc};
use log::{debug, info};
use model::events::OrderBookEvent;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

const MAX_WINDOW_LENGTH_S: i64 = 2000;
const TRAIN_DATA_MINUTES: u16 = 45;
const TRAIN_INTERVAL_MINUTES: u16 = 1;
const DEFAULT_DECAY_TIME: f64 = 15.0;
const STD_DIAPASON_SECONDS: u16 = 1200;

pub const HOLD_TIME_VARIANTS: [u16; 6] = [200, 300, 500, 750, 1000, 1500];

#[derive(Copy, Clone)]
pub enum TradeSignal {
    None,
    Sell1Buy2(u16),
    Buy1Sell2(u16),
}

#[derive(Copy, Clone, Debug)]
pub struct ModelConfig {
    pub std_diapason_s: u16,
    pub train_data_minutes: u16,
    pub train_interval_minutes: u16,
    pub decay_time: f64,
    pub fixed_hold_time: Option<u16>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            std_diapason_s: STD_DIAPASON_SECONDS,
            train_data_minutes: TRAIN_DATA_MINUTES,
            train_interval_minutes: TRAIN_INTERVAL_MINUTES,
            decay_time: DEFAULT_DECAY_TIME,
            fixed_hold_time: None,
        }
    }
}

pub struct WindowData {
    values1_calculator: WindowValuesCalculator,
    values2_calculator: WindowValuesCalculator,
}

impl WindowData {
    pub fn new() -> Self {
        Self {
            values1_calculator: WindowValuesCalculator::new(Leg::First),
            values2_calculator: WindowValuesCalculator::new(Leg::Second),
        }
    }
}

#[derive(Clone)]
pub struct WalkForwardModel {
    instrument1_id: i32,
    instrument2_id: i32,

    config: ModelConfig,
    calculator: Arc<RwLock<Option<CalculatorsPair>>>,
    buffer: Arc<Mutex<Vec<OrderBookValues>>>,
}

#[derive(Copy, Clone, Default)]
pub struct CalibrationResponse {
    pub prev_cycle_performances: Option<PairPerformance>,
    pub up: Option<SignalCalculator>,
    pub down: Option<SignalCalculator>,
}

impl WalkForwardModel {
    pub fn new(instrument1_id: i32, instrument2_id: i32) -> Self {
        Self {
            instrument1_id,
            instrument2_id,
            config: ModelConfig::default(),
            calculator: Arc::new(RwLock::new(None)),
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn new_with_config(instrument1_id: i32, instrument2_id: i32, config: ModelConfig) -> Self {
        Self {
            instrument1_id,
            instrument2_id,
            config,
            calculator: Arc::new(RwLock::new(None)),
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_last_values(&self) -> Option<(OrderBookValues, OrderBookValues)> {
        let (mut v1, mut v2) = (None, None);

        if let Ok(lock) = self.buffer.lock() {
            let mut i = lock.len() - 1;
            while i > 0 && (v1.is_none() || v2.is_none()) {
                match lock[i].leg {
                    Leg::First => if v1.is_none() { v1 = Some(lock[i]) },
                    Leg::Second => if v2.is_none() { v2 = Some(lock[i]) },
                }
                i -= 1;
            }
        }

        if let Some(v1) = v1 && let Some(v2) = v2 {
            Some((v1, v2))
        } else {
            None
        }
    }

    pub fn process(&self, window_data: &mut WindowData, event: &OrderBookEvent) -> TradeSignal {
        let v = if self.instrument1_id == event.instrument_id {
            window_data.values1_calculator.calculate(&event.order_book, self.config.std_diapason_s)
        } else if self.instrument2_id == event.instrument_id {
            window_data.values2_calculator.calculate(&event.order_book, self.config.std_diapason_s)
        } else {
            return TradeSignal::None;
        };
        if let Ok(mut lock) = self.buffer.lock() {
            lock.push(v);
        }

        if let Ok(read) = self.calculator.read()
            && let Some(calculator) = read.as_ref()
            && let Some((v1, v2)) = self.get_last_values() {
            calculator.calculate(v1, v2)
        } else {
            TradeSignal::None
        }
    }
    
    pub fn inform_deal_result(&self, trade_signal: TradeSignal, profit: f64) {
        if let Ok(mut write) = self.calculator.write()
            && let Some(calc) = write.as_mut() {
            calc.deal_performance(trade_signal, profit);
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

        let decay_time = self.config.decay_time;
        let fixed_hold_time = self.config.fixed_hold_time;

        let buf_up = self.buffer.lock().map(|b| b.clone()).unwrap_or_default();
        let buf_down = buf_up.clone();
        let thread_up = thread::spawn(move || {
            calibrate_signal_calculator(&buf_up, true, decay_time, fixed_hold_time)
        });

        let thread_down = thread::spawn(move || {
            calibrate_signal_calculator(&buf_down, false, decay_time, fixed_hold_time)
        });

        if let Ok(up) = thread_up.join()
            && let Ok(down) = thread_down.join() {
            let last_time = self.buffer.lock()
                .map(|buf| buf[buf.len() - 1].time).unwrap_or(Utc::now());
            let calc = CalculatorsPair::new(up, down, last_time);
            response.up = up;
            response.down = down;

            debug!("{calc:?}");
            if let Ok(mut write) = self.calculator.write() {
                *write = Some(calc);
            }
        }

        response
    }
    
    pub fn set_calculator(&self, calculator: Option<CalculatorsPair>) {
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