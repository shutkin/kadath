mod orderbook_values;
pub mod signal_calculator;
mod optimizer;
mod simulation;
mod strategy_optimizer;

use crate::optimizer::calibrate_signal_calculator;
use crate::orderbook_values::{Leg, OrderBookValues, WindowValuesCalculator};
use crate::signal_calculator::{CalculatorsPair, PairPerformance};
use chrono::TimeDelta;
use log::debug;
use std::thread;
use model::events::OrderBookEvent;

const MAX_WINDOW_LENGTH_S: i64 = 2000;
const TRAIN_DATA_MINUTES: u16 = 90;
const TRAIN_INTERVAL_MINUTES: u16 = 1;
const DEFAULT_DECAY_TIME: f64 = 0.25 * 60.0;
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

pub struct WalkForwardModel {
    instrument1_id: i32,
    instrument2_id: i32,

    config: ModelConfig,

    values1_calculator: WindowValuesCalculator,
    values2_calculator: WindowValuesCalculator,

    calculator: Option<CalculatorsPair>,

    buffer: Vec<OrderBookValues>,
}

impl WalkForwardModel {
    pub fn new(instrument1_id: i32, instrument2_id: i32) -> Self {
        Self {
            instrument1_id,
            instrument2_id,
            config: ModelConfig::default(),

            values1_calculator: WindowValuesCalculator::new(Leg::First),
            values2_calculator: WindowValuesCalculator::new(Leg::Second),

            calculator: None,

            buffer: Vec::new(),
        }
    }

    pub fn new_with_config(instrument1_id: i32, instrument2_id: i32, config: ModelConfig) -> Self {
        Self {
            instrument1_id,
            instrument2_id,
            config,

            values1_calculator: WindowValuesCalculator::new(Leg::First),
            values2_calculator: WindowValuesCalculator::new(Leg::Second),

            calculator: None,

            buffer: Vec::new(),
        }
    }

    fn get_last_values(&self) -> Option<(&OrderBookValues, &OrderBookValues)> {
        let (mut v1, mut v2) = (None, None);
        let mut i = self.buffer.len() - 1;
        while i > 0 && (v1.is_none() || v2.is_none()) {
            match self.buffer[i].leg {
                Leg::First => if v1.is_none() { v1 = Some(&self.buffer[i]) },
                Leg::Second => if v2.is_none() { v2 = Some(&self.buffer[i]) },
            }
            i -= 1;
        }

        if let Some(v1) = v1 && let Some(v2) = v2 {
            Some((v1, v2))
        } else {
            None
        }
    }

    pub fn process(&mut self, event: &OrderBookEvent) -> TradeSignal {
        let v = if self.instrument1_id == event.instrument_id {
            self.values1_calculator.calculate(&event.order_book, self.config.std_diapason_s)
        } else if self.instrument2_id == event.instrument_id {
            self.values2_calculator.calculate(&event.order_book, self.config.std_diapason_s)
        } else {
            return TradeSignal::None;
        };
        self.buffer.push(v);

        if let Some(calculator) = &self.calculator
            && let Some((v1, v2)) = self.get_last_values() {
            calculator.calculate(v1, v2)
        } else {
            TradeSignal::None
        }
    }
    
    pub fn inform_deal_result(&mut self, trade_signal: TradeSignal, profit: f64) {
        if let Some(calc) = self.calculator.as_mut() {
            calc.deal_performance(trade_signal, profit);
        }
    }
    
    pub fn calibrate(&mut self) -> Option<PairPerformance> {
        if self.buffer.len() < 2 {
            return None;
        }
        let delta = self.buffer[self.buffer.len() - 1].time - self.buffer[0].time;
        let buf_minutes = delta.num_minutes() as u16;
        if buf_minutes < self.config.train_data_minutes - 1 {
            return None;
        }

        let last_time = self.buffer[self.buffer.len() - 1].time;
        if self.calculator.as_ref()
            .map(|calc| last_time - calc.get_created_on() > TimeDelta::minutes(self.config.train_interval_minutes as i64))
            .unwrap_or(true) {
            self.shrink_buffer();
            
            let performances = self.calculator.map(|calc| calc.get_performances());
            
            let decay_time = self.config.decay_time;
            let fixed_hold_time = self.config.fixed_hold_time;

            let buf_clone = self.buffer.clone();
            let thread_up = thread::spawn(move || {
                calibrate_signal_calculator(&buf_clone, true, decay_time, fixed_hold_time)
            });

            let buf_clone = self.buffer.clone();
            let thread_down = thread::spawn(move || {
                calibrate_signal_calculator(&buf_clone, false, decay_time, fixed_hold_time)
            });

            if let Ok(up) = thread_up.join()
                && let Ok(down) = thread_down.join() {
                let last_time = self.buffer[self.buffer.len() - 1].time;
                let calc = CalculatorsPair::new(up, down, last_time);

                debug!("{calc:?}");
                self.calculator = Some(calc);
            }
            
            performances
        } else {
            None
        }
    }

    fn shrink_buffer(&mut self) {
        let threshold = self.buffer[self.buffer.len() - 1].time - TimeDelta::minutes(self.config.train_data_minutes as i64);
        let mut to_delete = 0;
        while to_delete < self.buffer.len() && self.buffer[to_delete].time < threshold {
            to_delete += 1;
        }

        if to_delete > 0 {
            self.buffer.drain(0..to_delete);
        }
    }
}