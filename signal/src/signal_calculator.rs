use crate::TradeSignal;
use crate::orderbook_values::{IMBALANCE_LEVELS, OrderBookValues};
use chrono::{DateTime, Utc};
use ndarray::Array1;

#[derive(Copy, Clone, Debug)]
pub struct SignalCalculator {
    hold_time_ms: u16,
    threshold: f64,
    derivative1_weight: f64,
    derivative2_weight: f64,
    imbalance1_weights: [f64; IMBALANCE_LEVELS.len()],
    imbalance2_weights: [f64; IMBALANCE_LEVELS.len()],
    performance: SignalPerformance,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct SignalPerformance {
    pub chosen_hold_time: u16,

    pub training_deals: u32,
    pub training_total_pnl: f64,

    pub actual_deals: u32,
    pub actual_wins: u32,
    pub actual_total_pnl: f64,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct PairPerformance {
    pub created_on: DateTime<Utc>,
    pub up: SignalPerformance,
    pub down: SignalPerformance,
}

impl SignalCalculator {
    pub fn from_optimizer_param(hold_time_ms: u16, param: &Array1<f64>) -> Self {
        Self {
            hold_time_ms,
            threshold: param[0],
            derivative1_weight: param[1].max(0.0),
            derivative2_weight: param[2],
            imbalance1_weights: [param[3], param[4], param[5], param[6], param[7]],
            imbalance2_weights: [param[8], param[9], param[10], param[11], param[12]],
            performance: SignalPerformance::default(),
        }
    }

    pub fn set_performance(&mut self, performance: SignalPerformance) {
        self.performance = performance;
    }
    
    pub fn get_train_profit(&self) -> f64 {
        self.performance.training_total_pnl
    }
    
    pub fn get_hold_time_ms(&self) -> u16 {
        self.hold_time_ms
    }
    
    fn deal_performance(&mut self, deal_profit: f64) {
        self.performance.actual_total_pnl += deal_profit;
        self.performance.actual_deals += 1;
        if deal_profit > 0.0 {
            self.performance.actual_wins += 1;
        }
    }

    fn calculate(&self, v1: &OrderBookValues, v2: &OrderBookValues) -> f64 {
        let mut score = self.derivative1_weight * v1.normal_derivative.abs()
            + self.derivative2_weight * v2.normal_derivative;
        for i in 0..IMBALANCE_LEVELS.len() {
            score += self.imbalance1_weights[i] * v1.imbalances[i]
                + self.imbalance2_weights[i] * v2.imbalances[i];
        }
        score - self.threshold
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CalculatorsPair {
    created_on: DateTime<Utc>,
    up: Option<SignalCalculator>,
    down: Option<SignalCalculator>,
}

impl CalculatorsPair {
    pub fn new_up(up: SignalCalculator, last_time: DateTime<Utc>) -> Self {
        Self {
            created_on: last_time,
            up: Some(up),
            down: None,
        }
    }
    
    pub fn new_down(down: SignalCalculator, last_time: DateTime<Utc>) -> Self {
        Self {
            created_on: last_time,
            up: None,
            down: Some(down),
        }
    }
    
    pub fn new(up: Option<SignalCalculator>, down: Option<SignalCalculator>, last_time: DateTime<Utc>) -> Self {
        Self {
            created_on: last_time,
            up,
            down,
        }
    }
    
    pub fn deal_performance(&mut self, trade_signal: TradeSignal, deal_profit: f64) {
        match trade_signal {
            TradeSignal::Sell1Buy2(_) => {
                if let Some(up) = self.up.as_mut() {
                    up.deal_performance(deal_profit);
                }
            }
            TradeSignal::Buy1Sell2(_) => {
                if let Some(down) = self.down.as_mut() {
                    down.deal_performance(deal_profit);
                }
            }
            TradeSignal::None => {}
        }
    }

    pub fn get_performances(&self) -> PairPerformance {
        let perf_up = self.up.map(|i| i.performance).unwrap_or_default();
        let perf_down = self.down.map(|i| i.performance).unwrap_or_default();
        
        PairPerformance {
            up: perf_up,
            down: perf_down,
            created_on: self.created_on,
        }
    }
    
    pub fn get_created_on(&self) -> DateTime<Utc> {
        self.created_on
    }
    
    pub fn calculate(&self, v1: &OrderBookValues, v2: &OrderBookValues) -> TradeSignal {
        let calculator = if v1.normal_derivative > 0.0 {&self.up} else {&self.down};
        if let Some(calculator) = calculator {
            if calculator.calculate(v1, v2) > 0.0 {
                if v1.normal_derivative > 0.0 {
                    TradeSignal::Sell1Buy2(calculator.hold_time_ms)
                } else {
                    TradeSignal::Buy1Sell2(calculator.hold_time_ms)
                }
            } else {
                TradeSignal::None
            }
        } else {
            TradeSignal::None
        }
    }
}
