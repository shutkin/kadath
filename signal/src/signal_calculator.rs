use crate::orderbook_values::{IMBALANCE_LEVELS, OrderBookValues};
use chrono::{DateTime, Utc};
use ndarray::Array1;
use std::collections::HashMap;
use model::common::TimeDiapason;

#[derive(Copy, Clone, Debug)]
pub struct SignalCalculator {
    created_on: DateTime<Utc>,
    
    hold_time_ms: u16,
    threshold: f64,
    imbalance_weights: [f64; IMBALANCE_LEVELS.len()],
    imbalance_derivative_weights: [f64; IMBALANCE_LEVELS.len()],
    trend_weight: f64,
    mid_derivative_weight: f64,

    performance: SignalPerformance,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct SignalPerformance {
    pub train_diapason: TimeDiapason,

    pub hold_time: u16,

    pub training_score: f64,
    pub training_correct_predictions: i32,
    pub training_incorrect_predictions: i32,

    pub actual_deals: u32,
    pub actual_wins: u32,
    pub actual_total_pnl: f64,
}

impl SignalCalculator {
    pub fn from_optimizer_param(time: DateTime<Utc>, hold_time_ms: u16, param: &Array1<f64>) -> Self {
        Self {
            created_on: time,
            hold_time_ms,

            threshold: 0.0001 + param[0].abs(),
            imbalance_weights: [param[1], param[2], param[3], param[4], param[5]],
            imbalance_derivative_weights: [param[6], param[7], param[8], param[9], param[10]],
            trend_weight: param[11],
            mid_derivative_weight: 0.0,// param[12],
            
            performance: SignalPerformance::default(),
        }
    }

    pub fn set_performance(&mut self, performance: SignalPerformance) {
        self.performance = performance;
    }
    
    pub fn get_train_score(&self) -> f64 {
        self.performance.training_score
    }
    
    pub fn get_hold_time_ms(&self) -> u16 {
        self.hold_time_ms
    }
    
    pub(crate) fn deal_performance(&mut self, deal_profit: f64) {
        self.performance.actual_total_pnl += deal_profit;
        self.performance.actual_deals += 1;
        if deal_profit > 0.0 {
            self.performance.actual_wins += 1;
        }
    }

    pub fn get_performances(&self) -> SignalPerformance {
        self.performance.clone()
    }

    pub fn calculate(&self, v: &OrderBookValues) -> f64 {
        let mut score = 0.0;

        for (i, w) in self.imbalance_weights.iter().enumerate() {
            score += v.imbalances[i] * w
        }

        for (i, w) in self.imbalance_derivative_weights.iter().enumerate() {
            score += v.imbalances_derivative[i] * w
        }

        score += (v.trend - v.mid) * self.trend_weight;

        if score >= 0.0 {
            if score > self.threshold {
                score
            } else {
                0.0
            }
        } else {
            if score < -self.threshold {
                score
            } else {
                0.0
            }
        }
    }

    pub fn get_created_on(&self) -> DateTime<Utc> {
        self.created_on
    }

    pub fn to_params_map(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("hold".to_string(), self.hold_time_ms.into());
        map.insert("threshold".to_string(), self.threshold.into());
        map.insert("trend".to_string(), self.trend_weight.into());
        for (i, level) in IMBALANCE_LEVELS.iter().enumerate() {
            map.insert(format!("i1_{level}"), self.imbalance_weights[i]);
            map.insert(format!("di2_{level}"), self.imbalance_derivative_weights[i]);
        }
        map.insert("score".to_string(), self.performance.training_score);
        map
    }
    
    /*pub fn from_params_map(map: &HashMap<String, f64>) -> Option<Self> {
        Some(Self {
            hold_time_ms: map.get("hold").copied()? as u16,
            threshold: map.get("threshold").copied()?,
            derivative1_weight: map.get("d1").copied()?,
            derivative2_weight: map.get("d2").copied()?,
            imbalance1_weights: IMBALANCE_LEVELS.iter()
                .map(|level| map.get(&format!("i1_{level}")).copied().unwrap_or_default())
                .collect::<Vec<f64>>().try_into().ok()?,
            imbalance2_weights: IMBALANCE_LEVELS.iter()
                .map(|level| map.get(&format!("i2_{level}")).copied().unwrap_or_default())
                .collect::<Vec<f64>>().try_into().ok()?,
            performance: SignalPerformance::default(),
        })
    }*/
}
