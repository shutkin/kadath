use chrono::{DateTime, TimeDelta, Utc};
use std::cmp::Ordering::Equal;
use std::collections::HashMap;

pub fn standard_deviation(data: &[f64]) -> Option<f64> {
    let count = data.len();
    if count < 2 {
        return None;
    }

    let mean = data.iter().sum::<f64>() / count as f64;
    let variance = data.iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>() / count as f64;

    Some(variance.sqrt())
}

pub fn median(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal));
    if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}

pub fn p10_p90(data: &[f64]) -> (f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (sorted[sorted.len() / 10], sorted[sorted.len() - 1 - sorted.len() / 10])
}

pub fn p05_p95(data: &[f64]) -> (f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (sorted[sorted.len() / 20], sorted[sorted.len() - 1 - sorted.len() / 20])
}

pub fn percentile(data: &[f64], percentile: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((sorted.len() - 1) as f64 * percentile) as usize;
    sorted[index]
}

pub fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

#[derive(Copy, Clone)]
pub struct ExponentFunction {
    a: f64,
    b: f64,
}

impl ExponentFunction {
    pub fn find(y_low: f64, y_mid: f64, y_high: f64) -> Self {
        let (low_ln, mid_ln, high_ln) = (y_low.ln(), y_mid.ln(), y_high.ln());
        let sum_ln = low_ln + mid_ln + high_ln;
        let sum_x_ln = -low_ln + high_ln;
        let b = sum_ln / 3.0;
        let a = sum_x_ln / 2.0;
        Self { a, b }
    }

    pub fn calculate(&self, x: f64) -> f64 {
        (x * self.a + self.b).exp()
    }
}

#[derive(Clone, Debug)]
pub struct TimedFeatures {
    pub time: DateTime<Utc>,
    pub features: Vec<f64>,
}

impl TimedFeatures {
    pub fn new(time: DateTime<Utc>, features: &[f64]) -> Self {
        Self {
            time,
            features: features.to_vec(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum OutputType {
    Raw(&'static str),
    DerivativeNorm(&'static str),
    TrendDeviation(&'static str),
    TrendDeviationDerivativeNorm(&'static str),
}

#[derive(Clone)]
pub struct FeaturesRollingWindow {
    norm_diapason: TimeDelta,
    trend_tau: TimeDelta,

    name_to_index: HashMap<String, usize>,
    outputs: Vec<OutputType>,

    prev_features: Option<TimedFeatures>,
    times: Vec<DateTime<Utc>>,
    derivatives: Vec<Vec<f64>>,
    trends: Vec<f64>,
    trend_derivatives: Vec<Vec<f64>>,

    derivatives_indices: Vec<usize>,
    trends_indices: Vec<usize>,
    trend_derivatives_indices: Vec<usize>,
}

impl FeaturesRollingWindow {
    pub fn new(norm_diapason: TimeDelta, trend_tau: TimeDelta, names: &[&'static str], outputs: &[OutputType]) -> Self {
        let mut name_to_index = HashMap::new();
        for (i, name) in names.iter().enumerate() {
            name_to_index.insert(name.to_string(), i);
        }

        let mut derivatives_indices = Vec::new();
        let mut trends_indices = Vec::new();
        let mut trend_derivatives_indices = Vec::new();
        for output in outputs {
            match output {
                OutputType::Raw(_) => {}
                OutputType::DerivativeNorm(name) => {
                    derivatives_indices.push(name_to_index[*name]);
                }
                OutputType::TrendDeviation(name) => {
                    trends_indices.push(name_to_index[*name]);
                }
                OutputType::TrendDeviationDerivativeNorm(name) => {
                    trend_derivatives_indices.push(name_to_index[*name]);
                }
            }
        }

        Self {
            norm_diapason,
            trend_tau,

            name_to_index,
            outputs: outputs.to_vec(),

            prev_features: None,
            times: Vec::new(),
            derivatives: vec![Vec::new(); names.len()],
            trends: vec![0.0; names.len()],
            trend_derivatives: vec![Vec::new(); names.len()],

            derivatives_indices,
            trends_indices,
            trend_derivatives_indices,
        }
    }

    pub fn process(&mut self, new_data: TimedFeatures) -> Vec<f32> {
        let last_time = new_data.time;
        while let Some(first_time) = self.times.first() && *first_time < last_time - self.norm_diapason {
            self.times.remove(0);
            self.derivatives.iter_mut()
                .for_each(|derivatives| {
                    derivatives.remove(0);
                });
            self.trend_derivatives.iter_mut()
                .for_each(|derivatives| {
                    derivatives.remove(0);
                });
        }
        let history_len = self.times.len();

        let mut derivatives = vec![0.0; self.name_to_index.len()];
        let mut derivatives_norm = vec![0.0; self.name_to_index.len()];

        let mut new_trends = vec![0.0; self.name_to_index.len()];
        let mut trend_derivatives = vec![0.0; self.name_to_index.len()];
        let mut trend_derivatives_norm = vec![0.0; self.name_to_index.len()];

        if let Some(prev_features) = &self.prev_features {
            let dt = (new_data.time - prev_features.time).as_seconds_f64();

            // calculate new trends
            let alpha = 1.0 - (-dt / self.trend_tau.as_seconds_f64()).exp();
            for &i in &self.trends_indices {
                new_trends[i] = self.trends[i] + alpha * (new_data.features[i] - self.trends[i]);
            }

            // calculate trends derivatives
            for &i in &self.trend_derivatives_indices {
                trend_derivatives[i] = if dt > 0.0 {
                    let new_deviation = new_data.features[i] - new_trends[i];
                    let prev_deviation = prev_features.features[i] - self.trends[i];
                    (new_deviation - prev_deviation) / dt
                } else {
                    0.0
                };

                // normalize
                let mut variance = 0.0;
                if !self.trend_derivatives[i].is_empty() {
                    let sum = self.trend_derivatives[i].iter()
                        .map(|value| { value * value })
                        .sum::<f64>() / history_len as f64;
                    variance = sum.sqrt();
                }
                trend_derivatives_norm[i] = if variance > 0.0 {
                    trend_derivatives[i] / variance
                } else {
                    trend_derivatives[i]
                };
            }

            // calculate derivatives
            for &i in &self.derivatives_indices {
                derivatives[i] = if dt > 0.0 {
                    (new_data.features[i] - prev_features.features[i]) / dt
                } else {
                    0.0
                };

                // normalize
                let mut variance = 0.0;
                if !self.derivatives[i].is_empty() {
                    let sum = self.derivatives[i].iter()
                        .map(|value| { value * value })
                        .sum::<f64>() / history_len as f64;
                    variance = sum.sqrt();
                }
                derivatives_norm[i] = if variance > 0.0 {
                    derivatives[i] / variance
                } else {
                    derivatives[i]
                };
            }
        } else {
            for &i in &self.trends_indices {
                new_trends[i] = new_data.features[i];
            }
        }

        let mut result = Vec::with_capacity(self.outputs.len());
        for output in &self.outputs {
            match output {
                OutputType::Raw(name) => {
                    let index = self.name_to_index[*name];
                    result.push(new_data.features[index]);
                }
                OutputType::DerivativeNorm(name) => {
                    let index = self.name_to_index[*name];
                    result.push(derivatives_norm[index]);
                }
                OutputType::TrendDeviation(name) => {
                    let index = self.name_to_index[*name];
                    result.push(new_data.features[index] - new_trends[index]);
                }
                OutputType::TrendDeviationDerivativeNorm(name) => {
                    let index = self.name_to_index[*name];
                    result.push(trend_derivatives_norm[index]);
                }
            }
        }

        self.trends = new_trends;
        self.derivatives.iter_mut().enumerate().for_each(|(i, history)| {
            history.push(derivatives[i]);
        });
        self.trend_derivatives.iter_mut().enumerate().for_each(|(i, history)| {
            history.push(trend_derivatives[i]);
        });
        self.times.push(new_data.time);
        self.prev_features = Some(new_data);

        result.into_iter().map(|f| f as f32).collect()
    }
}