use std::cmp::Ordering::Equal;

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
