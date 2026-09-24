use chrono::TimeDelta;
use crate::OrderBookValues;

pub const DEVIATION_MAX_VALUE: f64 = 100.0;

pub fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    }
}

pub fn p05_p95(data: &[f64]) -> (f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (sorted[sorted.len() / 20], sorted[sorted.len() - 1 - sorted.len() / 20])
}

pub fn positive_percentile(data: &[f64]) -> f64 {
    let positive = data.iter().filter(|x| **x > 0.0).count() as f64;
    100.0 * positive / data.len() as f64
}

fn calc_derivative(values: &[OrderBookValues], i: usize) -> f64 {
    if i < 1 {
        return 0.0;
    }

    let mut pi = i - 1;
    while values[pi].time == values[i].time {
        if pi == 0 {
            return 0.0;
        }
        pi -= 1;
    }
    (values[i].mid - values[pi].mid) / (values[i].time.timestamp_millis() - values[pi].time.timestamp_millis()) as f64
}

pub fn percentile(values: &[OrderBookValues], i: usize) -> f64 {
    if i < 3 {
        return 0.0;
    }

    let mut pi = i - 1;
    let avg_period_start = values[i].time - TimeDelta::seconds(5 * 60);
    let mut window_values = Vec::new();
    while pi > 1 && values[pi].time > avg_period_start {
        let derivative = calc_derivative(values, pi);
        window_values.push(derivative);
        pi -= 1;
    }
    window_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let derivative = calc_derivative(values, i);
    let mut window_index = 0;
    while window_index < window_values.len() && derivative < window_values[window_index] {
        window_index += 1;
    }
    let x = window_index as f64 / window_values.len() as f64;
    (x * 2.0).log2()
}

fn mad_normalize(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    if v.len() < 3 {
        return v;
    }

    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = v[v.len() / 2];

    let mut deviations = v.into_iter().map(|v| (v - median).abs()).collect::<Vec<_>>();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = deviations[deviations.len() / 2];

    let mut scale = 1.4826 * mad;
    if scale < f64::MIN_POSITIVE {
        scale = 1.0;
    }
    values.iter().map(|&v| (v - median) / scale).collect()
}

pub fn huber_loss(x: &[f64], target: &[f64], delta: f64) -> f64 {
    let target = mad_normalize(target);
    let mut result = 0.0;
    x.iter().zip(target.iter()).for_each(|(x, target)| {
        let r = target - x;
        let r_abs = r.abs();
        result += if r_abs <= delta {
            0.5 * r * r
        } else {
            delta * (r_abs - 0.5 * delta)
        };
    });
    result / x.len() as f64
}

#[cfg(test)]
mod tst {
#[test]
    fn tstexp() {
        for i in 0..7 {
            let x = 0.5 + 0.5 * i as f64 / 7.0;
            let y = (x * 2.0).log2();
            println!("{x} -> {y}");
        }
    }
}
