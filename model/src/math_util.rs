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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

pub fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}