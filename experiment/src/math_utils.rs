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
