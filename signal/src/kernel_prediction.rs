use chrono::{DateTime, TimeDelta, Utc};
use crate::orderbook_values::{IMBALANCE_LEVELS, OrderBookValues};

#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    /// Ширина гауссова ядра.
    /// Меньше — учитываются только очень похожие состояния.
    pub bandwidth: f64,

    /// Минимальное количество эффективных наблюдений,
    /// при котором оценка поддержки становится существенной.
    pub min_effective_samples: f64,

    /// Минимальное движение цены, считающееся ростом/падением.
    pub epsilon: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            bandwidth: 1.0,
            min_effective_samples: 5.0,
            epsilon: 0.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct KernelPrediction {
    /// Ожидаемое изменение mid price на выбранном горизонте.
    pub predicted_change: f64,

    /// Взвешенная вероятность роста.
    pub probability_up: f64,

    /// Взвешенная вероятность падения.
    pub probability_down: f64,

    /// Взвешенная вероятность нейтрального исхода.
    pub probability_flat: f64,

    /// Сила прогноза в диапазоне [0, 1].
    pub strength: f64,

    /// Согласованность направлений исторических исходов [0, 1].
    pub direction_consistency: f64,

    /// Сходство с наиболее близким историческим состоянием [0, 1].
    pub nearest_similarity: f64,

    /// Эффективное количество наблюдений с учётом весов.
    pub effective_samples: f64,

    /// Количество исторических наблюдений с ненулевым весом.
    pub contributing_samples: usize,
}

#[derive(Debug, Clone)]
struct HistoricalSample {
    features: Vec<f64>,
    change: f64,
}

#[derive(Debug, Clone)]
struct FeatureScaler {
    /// Индексы признаков с ненулевой исторической дисперсией.
    active_indices: Vec<usize>,

    /// Стандартное отклонение каждого признака.
    scales: Vec<f64>,
}

fn extract_features(book: &OrderBookValues) -> Vec<f64> {
    let mut features = Vec::with_capacity(
        2 * IMBALANCE_LEVELS.len() + 2,
    );

    features.push(book.mid_derivative);

    features.extend(book.imbalances.iter().copied());
    features.extend(book.imbalances_derivatives.iter().copied());

    features.push(book.mid - book.trend);

    features
}

fn fit_scaler(samples: &[HistoricalSample]) -> Option<FeatureScaler> {
    if samples.is_empty() {
        return None;
    }

    let feature_count = samples[0].features.len();

    let mut means = vec![0.0; feature_count];

    for sample in samples {
        for (i, &value) in sample.features.iter().enumerate() {
            means[i] += value;
        }
    }

    for mean in &mut means {
        *mean /= samples.len() as f64;
    }

    let mut variances = vec![0.0; feature_count];

    for sample in samples {
        for (i, &value) in sample.features.iter().enumerate() {
            let delta = value - means[i];
            variances[i] += delta * delta;
        }
    }

    for variance in &mut variances {
        *variance /= samples.len() as f64;
    }

    let scales: Vec<f64> = variances
        .iter()
        .map(|variance| variance.sqrt())
        .collect();

    // Постоянные признаки не участвуют в расстоянии:
    // по истории они не помогают различать состояния.
    let active_indices: Vec<usize> = scales
        .iter()
        .enumerate()
        .filter_map(|(i, &scale)| {
            (scale.is_finite() && scale > 1e-12).then_some(i)
        })
        .collect();

    if active_indices.is_empty() {
        return None;
    }

    Some(FeatureScaler {
        active_indices,
        scales,
    })
}

fn squared_distance(
    a: &[f64],
    b: &[f64],
    scaler: &FeatureScaler,
) -> f64 {
    let mut sum = 0.0;

    for &i in &scaler.active_indices {
        let scale = scaler.scales[i];
        let delta = (a[i] - b[i]) / scale;

        sum += delta * delta;
    }

    // Средний квадрат нормализованного расстояния.
    // Это делает bandwidth менее зависимым от числа признаков.
    sum / scaler.active_indices.len() as f64
}

fn get_future_mid(history: &[OrderBookValues], index: usize, target_time: DateTime<Utc>) -> Option<f64> {
    let mut i = index;
    while i < history.len() && history[i].time < target_time {
        i += 1;
    }
    if i < history.len() {
        Some(history[i].mid)
    } else {
        None
    }
}

/// history:
///   (OrderBookValues, future_mid)
///
/// current:
///   состояние стакана, для которого строим прогноз.
///
/// В history желательно передавать только доступные на момент
/// прогнозирования наблюдения из выбранного временного окна.
pub fn predict_kernel(
    history: &[OrderBookValues],
    current: &OrderBookValues,
    config: KernelConfig,
    horizon_ms: i32,
) -> Option<KernelPrediction> {
    if history.is_empty()
        || !config.bandwidth.is_finite()
        || config.bandwidth <= 0.0
        || !config.min_effective_samples.is_finite()
        || config.min_effective_samples <= 0.0
        || !config.epsilon.is_finite()
        || config.epsilon < 0.0 {
        return None;
    }

    let current_features = extract_features(current);

    let time_delta = TimeDelta::milliseconds(horizon_ms as i64);
    let samples: Vec<HistoricalSample> = history
        .iter()
        .enumerate()
        .filter_map(|(i, book)| {
            let features = extract_features(book);
            
            if let Some(future_mid) = get_future_mid(history, i, book.time + time_delta) {
                let change = future_mid - book.mid;

                if features.len() != current_features.len()
                    || !future_mid.is_finite()
                    || !change.is_finite()
                    || !features.iter().all(|x| x.is_finite()) {
                    return None;
                }

                Some(HistoricalSample { features, change })
            } else {
                None
            }
        })
        .collect();

    let scaler = fit_scaler(&samples)?;

    if !current_features.iter().all(|x| x.is_finite()) {
        return None;
    }

    let bandwidth_sq = config.bandwidth * config.bandwidth;

    let mut sum_weights = 0.0;
    let mut sum_weights_sq = 0.0;
    let mut weighted_change = 0.0;
    let mut weighted_abs_change = 0.0;

    let mut up_weight = 0.0;
    let mut down_weight = 0.0;
    let mut flat_weight = 0.0;

    let mut nearest_similarity: f64 = 0.0;
    let mut contributing_samples = 0;

    for sample in &samples {
        let distance_sq = squared_distance(
            &current_features,
            &sample.features,
            &scaler,
        );

        if !distance_sq.is_finite() {
            continue;
        }

        // Гауссово ядро.
        let weight = (-distance_sq / (2.0 * bandwidth_sq)).exp();

        if weight <= 1e-12 {
            continue;
        }

        nearest_similarity = nearest_similarity.max(weight);
        contributing_samples += 1;

        sum_weights += weight;
        sum_weights_sq += weight * weight;

        weighted_change += weight * sample.change;
        weighted_abs_change += weight * sample.change.abs();

        if sample.change > config.epsilon {
            up_weight += weight;
        } else if sample.change < -config.epsilon {
            down_weight += weight;
        } else {
            flat_weight += weight;
        }
    }

    if sum_weights <= 0.0 || sum_weights_sq <= 0.0 {
        return None;
    }

    let predicted_change = weighted_change / sum_weights;

    let probability_up = up_weight / sum_weights;
    let probability_down = down_weight / sum_weights;
    let probability_flat = flat_weight / sum_weights;

    // Эффективное количество наблюдений:
    // при одинаковых весах оно равно количеству наблюдений,
    // а при доминировании одного наблюдения приближается к 1.
    let effective_samples =
        sum_weights * sum_weights / sum_weights_sq;

    // Насколько согласованы знаки будущих изменений.
    // +5 и -5 дают 0, а +5 и +4 дают значение, близкое к 1.
    let direction_consistency = if weighted_abs_change > 1e-12 {
        (weighted_change.abs() / weighted_abs_change).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Поддержка прогнозa большим количеством наблюдений.
    let sample_support = effective_samples
        / (effective_samples + config.min_effective_samples);

    // Сила падает, если:
    // - ближайшие исторические состояния недостаточно похожи;
    // - направления будущих движений противоречат друг другу;
    // - эффективных наблюдений мало.
    let strength = (
        nearest_similarity
            * direction_consistency
            * sample_support
    )
        .clamp(0.0, 1.0);

    Some(KernelPrediction {
        predicted_change,
        probability_up,
        probability_down,
        probability_flat,
        strength,
        direction_consistency,
        nearest_similarity,
        effective_samples,
        contributing_samples,
    })
}