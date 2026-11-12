use gbdt::config::Config;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use log::{error, info, warn};
use model::common::CommonError;
use model::math_util::percentile;
use optimizer::Study;
use optimizer::prelude::{FloatParam, IntParam, Parameter};
use std::sync::Arc;

#[derive(Copy, Clone, Debug)]
pub enum PredictionType {
    ClassificationPositive(f32),
    ClassificationNegative(f32),
}

impl PredictionType {
    pub fn get_label(&self, horizon: f32) -> f32 {
        match self {
            PredictionType::ClassificationPositive(threshold) => if horizon > *threshold {1.0} else {0.0},
            PredictionType::ClassificationNegative(threshold) => if horizon < *threshold {1.0} else {0.0},
        }
    }
}

pub struct GradientBoost {
    booster: GBDT,
}

pub enum DataVec<'a> {
    Owned(Vec<Vec<f32>>), Pointer(&'a [Vec<f32>])
}

impl GradientBoost {
    pub fn train(prediction_type: PredictionType, config: &GbdtConfig, data: DataVec, horizons: &[f32]) -> Result<Self, CommonError> {
        let mut train_data = match data {
            DataVec::Owned(vec) => {
                vec.iter().zip(horizons.iter())
                    .map(|(features, horizon)| {
                        Data::new_training_data(features.to_vec(), 1.0, prediction_type.get_label(*horizon), None)
                    }).collect::<Vec<_>>()
            }
            DataVec::Pointer(ptr) => {
                ptr.iter().zip(horizons.iter())
                    .map(|(features, horizon)| {
                        Data::new_training_data(features.to_vec(), 1.0, prediction_type.get_label(*horizon), None)
                    }).collect::<Vec<_>>()
            }
        };
        let mut gbdt = GBDT::new(&config.to_config());
        gbdt.fit(&mut train_data);
        Ok(Self {
            booster: gbdt,
        })
    }

    pub fn test(&self, data: &[Vec<f32>]) -> Result<Vec<f64>, CommonError> {
        let test_data = data.iter().map(|values| {
            Data::new_training_data(values.to_vec(), 1.0, 0.0, None)
        }).collect::<Vec<_>>();
        let predict_vec = self.booster.predict(&test_data);
        Ok(predict_vec.into_iter().map(|p| p as f64).collect())
    }
}

fn export_to_csv(filename: &str, data: &[Vec<f32>], horizons: &[f32]) {
    let size = data[0].len();
    let mut csv = Vec::new();

    let header = format!("{},horizon", (0..size).map(|i| format!("f{i}")).collect::<Vec<_>>().join(","));
    csv.push(header);

    for (features, horizon) in data.iter().zip(horizons.iter()) {
        if features.len() != size {
            warn!("Feature length mismatch: expected {}, got {}", size, features.len())
        }
        let row = format!("{},{horizon:.2}", features.iter().map(|f| format!("{f:.4}")).collect::<Vec<_>>().join(","));
        csv.push(row);
    }

    if let Err(err) = std::fs::write(filename, csv.join("\n")) {
        error!("Failed to export data to CSV: {}", err);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GbdtConfig {
    pub max_depth: i64,
    pub iterations: i64,
    pub shrinkage: f64,
    pub min_leaf_size: i64,
    pub data_sample_ratio: f64,
    pub feature_sample_ratio: f64,
}

impl GbdtConfig {
    pub fn to_config(&self) -> Config {
        let mut cfg = Config::new();
        cfg.set_feature_size(13);
        cfg.set_debug(false);
        cfg.set_loss("LogLikelyhood");

        cfg.set_data_sample_ratio(self.data_sample_ratio);
        cfg.set_feature_sample_ratio(self.feature_sample_ratio);
        cfg.set_max_depth(self.max_depth as u32);
        cfg.set_iterations(self.iterations as usize);
        cfg.set_shrinkage(self.shrinkage as f32);
        cfg.set_min_leaf_size(self.min_leaf_size as usize);
        cfg
    }
}

pub fn optimize_gbdt (
    prediction_type: PredictionType,
    train_data: Arc<Vec<Vec<f32>>>,
    train_horizons: Arc<Vec<f32>>,
    test_data: Arc<Vec<Vec<f32>>>,
    test_horizons: Arc<Vec<f32>>,
    iterations: usize,
) -> Result<GbdtConfig, CommonError> {
    let max_depth_param = IntParam::new(2, 7).name("max depth");
    let iterations_param = IntParam::new(50, 200).name("iterations");
    let shrinkage_param = FloatParam::new(0.005, 0.1).name("shrinkage").log_scale();
    let min_leaf_size_param = IntParam::new(100, 500).name("min-leaf size");
    let data_sample_ratio = FloatParam::new(0.5, 1.0).name("data sample ratio");
    let feature_sample_ratio = FloatParam::new(0.5, 1.0).name("feature sample ratio");

    info!("Starting optimization");
    let study: Study<f64> = Study::new(optimizer::Direction::Maximize);
    study
        .optimize(iterations, |trial: &mut optimizer::Trial| {
            if let Some(max_depth) = max_depth_param.suggest(trial).ok()
                && let Some(iterations) = iterations_param.suggest(trial).ok()
                && let Some(shrinkage) = shrinkage_param.suggest(trial).ok()
                && let Some(min_leaf_size) = min_leaf_size_param.suggest(trial).ok()
                && let Some(data_sample_ratio) = data_sample_ratio.suggest(trial).ok()
                && let Some(feature_sample_ratio) = feature_sample_ratio.suggest(trial).ok() {
                let config = GbdtConfig {
                    max_depth,
                    iterations,
                    shrinkage,
                    min_leaf_size,
                    data_sample_ratio,
                    feature_sample_ratio,
                };
                check(&config, prediction_type, train_data.as_ref(), train_horizons.as_ref(), test_data.as_ref(), test_horizons.as_ref())
            } else {
                Err(optimizer::Error::Internal("Failed to suggest config"))
            }
        })?;

    let best = study.best_trial()?;
    info!("{} best: {}", if matches!(prediction_type, PredictionType::ClassificationPositive(_)) { "UP" } else { "DOWN" }, best.value);
    let config = GbdtConfig {
        max_depth: best.get(&max_depth_param).ok_or("max_depth")?,
        iterations: best.get(&iterations_param).ok_or("iterations")?,
        shrinkage: best.get(&shrinkage_param).ok_or("shrinkage")?,
        min_leaf_size: best.get(&min_leaf_size_param).ok_or("min_leaf_size")?,
        data_sample_ratio: best.get(&data_sample_ratio).ok_or("data_sample_ratio")?,
        feature_sample_ratio: best.get(&feature_sample_ratio).ok_or("feature_sample_ratio")?,
    };
    Ok(config)
}

fn check(
    config: &GbdtConfig,
    prediction_type: PredictionType,
    train_data: &[Vec<f32>],
    train_horizons: &[f32],
    test_data: &[Vec<f32>],
    test_horizons: &[f32],
) -> Result<f64, optimizer::Error> {
    let machine = GradientBoost::train(prediction_type, config, DataVec::Pointer(train_data), train_horizons)
        .map_err(|err| {
            error!("Failed to train LightGBM: {err}");
            optimizer::Error::Internal("Failed to train LightGBM")
        })?;
    let predictions = machine.test(test_data)
        .map_err(|err| {
            error!("Failed to test LightGBM: {err}");
            optimizer::Error::Internal("Failed to test LightGBM")
        })?;
    let score = calculate_predictions_brier(&predictions, test_horizons, prediction_type);
    info!("{} {config:?}: {score}", if matches!(prediction_type, PredictionType::ClassificationPositive(_)) { "UP" } else { "DOWN" });
    Ok::<_, optimizer::Error>(score)
}

pub fn calculate_predictions_pnl(predictions: &[f64], horizons: &[f32], prediction_type: PredictionType) -> f64 {
    let threshold = percentile(predictions, 0.999);
    let (mut pnl, mut commission) = (0.0, 0.0);
    predictions.iter().zip(horizons)
        .filter(|(p, _)| **p >= threshold)
        .for_each(|(_, h)| {
            match prediction_type {
                PredictionType::ClassificationPositive(_) => {
                    pnl += *h as f64;
                    commission += 2.0;
                },
                PredictionType::ClassificationNegative(_) => {
                    pnl -= *h as f64;
                    commission += 2.0;
                },
            };
        });
    pnl - commission
}

pub fn calculate_predictions_brier(predictions: &[f64], horizons: &[f32], prediction_type: PredictionType) -> f64 {
    let count = predictions.len();
    let variance = predictions.iter().zip(horizons)
        .map(|(p, h)| {
            let diff = match prediction_type {
                PredictionType::ClassificationPositive(threshold) => if *h > threshold { 1.0 - *p } else { *p },
                PredictionType::ClassificationNegative(threshold) => if *h < threshold { 1.0 - *p } else { *p },
            };
            diff * diff
        })
        .sum::<f64>() / count as f64;
    let error = variance.sqrt();
    1.0 - error
}
