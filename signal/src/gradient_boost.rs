use lightgbm3::{Booster, Dataset};
use log::{debug, error, info};
use model::common::CommonError;
use optimizer::Study;
use optimizer::prelude::{FloatParam, IntParam, Parameter};
use serde_json::{Value, json};
use model::math_util::percentile;

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

struct BoosterSafe(Booster);
unsafe impl Sync for BoosterSafe {}
unsafe impl Send for BoosterSafe {}

pub struct GradientBoost {
    booster: BoosterSafe,
}

pub enum DataVec<'a> {
    Owned(Vec<Vec<f32>>), Pointer(&'a [Vec<f32>])
}

impl GradientBoost {
    pub fn train(prediction_type: PredictionType, config: &Value, data: DataVec, horizons: &[f32]) -> Result<Self, CommonError> {
        let labels = horizons.iter().map(|&h| prediction_type.get_label(h)).collect::<Vec<_>>();
        let dataset = match data {
            DataVec::Owned(vec) => Dataset::from_vec_of_vec(vec, labels, true),
            DataVec::Pointer(ptr) => Dataset::from_vec_of_vec(ptr.to_vec(), labels, true)
        }?;
        let booster = Booster::train(dataset, config)?;
        Ok(Self {
            booster: BoosterSafe(booster),
        })
    }

    pub fn test(&self, data: &[Vec<f32>]) -> Result<Vec<f64>, CommonError> {
        let output = self.booster.0.predict_from_vec_of_vec(data.to_vec(), true)?;
        Ok(output.iter().map(|v| v[0]).collect())
    }
}

struct LightGBMConfig {
    num_leaves: Option<i64>,
    min_child_samples: Option<i64>,
    learning_rate: Option<f64>,
    feature_fraction: Option<f64>,
    bagging_fraction: Option<f64>,
    bagging_freq: Option<i64>,
    lambda_l1: Option<f64>,
    lambda_l2: Option<f64>,
}

impl LightGBMConfig {
    fn to_json(&self) -> Option<Value> {
        if let Some(num_leaves) = self.num_leaves
            && let Some(min_child_samples) = self.min_child_samples
            && let Some(learning_rate) = self.learning_rate
            && let Some(feature_fraction) = self.feature_fraction
            && let Some(bagging_fraction) = self.bagging_fraction
            && let bagging_freq = self.bagging_freq
            && let lambda_l1 = self.lambda_l1
            && let lambda_l2 = self.lambda_l2
        {
            Some(json! {
                {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",

                    "num_leaves": num_leaves,
                    "min_child_samples": min_child_samples,
                    "learning_rate": learning_rate,
                    "feature_fraction": feature_fraction,
                    "bagging_fraction": bagging_fraction,
                    "bagging_freq": bagging_freq,
                    "lambda_l1": lambda_l1,
                    "lambda_l2": lambda_l2,

                    "num_iterations": 2000,
                    "max_depth": -1,
                    "verbosity": -1,
                    "random_state": 42,
                    "n_jobs": -1,
                }
            })
        } else {
            None
        }
    }
}

pub fn optimize_light_gbm (
    prediction_type: PredictionType,
    train_data: &[Vec<f32>],
    train_horizons: &[f32],
    test_data: &[Vec<f32>],
    test_horizons: &[f32],
    iterations: usize,
) -> Result<Value, CommonError> {
    let num_leaves_param = IntParam::new(7, 127).name("num leaves");
    let learning_rate_param = FloatParam::new(0.005, 0.1).name("learning rate").log_scale();
    let feature_fraction_param = FloatParam::new(0.5, 1.0).name("feature fraction").log_scale();
    let bagging_fraction_param = FloatParam::new(0.6, 1.0).name("bagging fraction");
    let bagging_freq_param = IntParam::new(1, 5).name("bagging freq");
    let min_child_samples_param = IntParam::new(50, 2000).name("min-child samples").log_scale();
    let lambda_l1_param = FloatParam::new(0.0001, 10.0).name("lambda l1").log_scale();
    let lambda_l2_param = FloatParam::new(0.0001, 10.0).name("lambda l2").log_scale();

    info!("Start optimization");
    let study: Study<f64> = Study::new(optimizer::Direction::Maximize);
    study
        .optimize(iterations, |trial: &mut optimizer::Trial| {
            let config = LightGBMConfig {
                learning_rate: learning_rate_param.suggest(trial).ok(),
                num_leaves: num_leaves_param.suggest(trial).ok(),
                min_child_samples: min_child_samples_param.suggest(trial).ok(),
                feature_fraction: feature_fraction_param.suggest(trial).ok(),
                bagging_fraction: bagging_fraction_param.suggest(trial).ok(),
                bagging_freq: bagging_freq_param.suggest(trial).ok(),
                lambda_l1: lambda_l1_param.suggest(trial).ok(),
                lambda_l2: lambda_l2_param.suggest(trial).ok(),
            };
            if let Some(value) = config.to_json() {
                check(&value, prediction_type, train_data, train_horizons, test_data, test_horizons)
            } else {
                Err(optimizer::Error::Internal("Failed to suggest config"))
            }
        })?;

    let best_score = study.best_value()?;
    info!("{} best score: {best_score}", if matches!(prediction_type, PredictionType::ClassificationPositive(_)) { "UP" } else { "DOWN" });
    let best = study.best_trial()?;
    let best_config = LightGBMConfig {
        learning_rate: best.get(&learning_rate_param),
        num_leaves: best.get(&num_leaves_param),
        min_child_samples: best.get(&min_child_samples_param),
        feature_fraction: best.get(&feature_fraction_param),
        bagging_fraction: best.get(&bagging_fraction_param),
        bagging_freq: best.get(&bagging_freq_param),
        lambda_l1: best.get(&lambda_l1_param),
        lambda_l2: best.get(&lambda_l2_param),
    };
    if let Some(best_value) = best_config.to_json() {
        info!(
            "{} best config: {}",
            if matches!(prediction_type, PredictionType::ClassificationPositive(_)) { "UP" } else { "DOWN" },
            serde_json::to_string(&best_value)?,
        );
        Ok(best_value)
    } else {
        Err("Failed to get best config".into())
    }
}

fn check(
    config: &Value,
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
    let score = calculate_predictions_pnl(&predictions, test_horizons, prediction_type);
    debug!("{} {config}: {score}", if matches!(prediction_type, PredictionType::ClassificationPositive(_)) { "UP" } else { "DOWN" });
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
