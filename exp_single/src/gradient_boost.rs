use std::sync::Arc;
use chrono::TimeDelta;
use crate::orderbook_values::OrderBookValues;
use gbdt::config::Config;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use lightgbm3::{Booster, Dataset};
use log::{error, info, warn};
use optimizer::prelude::{FloatParam, IntParam, Parameter};
use optimizer::Study;
use model::common::CommonError;
use plotters::prelude::LogScalable;
use serde_json::{json, Value};
use model::math_util::{FeaturesRollingWindow, OutputType, TimedFeatures};
use model::math_util::OutputType::{DerivativeNorm, Raw, TrendDeviation, TrendDeviationDerivativeNorm};

#[derive(Copy, Clone, Debug)]
pub enum PredictionType {
    ClassificationPositive(f64), ClassificationNegative(f64)
}

impl PredictionType {
    pub fn get_label(&self, horizon: f64) -> f32 {
        match self {
            PredictionType::ClassificationPositive(threshold) => if horizon > *threshold {1.0} else {0.0},
            PredictionType::ClassificationNegative(threshold) => if horizon < *threshold {1.0} else {0.0},
        }
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
        cfg.set_feature_size(12);
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

#[derive(Clone, Debug)]
pub enum MachineConfig {
    GBDT(GbdtConfig), LightGBM(Value)
}

pub enum Machine {
    GBDT(GBDT), LightGBM(Booster)
}

pub struct GradientBoost {
    prediction_type: PredictionType,
    machine: Machine,
}

impl GradientBoost {
    pub fn train(prediction_type: PredictionType, config: &MachineConfig, data: &[OrderBookValues], horizons: &[f64]) -> Result<Self, CommonError> {
        let machine = match config {
            MachineConfig::GBDT(cfg) => {
                let gbdt = Self::train_gbdt(&cfg.to_config(), data, horizons, &prediction_type);
                Machine::GBDT(gbdt)
            }
            MachineConfig::LightGBM(cfg) => {
                let booster = Self::train_light_gbm(cfg, data, horizons, &prediction_type)?;
                Machine::LightGBM(booster)
            }
        };

        Ok(Self {
            prediction_type,
            machine,
        })
    }

    fn train_gbdt(cfg: &Config, data: &[OrderBookValues], horizons: &[f64], prediction_type: &PredictionType) -> GBDT {
        let mut train_data = data.iter().zip(horizons.iter())
            .map(|(values, horizon)| {
                Data::new_training_data(values_to_features(values), 1.0, prediction_type.get_label(*horizon), None)
            }).collect::<Vec<_>>();
        let mut gbdt = GBDT::new(cfg);
        gbdt.fit(&mut train_data);
        gbdt
    }

    fn train_light_gbm(cfg: &Value, data: &[OrderBookValues], horizons: &[f64], prediction_type: &PredictionType) -> Result<Booster, CommonError> {
        let features = data.iter().map(|values| values_to_features(values)).collect::<Vec<_>>();
        let labels = horizons.iter().map(|&h| prediction_type.get_label(h)).collect::<Vec<_>>();
        let dataset = Dataset::from_vec_of_vec(features, labels, true)?;
        let booster = Booster::train(dataset, cfg)?;
        Ok(booster)
    }

    pub fn test(&self, data: &[OrderBookValues]) -> Result<Vec<f64>, CommonError> {
        match &self.machine {
            Machine::GBDT(gbdt) => {
                Ok(Self::test_gbdt(gbdt, data))
            }
            Machine::LightGBM(booster) => {
                Self::test_light_gbm(booster, data)
            }
        }
    }

    fn test_gbdt(gbdt: &GBDT, data: &[OrderBookValues]) -> Vec<f64> {
        let test_data = data.iter().map(|values| {
            Data::new_training_data(values_to_features(values), 1.0, 0.0, None)
        }).collect::<Vec<_>>();
        let predict_vec = gbdt.predict(&test_data);
        predict_vec.into_iter().map(|p| p.as_f64()).collect()
    }

    fn test_light_gbm(booster: &Booster, data: &[OrderBookValues]) -> Result<Vec<f64>, CommonError> {
        let features = data.iter().map(|values| values_to_features(values)).collect::<Vec<_>>();
        let output = booster.predict_from_vec_of_vec(features, true)?;
        Ok(output.iter().map(|v| v[0]).collect())
    }
}

pub fn test_features_rolling_window(normalize_seconds: i64, trend_tau_seconds: i64, values: &[OrderBookValues]) {
    let mut window = FeaturesRollingWindow::new(
        TimeDelta::seconds(normalize_seconds),
        TimeDelta::seconds(trend_tau_seconds),
        &["mid", "i3", "i5", "i10", "i20", "i50"],
        &[
            TrendDeviation("mid"),
            TrendDeviationDerivativeNorm("mid"),
            Raw("i3"),
            Raw("i5"),
            Raw("i10"),
            Raw("i20"),
            Raw("i50"),
            DerivativeNorm("i3"),
            DerivativeNorm("i5"),
            DerivativeNorm("i10"),
            DerivativeNorm("i20"),
            DerivativeNorm("i50"),
        ]
    );
    for v in values {
        let features = values_to_features(v);
        let rolling_features = window.process(TimedFeatures::new(
            v.time,
            &[v.mid, v.imbalances[0], v.imbalances[1], v.imbalances[2], v.imbalances[3], v.imbalances[4]],
        ));

        let mut v_diffs = Vec::new();
        for i in 0..12 {
            if (features[i] - rolling_features[i]).abs() > 0.01 {
                v_diffs.push(format!("[{i}] {} != {}", features[i], rolling_features[i]))
            }
        }
        if !v_diffs.is_empty() {
            warn!("{}: {}", v.time, v_diffs.join(", "));
        }
    }
    info!("Test completed");
}

fn values_to_features(values: &OrderBookValues) -> Vec<f32> {
    vec![
        values.trend_deviation as f32,
        values.trend_deviation_derivative as f32,
        values.imbalances[0] as f32,
        values.imbalances[1] as f32,
        values.imbalances[2] as f32,
        values.imbalances[3] as f32,
        values.imbalances[4] as f32,
        values.imbalances_derivative[0] as f32,
        values.imbalances_derivative[1] as f32,
        values.imbalances_derivative[2] as f32,
        values.imbalances_derivative[3] as f32,
        values.imbalances_derivative[4] as f32,
    ]
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
    train_data: Arc<Vec<OrderBookValues>>,
    train_horizons: Arc<Vec<f64>>,
    test_data: Arc<Vec<OrderBookValues>>,
    test_horizons: Arc<Vec<f64>>,
) -> Result<MachineConfig, CommonError> {
    let num_leaves_param = IntParam::new(7, 127).name("num leaves");
    let learning_rate_param = FloatParam::new(0.005, 0.1).name("learning rate").log_scale();
    let feature_fraction_param = FloatParam::new(0.5, 1.0).name("feature fraction").log_scale();
    let bagging_fraction_param = FloatParam::new(0.6, 1.0).name("bagging fraction");
    let bagging_freq_param = IntParam::new(1, 5).name("bagging freq");
    let min_child_samples_param = IntParam::new(50, 2000).name("min-child samples").log_scale();
    let lambda_l1_param = FloatParam::new(0.0001, 10.0).name("lambda l1").log_scale();
    let lambda_l2_param = FloatParam::new(0.0001, 10.0).name("lambda l2").log_scale();

    info!("Starting optimization");
    let study: Study<f64> = Study::new(optimizer::Direction::Maximize);
    study
        .optimize(100, |trial: &mut optimizer::Trial| {
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
                check(&MachineConfig::LightGBM(value), prediction_type, train_data.as_ref(), train_horizons.as_ref(), test_data.as_ref(), test_horizons.as_ref())
            } else {
                Err(optimizer::Error::Internal("Failed to suggest config"))
            }
        })?;

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
        Ok(MachineConfig::LightGBM(best_value))
    } else {
        Err("Failed to get best config".into())
    }
}

pub fn optimize_gbdt (
    prediction_type: PredictionType,
    train_data: Arc<Vec<OrderBookValues>>,
    train_horizons: Arc<Vec<f64>>,
    test_data: Arc<Vec<OrderBookValues>>,
    test_horizons: Arc<Vec<f64>>,
) -> Result<MachineConfig, CommonError> {
    let max_depth_param = IntParam::new(2, 7).name("max depth");
    let iterations_param = IntParam::new(50, 200).name("iterations");
    let shrinkage_param = FloatParam::new(0.005, 0.1).name("shrinkage").log_scale();
    let min_leaf_size_param = IntParam::new(100, 500).name("min-leaf size");
    let data_sample_ratio = FloatParam::new(0.5, 1.0).name("data sample ratio");
    let feature_sample_ratio = FloatParam::new(0.5, 1.0).name("feature sample ratio");

    info!("Starting optimization");
    let study: Study<f64> = Study::new(optimizer::Direction::Minimize);
    study
        .optimize(200, |trial: &mut optimizer::Trial| {
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
                check(&MachineConfig::GBDT(config), prediction_type, train_data.as_ref(), train_horizons.as_ref(), test_data.as_ref(), test_horizons.as_ref())
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
    Ok(MachineConfig::GBDT(config))
}

fn check(
    machine_config: &MachineConfig,
    prediction_type: PredictionType,
    train_data: &[OrderBookValues],
    train_horizons: &[f64],
    test_data: &[OrderBookValues],
    test_horizons: &[f64],
) -> Result<f64, optimizer::Error> {
    let machine = GradientBoost::train(prediction_type, machine_config, train_data, train_horizons)
        .map_err(|err| {
            error!("Failed to train LightGBM: {err}");
            optimizer::Error::Internal("Failed to train LightGBM")
        })?;
    let predictions = machine.test(test_data)
        .map_err(|err| {
            error!("Failed to test LightGBM: {err}");
            optimizer::Error::Internal("Failed to test LightGBM")
        })?;
    let error = calculate_predictions_error(&predictions, test_horizons, prediction_type);
    info!("{} {machine_config:?}: {error}", if matches!(prediction_type, PredictionType::ClassificationPositive(_)) { "UP" } else { "DOWN" });
    Ok::<_, optimizer::Error>(error)
}

pub fn calculate_predictions_error(predictions: &[f64], horizons: &[f64], prediction_type: PredictionType) -> f64 {
    let count = predictions.len();
    let variance = predictions.iter().zip(horizons)
        .map(|(p, h)| {
            let diff = match prediction_type {
                PredictionType::ClassificationPositive(threshold) => if *h > threshold { 1.0 - p } else { *p },
                PredictionType::ClassificationNegative(threshold) => if *h < threshold { 1.0 - p } else { *p },
            };
            diff * diff
        })
        .sum::<f64>() / count as f64;
    variance.sqrt()
}
