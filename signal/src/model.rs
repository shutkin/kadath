extern crate core;

pub mod orderbook_values;
pub mod gradient_boost;
mod nelder_mead;

use crate::gradient_boost::{DataVec, GbdtConfig, GradientBoost, PredictionType, optimize_gbdt};
use crate::nelder_mead::{LinearSignal, calibrate_linear_signal};
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use log::{error, info};
use model::events::OrderBookEvent;
use model::math_util::OutputType::{DerivativeNorm, Raw, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{FeaturesRollingWindow, get_normalized_ranks, percentile};
use model::{OrderBook, TradeDirection};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

const LAST_PREDICTIONS_SIZE: usize = 50000;

#[derive(Copy, Clone, Default)]
pub enum TradeSignal {
    #[default]
    None,
    Sell(f64, f64),
    Buy(f64, f64),
}

#[derive(Copy, Clone, Debug)]
pub struct ModelConfig {
    pub derivatives_normalize_diapason_s: f64,
    pub trend_tau_s: f64,
    pub train_buf_size: usize,
    pub train_min_size: usize,
    pub optimize_iterations: usize,
    pub hold_time: TimeDelta,
    pub expected_delta: f32,
    pub prediction_percentile_linear: f64,
    pub prediction_percentile_gbm: f64,
}

pub struct WindowData {
    window: FeaturesRollingWindow,
    last_predictions_linear: Vec<f64>,
    last_predictions_gbm: Vec<f64>,
    last_predictions_gbm_positive: Vec<f64>,
    last_predictions_gbm_negative: Vec<f64>,
}

impl WindowData {
    pub fn new(config: &ModelConfig) -> Self {
        let window = FeaturesRollingWindow::new(
            TimeDelta::milliseconds((config.derivatives_normalize_diapason_s * 1000.0) as i64),
            TimeDelta::milliseconds((config.trend_tau_s * 1000.0) as i64),
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
            ],
        );
        Self {
            window,
            last_predictions_linear: Vec::new(),
            last_predictions_gbm: Vec::new(),
            last_predictions_gbm_negative: Vec::new(),
            last_predictions_gbm_positive: Vec::new(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DealData {
    direction: TradeDirection,
    open_time: DateTime<Utc>,
    open_price: f64,

    trend: f64,
    max_trend: f64,
    prev_price_time: DateTime<Utc>,
}

impl DealData {
    pub fn new(book: &OrderBook, direction: TradeDirection) -> Self {
        let cur_time = book.timestamp;
        let (_ask, _bid, mid) = OrderBookValues::calculate_ask_bid_mid(book);
        Self {
            direction,
            open_time: cur_time,
            open_price: mid,
            trend: 0.0,
            max_trend: 0.0,
            prev_price_time: cur_time,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct PriceHistory {
    time: DateTime<Utc>,
    ask: f32,
    bid: f32,
}

#[derive(Clone)]
pub struct WalkForwardModel {
    instrument_id: i32,
    config: ModelConfig,
    gbm_config_positive: Arc<RwLock<Option<GbdtConfig>>>,
    gbm_positive: Arc<RwLock<Option<GradientBoost>>>,
    gbm_config_negative: Arc<RwLock<Option<GbdtConfig>>>,
    gbm_negative: Arc<RwLock<Option<GradientBoost>>>,
    linear_signal: Arc<RwLock<Option<LinearSignal>>>,
    features_buffer: Arc<Mutex<Vec<Vec<f32>>>>,
    prices_buffer: Arc<Mutex<Vec<PriceHistory>>>,
}

impl WalkForwardModel {
    pub fn new(instrument_id: i32, config: ModelConfig) -> Self {
        Self {
            instrument_id,
            config,
            gbm_config_positive: Arc::new(RwLock::new(None)),
            gbm_config_negative: Arc::new(RwLock::new(None)),
            gbm_positive: Arc::new(RwLock::new(None)),
            gbm_negative: Arc::new(RwLock::new(None)),
            linear_signal: Arc::new(RwLock::new(None)),
            features_buffer: Arc::new(Mutex::new(Vec::new())),
            prices_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn is_it_time_to_close_deal(&self, book: &OrderBook, deal_data: &mut DealData) -> bool {
        let cur_time = book.timestamp;
        cur_time > deal_data.open_time + self.config.hold_time
    }

    pub fn process(&self, window_data: &mut WindowData, event: &OrderBookEvent) -> TradeSignal {
        let price;
        let features;
        if self.instrument_id == event.instrument_id {
            let book = OrderBookValues::from_order_book(&event.order_book);
            price = PriceHistory {
                time: event.order_book.timestamp,
                ask: book.ask as f32,
                bid: book.bid as f32,
            };
            features = window_data.window.process(book.to_features());
        } else {
            return TradeSignal::None;
        };

        let mut prediction_gbm_positive = None;
        let mut prediction_gbm_negative = None;
        let mut prediction_linear = None;

        if let Ok(mut lock) = self.features_buffer.lock() {
            lock.push(features.clone());
        }
        if let Ok(mut lock) = self.prices_buffer.lock() {
            lock.push(price);
        }

        if let Ok(read) = self.linear_signal.read()
            && let Some(linear) = read.as_ref() {
            prediction_linear = Some(linear.calculate(&features));
        }

        if let Ok(read) = self.gbm_positive.read()
            && let Some(booster) = read.as_ref() {
            match booster.test(&[features.clone()]) {
                Ok(predictions) => {
                    prediction_gbm_positive = predictions.into_iter().next();
                }
                Err(err) => {
                    error!("Failed to make positive prediction: {err}");
                }
            }
        }

        if let Ok(read) = self.gbm_negative.read()
            && let Some(booster) = read.as_ref() {
            match booster.test(&[features]) {
                Ok(predictions) => {
                    prediction_gbm_negative = predictions.into_iter().next();
                }
                Err(err) => {
                    error!("Failed to make negative prediction: {err}");
                }
            }
        }

        if let Some(prediction_linear) = prediction_linear
            && let Some(prediction_gbm_positive) = prediction_gbm_positive
            && let Some(prediction_gbm_negative) = prediction_gbm_negative {

            window_data.last_predictions_gbm_positive.push(prediction_gbm_positive);
            window_data.last_predictions_gbm_negative.push(prediction_gbm_negative);
            let prediction_gbm = if window_data.last_predictions_gbm_positive.len() > 3 {
                let ranks_positive = get_normalized_ranks(&window_data.last_predictions_gbm_positive, false);
                let ranks_negative = get_normalized_ranks(&window_data.last_predictions_gbm_negative, true);
                ranks_positive[ranks_positive.len() - 1] * ranks_negative[ranks_negative.len() - 1]
            } else {
                0.0
            };
            window_data.last_predictions_linear.push(prediction_linear);
            window_data.last_predictions_gbm.push(prediction_gbm);

            if window_data.last_predictions_linear.len() == LAST_PREDICTIONS_SIZE {
                let linear_threshold = percentile(&window_data.last_predictions_linear, self.config.prediction_percentile_linear);
                let gbm_threshold = percentile(&window_data.last_predictions_gbm, self.config.prediction_percentile_gbm);

                window_data.last_predictions_gbm_positive.remove(0);
                window_data.last_predictions_gbm_negative.remove(0);
                window_data.last_predictions_gbm.remove(0);
                window_data.last_predictions_linear.remove(0);

                if prediction_linear >= linear_threshold && prediction_gbm >= gbm_threshold {
                    return TradeSignal::Sell(prediction_linear, prediction_gbm)
                }
            }
        }

        TradeSignal::None
    }

    fn get_price_on_horizon(&self, prices: &[PriceHistory], index: usize) -> Option<(f32, f32)> {
        let mut i = index + 1;
        let target_time = prices[index].time + self.config.hold_time;
        while i < prices.len() {
            if prices[i].time >= target_time {
                return Some((prices[i].ask, prices[i].bid));
            }
            i += 1;
        }

        None
    }

    fn get_horizons(&self, direction: TradeDirection) -> Vec<f32> {
        self.prices_buffer.lock()
            .map(|prices| {
                if prices.len() < self.config.train_min_size {
                    Vec::new()
                } else {
                    let mut horizons = Vec::with_capacity(prices.len());
                    
                    for (i, price) in prices.iter().enumerate() {
                        if let Some((h_ask, h_bid)) = self.get_price_on_horizon(&prices, i) {
                            let pnl = h_bid - price.ask;
                            horizons.push(pnl);
                        }
                    }
                    horizons
                } 
            }).unwrap_or_default()
    }

    fn train_gbm(features: &[Vec<f32>], horizons: &[f32], cfg: GbdtConfig, prediction_type: PredictionType) -> Option<GradientBoost> {
        match GradientBoost::train(
            prediction_type,
            &cfg,
            DataVec::Pointer(features),
            &horizons,
        ) {
            Ok(boost) => {
                info!("New booster trained");
                Some(boost)
            }
            Err(err) => {
                error!("Failed to train: {err}");
                None
            }
        }
    }

    pub fn train(&self) -> bool {
        self.shrink_buffer();

        let horizons = self.get_horizons(TradeDirection::Buy);
        if horizons.len() >= self.config.train_min_size {
            let features = self.features_buffer.lock()
                .map(|buf| buf[..horizons.len()].to_vec());

            let cfg_positive = self.gbm_config_positive.read().map(|cfg| cfg.clone());
            let cfg_negative = self.gbm_config_negative.read().map(|cfg| cfg.clone());
            if let Ok(Some(cfg_positive)) = cfg_positive
                && let Ok(Some(cfg_negative)) = cfg_negative
                && let Ok(features_train) = features {

                let linear = calibrate_linear_signal(None, &features_train, &horizons);
                if let Some(linear) = linear && let Ok(mut write) = self.linear_signal.write() {
                    *write = Some(linear);
                }

                info!("Start training on {} events", features_train.len());
                let features_arc = Arc::new(features_train);
                let horizons_arc = Arc::new(horizons);
                let expected_delta = self.config.expected_delta;

                let features_clone = features_arc.clone();
                let horizons_clone = horizons_arc.clone();
                let h_positive = thread::spawn(move || {
                    Self::train_gbm(features_clone.as_ref(), horizons_clone.as_ref(), cfg_positive, PredictionType::ClassificationPositive(expected_delta))
                });

                let h_negative = thread::spawn(move || {
                    Self::train_gbm(features_arc.as_ref(), horizons_arc.as_ref(), cfg_negative, PredictionType::ClassificationNegative(0.0))
                });

                if let Ok(gbm_positive)  = h_positive.join() && let Ok(gbm_negative)  = h_negative.join()
                    && let Some(gbm_positive) = gbm_positive && let Some(gbm_negative) = gbm_negative {
                    if let Ok(mut write) = self.gbm_positive.write() {
                        *write = Some(gbm_positive);
                    }
                    if let Ok(mut write) = self.gbm_negative.write() {
                        *write = Some(gbm_negative);
                    }
                    info!("Training complete");
                    return true;
                }
            }
        }
        false
    }

    pub fn optimize_config(&self, predefined_configs: Option<(GbdtConfig, GbdtConfig)>) -> bool {
        if let Some((positive, negative)) = predefined_configs
            && let Ok(mut write_positive) = self.gbm_config_positive.write()
            && let Ok(mut write_negative) = self.gbm_config_negative.write() {
            *write_positive = Some(positive);
            *write_negative = Some(negative);
            info!("New predefined booster config saved");
            return true;
        }

        let horizons = self.get_horizons(TradeDirection::Buy);
        if horizons.len() >= self.config.train_min_size {
            let train_size = horizons.len() * 4 / 5;
            let features = self.features_buffer.lock()
                .map(|buf| {
                    (buf[..train_size].to_vec(), buf[train_size..horizons.len()].to_vec())
                });
            if let Ok((train, test)) = features {
                info!("Start GBM configs optimizations");

                let train_data_arc = Arc::new(train);
                let test_data_arc = Arc::new(test);
                let train_horizons_arc = Arc::new(horizons[..train_size].to_vec());
                let test_horizons_arc = Arc::new(horizons[train_size..].to_vec());
                let optimize_iterations = self.config.optimize_iterations;
                let expected_delta = self.config.expected_delta;

                let train_data_clone = train_data_arc.clone();
                let test_data_clone = test_data_arc.clone();
                let train_horizons_clone = train_horizons_arc.clone();
                let test_horizons_clone = test_horizons_arc.clone();
                let h_positive = thread::spawn(move || {
                    match optimize_gbdt(
                        PredictionType::ClassificationPositive(expected_delta),
                        train_data_clone, train_horizons_clone, test_data_clone, test_horizons_clone, optimize_iterations,
                    ) {
                        Ok(config) => Some(config),
                        Err(err) => {
                            error!("Failed to optimize: {err}");
                            None
                        }
                    }
                });

                let h_negative = thread::spawn(move || {
                    match optimize_gbdt(
                        PredictionType::ClassificationNegative(0.0),
                        train_data_arc, train_horizons_arc, test_data_arc, test_horizons_arc, optimize_iterations,
                    ) {
                        Ok(config) => Some(config),
                        Err(err) => {
                            error!("Failed to optimize: {err}");
                            None
                        }
                    }
                });

                if let Ok(positive) = h_positive.join() && let Ok(negative) = h_negative.join()
                    && let Some(positive) = positive && let Some(negative) = negative {
                    if let Ok(mut write_positive) = self.gbm_config_positive.write() {
                        *write_positive = Some(positive)
                    }
                    if let Ok(mut write_positive) = self.gbm_config_positive.write() {
                        *write_positive = Some(positive)
                    }
                    info!("New GBM configs saved");
                    return true;
                }
            }
        }

        false
    }

    fn shrink_buffer(&self) {
        let mut to_delete = 0;
        if let Ok(mut buffer) = self.features_buffer.lock()
            && buffer.len() > self.config.train_buf_size {
            to_delete = buffer.len() - self.config.train_buf_size;
            buffer.drain(0..to_delete);
        }

        if to_delete > 0 && let Ok(mut buffer) = self.prices_buffer.lock() {
            buffer.drain(0..to_delete);
        }

        if to_delete > 0 {
            info!("Delete {to_delete} buffer elements");
        }
    }
}