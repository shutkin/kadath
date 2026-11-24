extern crate core;

pub mod orderbook_values;
pub mod gradient_boost;
mod nelder_mead;
mod trend_filter;

use crate::gradient_boost::{DataVec, GradientBoost, PredictionType, optimize_light_gbm};
use crate::nelder_mead::{LinearSignal, calibrate_linear_signal};
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use log::{error, info};
use model::common::{PredictionDebugInfo, export_features_to_csv};
use model::events::OrderBookEvent;
use model::math_util::OutputType::{DerivativeNorm, Raw, StandardDeviation, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{BinaryRollingNormalizer, FeaturesRollingWindow, get_normalized_rank};
use model::{OrderBook, TradeDirection};
use serde_json::Value;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

const LAST_PREDICTIONS_SIZE: usize = 25000;

#[derive(Copy, Clone)]
pub enum TradeSignal {
    None(PredictionDebugInfo),
    Sell(PredictionDebugInfo),
    Buy(PredictionDebugInfo),
}

impl Default for TradeSignal {
    fn default() -> Self {
        TradeSignal::None(PredictionDebugInfo::default())
    }
}

impl TradeSignal {
    pub fn get_debug_info(&self) -> PredictionDebugInfo {
        match self {
            TradeSignal::None(info) => *info,
            TradeSignal::Sell(info) => *info,
            TradeSignal::Buy(info) => *info,
        }
    }
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
    pub trend_filter_length: f64,
}

pub struct WindowData {
    window: FeaturesRollingWindow,
    linear_normalizer: BinaryRollingNormalizer,
    gbm_positive_normalizer: BinaryRollingNormalizer,
    gbm_negative_normalizer: BinaryRollingNormalizer,
    //trend_filter: TrendFilter,
}

impl WindowData {
    pub fn new(config: &ModelConfig) -> Self {
        let window = FeaturesRollingWindow::new(
            TimeDelta::milliseconds((config.derivatives_normalize_diapason_s * 1000.0) as i64),
            TimeDelta::milliseconds((config.trend_tau_s * 1000.0) as i64),
            &["mid", "spread", "i3", "i5", "i10", "i20", "i50"],
            &[
                TrendDeviation("mid"),
                TrendDeviationDerivativeNorm("mid"),
                StandardDeviation("spread"),
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
            linear_normalizer: BinaryRollingNormalizer::new(LAST_PREDICTIONS_SIZE, false),
            gbm_positive_normalizer: BinaryRollingNormalizer::new(LAST_PREDICTIONS_SIZE, false),
            gbm_negative_normalizer: BinaryRollingNormalizer::new(LAST_PREDICTIONS_SIZE, true),
            //trend_filter: TrendFilter::new(TradeDirection::Buy, config.trend_filter_length),
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
    gbm_config_positive: Arc<RwLock<Option<Value>>>,
    gbm_positive: Arc<RwLock<Option<GradientBoost>>>,
    gbm_config_negative: Arc<RwLock<Option<Value>>>,
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

        let cur_price = OrderBookValues::from_order_book(book).mid;
        let pnl = match deal_data.direction {
            TradeDirection::Buy => cur_price - deal_data.open_price,
            TradeDirection::Sell => deal_data.open_price - cur_price,
        };
        if pnl < -0.5 {
            return true;
        }

        let dt = (cur_time - deal_data.prev_price_time).as_seconds_f64();
        let alpha = 1.0 - (-dt / 1.5).exp();
        deal_data.trend += alpha * (pnl - deal_data.trend);

        if deal_data.max_trend < deal_data.trend {
            deal_data.max_trend = deal_data.trend;
        }

        deal_data.prev_price_time = cur_time;

        cur_time > deal_data.open_time + self.config.hold_time && deal_data.trend < deal_data.max_trend - 0.5
    }

    pub fn process(&self, window_data: &mut WindowData, event: &OrderBookEvent) -> TradeSignal {
        let price;
        let features;
        let is_filtered_by_trend;

        let mut prediction_debug = PredictionDebugInfo::default();

        if self.instrument_id == event.instrument_id {
            let book = OrderBookValues::from_order_book(&event.order_book);
            price = PriceHistory {
                time: event.order_book.timestamp,
                ask: book.ask as f32,
                bid: book.bid as f32,
            };
            features = window_data.window.process(book.to_features());
            //window_data.trend_filter.process(&book);
            is_filtered_by_trend = true;
            prediction_debug.is_filtered_by_trend = is_filtered_by_trend;
        } else {
            return TradeSignal::None(PredictionDebugInfo::default());
        };

        if let Ok(mut lock) = self.features_buffer.lock() {
            lock.push(features.clone());
        }
        if let Ok(mut lock) = self.prices_buffer.lock() {
            lock.push(price);
        }

        let mut prediction_gbm_positive_opt = None;
        let mut prediction_gbm_negative_opt = None;
        let mut prediction_linear_opt = None;

        if let Ok(read) = self.linear_signal.read()
            && let Some(linear) = read.as_ref() {
            prediction_linear_opt = Some(linear.calculate(&features));
        }

        if let Ok(read) = self.gbm_positive.read()
            && let Some(booster) = read.as_ref() {
            match booster.test(std::slice::from_ref(&features)) {
                Ok(predictions) => {
                    prediction_gbm_positive_opt = predictions.into_iter().next();
                }
                Err(err) => {
                    error!("Failed to make positive prediction: {err}");
                }
            }
        }

        if let Ok(read) = self.gbm_negative.read()
            && let Some(booster) = read.as_ref() {
            match booster.test(std::slice::from_ref(&features)) {
                Ok(predictions) => {
                    prediction_gbm_negative_opt = predictions.into_iter().next();
                }
                Err(err) => {
                    error!("Failed to make negative prediction: {err}");
                }
            }
        }

        if let Some(prediction_linear_raw) = prediction_linear_opt
            && let Some(prediction_gbm_positive_raw) = prediction_gbm_positive_opt
            && let Some(prediction_gbm_negative_raw) = prediction_gbm_negative_opt {
            let linear_normalized = window_data.linear_normalizer.process(prediction_linear_raw);
            let gbm_positive_normalized = window_data.gbm_positive_normalizer.process(prediction_gbm_positive_raw);
            let gbm_negative_normalized = window_data.gbm_negative_normalizer.process(prediction_gbm_negative_raw);

            prediction_debug.linear_raw = prediction_linear_raw;
            prediction_debug.gbm_positive_raw = prediction_gbm_positive_raw;
            prediction_debug.gbm_negative_raw = prediction_gbm_negative_raw;

            if let Some(linear_normalized) = linear_normalized
                && let Some(gbm_positive_normalized) = gbm_positive_normalized
                && let Some(gbm_negative_normalized) = gbm_negative_normalized {
                let gbm_normalized = gbm_positive_normalized * (0.85 + 0.15 * gbm_negative_normalized);

                prediction_debug.linear_ranked = linear_normalized;
                prediction_debug.gbm_ranked = gbm_normalized;

                if is_filtered_by_trend
                    && linear_normalized < 0.98
                    && linear_normalized >= self.config.prediction_percentile_linear
                    && gbm_normalized >= self.config.prediction_percentile_gbm {
                    return TradeSignal::Buy(prediction_debug)
                }
            }
        }

        TradeSignal::None(prediction_debug)
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

    fn get_horizons(&self, direction: TradeDirection) -> (Vec<f32>, f64) {
        self.prices_buffer.lock()
            .map(|prices| {
                if prices.len() < self.config.train_min_size {
                    (Vec::new(), 1.0)
                } else {
                    let mut horizons = Vec::with_capacity(prices.len());

                    let millis = (prices[prices.len() - 1].time - prices[0].time).num_milliseconds();
                    let events_per_hold = self.config.hold_time.num_milliseconds() as f64 * prices.len() as f64 / millis as f64;
                    
                    for (i, price) in prices.iter().enumerate() {
                        if let Some((h_ask, h_bid)) = self.get_price_on_horizon(&prices, i) {
                            let pnl = match direction {
                                TradeDirection::Buy => h_bid - price.ask,
                                TradeDirection::Sell => price.bid - h_ask,
                            };
                            horizons.push(pnl);
                        }
                    }
                    (horizons, events_per_hold)
                } 
            }).unwrap_or_default()
    }

    fn do_gbm_training(features: &[Vec<f32>], horizons: &[f32], cfg: Value, prediction_type: PredictionType) -> Option<GradientBoost> {
        match GradientBoost::train(
            prediction_type,
            &cfg,
            DataVec::Pointer(features),
            horizons,
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

    pub fn train_linear(&self) -> bool {
        let (horizons, events_per_hold) = self.get_horizons(TradeDirection::Buy);
        if horizons.len() >= self.config.train_min_size {
            let features = self.features_buffer.lock()
                .map(|buf| buf[..horizons.len()].to_vec());

            if let Ok(features_train) = features {
                let prev = self.linear_signal.read().ok().and_then(|s| s.clone());
                let linear = calibrate_linear_signal(prev.as_ref(), &features_train, &horizons, events_per_hold as usize);
                if let Some(linear) = linear && let Ok(mut write) = self.linear_signal.write() {
                    *write = Some(linear);
                    return true;
                }
            }
        }
        false
    }

    pub fn train_gbm(&self, cur_time: DateTime<Utc>) -> bool {
        self.shrink_buffer();

        let (horizons, _) = self.get_horizons(TradeDirection::Buy);
        if horizons.len() >= self.config.train_min_size {
            let features = self.features_buffer.lock()
                .map(|buf| buf[..horizons.len()].to_vec());

            let cfg_positive = self.gbm_config_positive.read().map(|cfg| cfg.clone());
            let cfg_negative = self.gbm_config_negative.read().map(|cfg| cfg.clone());
            if let Ok(Some(cfg_positive)) = cfg_positive
                && let Ok(Some(cfg_negative)) = cfg_negative
                && let Ok(features_train) = features {

                //export_features_to_csv(&features_train, &horizons, "signal_train_data.csv");

                info!("Start training on {} events", features_train.len());
                let features_arc = Arc::new(features_train);
                let horizons_arc = Arc::new(horizons);
                let expected_delta = self.config.expected_delta;

                let features_clone = features_arc.clone();
                let horizons_clone = horizons_arc.clone();
                let h_positive = thread::spawn(move || {
                    Self::do_gbm_training(features_clone.as_ref(), horizons_clone.as_ref(), cfg_positive, PredictionType::ClassificationPositive(expected_delta))
                });

                let h_negative = thread::spawn(move || {
                    Self::do_gbm_training(features_arc.as_ref(), horizons_arc.as_ref(), cfg_negative, PredictionType::ClassificationNegative(0.0))
                });

                if let Ok(gbm_positive)  = h_positive.join() && let Ok(gbm_negative)  = h_negative.join()
                    && let Some(gbm_positive) = gbm_positive && let Some(gbm_negative) = gbm_negative {
                    if let Ok(mut write) = self.gbm_positive.write() {
                        *write = Some(gbm_positive);
                    }
                    if let Ok(mut write) = self.gbm_negative.write() {
                        *write = Some(gbm_negative);
                    }
                    info!("Training complete on {cur_time}");
                    return true;
                }
            }
        }
        false
    }

    pub fn optimize_config(&self, predefined_configs: Option<(Value, Value)>) -> bool {
        if let Some((positive, negative)) = predefined_configs
            && let Ok(mut write_positive) = self.gbm_config_positive.write()
            && let Ok(mut write_negative) = self.gbm_config_negative.write() {
            *write_positive = Some(positive);
            *write_negative = Some(negative);
            info!("New predefined booster config saved");
            return true;
        }

        let (horizons, _) = self.get_horizons(TradeDirection::Buy);
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
                    match optimize_light_gbm(
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
                    match optimize_light_gbm(
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
                        info!("Optimized positive: {positive}");
                        *write_positive = Some(positive)
                    }
                    if let Ok(mut write_negative) = self.gbm_config_negative.write() {
                        info!("Optimized negative: {negative}");
                        *write_negative = Some(negative)
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