extern crate core;

pub mod orderbook_values;
mod gradient_boost;
mod nelder_mead;

use crate::gradient_boost::{DataVec, GradientBoost, PredictionType, optimize_light_gbm};
use crate::nelder_mead::{LinearSignal, calibrate_linear_signal};
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use log::{error, info};
use model::events::OrderBookEvent;
use model::math_util::OutputType::{DerivativeNorm, Raw, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{FeaturesRollingWindow, percentile};
use model::{OrderBook, TradeDirection};
use serde_json::Value;
use std::sync::{Arc, Mutex, RwLock};

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
    last_features1: Option<Vec<f32>>,
    last_features2: Option<Vec<f32>>,
    window1: FeaturesRollingWindow,
    window2: FeaturesRollingWindow,
    last_predictions_linear: Vec<f64>,
    last_predictions_gbm: Vec<f64>,
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
            last_features1: None,
            last_features2: None,
            window1: window.clone(),
            window2: window.clone(),
            last_predictions_linear: Vec::new(),
            last_predictions_gbm: Vec::new(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DealData {
    direction: TradeDirection,
    open_time: DateTime<Utc>,
    open_price1: f64,
    open_price2: f64,

    trend: f64,
    max_trend: f64,
    prev_price_time: DateTime<Utc>,
}

impl DealData {
    pub fn new(book1: &OrderBook, book2: &OrderBook, direction: TradeDirection) -> Self {
        let cur_time = book1.timestamp.max(book2.timestamp);
        let (_ask, _bid, mid1) = OrderBookValues::calculate_ask_bid_mid(book1);
        let (_ask, _bid, mid2) = OrderBookValues::calculate_ask_bid_mid(book2);
        //let (open_price, close_price) = match direction {
        //    TradeDirection::Buy => (ask, bid),
        //    TradeDirection::Sell => (bid, ask),
        //};
        Self {
            direction,
            open_time: cur_time,
            open_price1: mid1,
            open_price2: mid2,
            trend: 0.0,
            max_trend: 0.0,
            prev_price_time: cur_time,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct PriceHistory {
    instrument_id: i32,
    time: DateTime<Utc>,
    ask: f32,
    bid: f32,
}

#[derive(Clone)]
pub struct WalkForwardModel {
    first_leg_id: i32,
    second_leg_id: i32,
    config: ModelConfig,
    booster_config: Arc<RwLock<Option<Value>>>,
    booster: Arc<RwLock<Option<GradientBoost>>>,
    linear_signal: Arc<RwLock<Option<LinearSignal>>>,
    features_buffer: Arc<Mutex<Vec<Vec<f32>>>>,
    prices_buffer: Arc<Mutex<Vec<PriceHistory>>>,
}

impl WalkForwardModel {
    pub fn new(first_leg_id: i32, second_leg_id: i32, config: ModelConfig) -> Self {
        Self {
            first_leg_id,
            second_leg_id,
            config,
            booster_config: Arc::new(RwLock::new(None)),
            booster: Arc::new(RwLock::new(None)),
            linear_signal: Arc::new(RwLock::new(None)),
            features_buffer: Arc::new(Mutex::new(Vec::new())),
            prices_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn is_it_time_to_close_deal(&self, book1: &OrderBook, book2: &OrderBook, deal_data: &mut DealData) -> bool {
        let cur_time = book1.timestamp.max(book2.timestamp);
        cur_time > deal_data.open_time + self.config.hold_time
    }

    pub fn process(&self, window_data: &mut WindowData, event: &OrderBookEvent) -> TradeSignal {
        let price;

        if self.first_leg_id == event.instrument_id {
            let book = OrderBookValues::from_order_book(&event.order_book);
            price = PriceHistory {
                instrument_id: event.instrument_id,
                time: event.order_book.timestamp,
                ask: book.ask as f32,
                bid: book.bid as f32,
            };
            window_data.last_features1 = Some(window_data.window1.process(book.to_features()));
        } else if self.second_leg_id == event.instrument_id {
            let book = OrderBookValues::from_order_book(&event.order_book);
            price = PriceHistory {
                instrument_id: event.instrument_id,
                time: event.order_book.timestamp,
                ask: book.ask as f32,
                bid: book.bid as f32,
            };
            window_data.last_features2 = Some(window_data.window2.process(book.to_features()));
        } else {
            return TradeSignal::None;
        };

        let mut prediction_gbm = None;
        let mut prediction_linear = None;

        if let Some(features1) = &window_data.last_features1
            && let Some(features2) = &window_data.last_features2 {
            let mut features = features1.clone();
            features.extend(features2.clone());
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

            if let Ok(read) = self.booster.read()
                && let Some(booster) = read.as_ref() {
                match booster.test(&[features]) {
                    Ok(predictions) => {
                        prediction_gbm = predictions.into_iter().next();
                    }
                    Err(err) => {
                        error!("Failed to make prediction: {err}");
                    }
                }
            }
        }

        if let Some(prediction_linear) = prediction_linear && let Some(prediction_gbm) = prediction_gbm {
            if window_data.last_predictions_linear.len() == LAST_PREDICTIONS_SIZE {
                let linear_threshold = percentile(&window_data.last_predictions_linear, self.config.prediction_percentile_linear);
                let gbm_threshold = percentile(&window_data.last_predictions_gbm, self.config.prediction_percentile_gbm);

                window_data.last_predictions_linear.remove(0);
                window_data.last_predictions_gbm.remove(0);
                window_data.last_predictions_linear.push(prediction_linear);
                window_data.last_predictions_gbm.push(prediction_gbm);

                if prediction_linear >= linear_threshold && prediction_gbm >= gbm_threshold {
                    return TradeSignal::Sell(prediction_linear, prediction_gbm)
                }
            } else {
                window_data.last_predictions_linear.push(prediction_linear);
                window_data.last_predictions_gbm.push(prediction_gbm);
            }
        }

        TradeSignal::None
    }

    fn get_price_on_horizon(&self, prices: &[PriceHistory], index: usize, instrument_id: i32) -> Option<(f32, f32)> {
        let mut i = index + 1;
        let target_time = prices[index].time + self.config.hold_time;
        while i < prices.len() {
            if prices[i].instrument_id == instrument_id
                && prices[i].time >= target_time {
                return Some((prices[i].ask, prices[i].bid));
            }
            i += 1;
        }

        None
    }

    fn get_horizons(&self, direction: TradeDirection) -> (Vec<f32>, usize) {
        let mut skipped = 0;
        self.prices_buffer.lock()
            .map(|prices| {
                if prices.len() < self.config.train_min_size {
                    (Vec::new(), 0)
                } else {
                    let (mut price1, mut price2) = (None, None);
                    let mut horizons = Vec::with_capacity(prices.len());
                    
                    for (i, price) in prices.iter().enumerate() {
                        if price.instrument_id == self.first_leg_id {
                            price1 = Some(price.bid);
                        } else {
                            price2 = Some(price.ask);
                        }

                        if let Some(price1) = price1 && let Some(price2) = price2 {
                            if let Some((h_ask1, h_bid1)) = self.get_price_on_horizon(&prices, i, self.first_leg_id)
                                && let Some((h_ask2, h_bid2)) = self.get_price_on_horizon(&prices, i, self.second_leg_id) {
                                let pnl = price1 - h_ask1 + h_bid2 - price2;
                                horizons.push(pnl);
                            }
                        } else {
                            skipped += 1;
                        }
                    }
                    (horizons, skipped)
                } 
            }).unwrap_or_default()
    }

    pub fn train(&self) -> bool {
        self.shrink_buffer();

        let mut booster = None;
        let mut linear = None;

        let (horizons, skipped) = self.get_horizons(TradeDirection::Buy);
        info!("Skipped {} first events", skipped);
        //let train_size = horizons.len() * 4 / 5;
        //let (horizons_train, horizons_test) = (&horizons[0..train_size], &horizons[train_size..]);
        if horizons.len() >= self.config.train_min_size {
            let features = self.features_buffer.lock()
                .map(|buf| buf[skipped..skipped + horizons.len()].to_vec());

            let config = self.booster_config.read().map(|cfg| cfg.clone());
            if let Ok(Some(config)) = &config && let Ok(features_train) = features {

                linear = calibrate_linear_signal(None, &features_train, &horizons);

                info!("Start training on {} events", features_train.len());
                match GradientBoost::train(
                    PredictionType::ClassificationPositive(self.config.expected_delta),
                    config,
                    DataVec::Owned(features_train),
                    &horizons,
                ) {
                    Ok(boost) => {
                        booster = Some(boost);
                        info!("New booster trained");
                    }
                    Err(err) => {
                        error!("Failed to train: {err}");
                    }
                }
            }
        }

        if let Some(linear) = linear && let Ok(mut write) = self.linear_signal.write() {
            *write = Some(linear);
        }

        if let Some(booster) = booster
            && let Ok(mut write) = self.booster.write() {
            *write = Some(booster);
            true
        } else {
            false
        }
    }

    pub fn optimize_config(&self, config_json: Option<&str>) -> bool {
        if let Some(config_json) = config_json
            && let Ok(config) = serde_json::from_str(config_json)
            && let Ok(mut write) = self.booster_config.write() {
            *write = Some(config);
            info!("New predefined booster config saved");
            return true;
        }

        let (horizons, skipped) = self.get_horizons(TradeDirection::Buy);
        if horizons.len() >= self.config.train_min_size {
            let train_size = horizons.len() * 4 / 5;
            let features = self.features_buffer.lock()
                .map(|buf| {
                    (buf[skipped..skipped + train_size].to_vec(), buf[skipped + train_size..horizons.len()].to_vec())
                });
            if let Ok((train, test)) = features {
                match optimize_light_gbm(
                    PredictionType::ClassificationPositive(self.config.expected_delta),
                    &train,
                    &horizons[..train_size],
                    &test,
                    &horizons[train_size..],
                    self.config.optimize_iterations,
                ) {
                    Ok(config) => {
                        if let Ok(mut write) = self.booster_config.write() {
                            *write = Some(config);
                            info!("New booster config saved");
                            return true;
                        }
                    }
                    Err(err) => {
                        error!("Failed to optimize: {err}");
                    }
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