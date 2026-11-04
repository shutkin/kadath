extern crate core;

pub mod orderbook_values;
mod gradient_boost;

use crate::gradient_boost::{DataVec, GradientBoost, PredictionType, optimize_light_gbm};
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use log::{error, info};
use model::events::OrderBookEvent;
use model::math_util::OutputType::{DerivativeNorm, Raw, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{FeaturesRollingWindow, p05_p95, percentile};
use model::{OrderBook, TradeDirection};
use serde_json::Value;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

const LAST_PREDICTIONS_INTERVAL: TimeDelta = TimeDelta::minutes(60);

#[derive(Copy, Clone, Default)]
pub enum TradeSignal {
    #[default]
    None,
    Sell(i32),
    Buy(i32),
}

#[derive(Copy, Clone, Debug)]
pub struct Regime {
    pub change: f64,
    pub er: f64,
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
    pub prediction_percentile: f64,
}

pub struct WindowData {
    last_book1: Option<OrderBookValues>,
    last_book2: Option<OrderBookValues>,
    window1: FeaturesRollingWindow,
    window2: FeaturesRollingWindow,
    last_predictions: Vec<f64>,
    last_predictions_times: Vec<DateTime<Utc>>,
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
            last_book1: None,
            last_book2: None,
            window1: window.clone(),
            window2: window.clone(),
            last_predictions: Vec::new(),
            last_predictions_times: Vec::new(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DealData {
    direction: TradeDirection,
    open_time: DateTime<Utc>,
    hold_to: DateTime<Utc>,
    open_price1: f64,
    open_price2: f64,

    trend: f64,
    max_trend: f64,
    prev_price_time: DateTime<Utc>,
}

impl DealData {
    pub fn new(book1: &OrderBook, book2: &OrderBook, direction: TradeDirection, hold_time_ms: i32) -> Self {
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
            hold_to: cur_time + TimeDelta::milliseconds(hold_time_ms as i64),
            open_price1: mid1,
            open_price2: mid2,
            trend: 0.0,
            max_trend: 0.0,
            prev_price_time: cur_time,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct HistoryPrices {
    time: DateTime<Utc>,
    price1: f32,
    price2: f32,
    price1_horizon: f32,
    price2_horizon: f32,
}

#[derive(Clone)]
pub struct WalkForwardModel {
    first_leg_id: i32,
    second_leg_id: i32,
    config: ModelConfig,
    booster_config: Arc<RwLock<Option<Value>>>,
    booster: Arc<RwLock<Option<GradientBoost>>>,
    features_buffer: Arc<Mutex<Vec<Vec<f32>>>>,
    prices_buffer: Arc<Mutex<Vec<HistoryPrices>>>,
}

impl WalkForwardModel {
    pub fn new(first_leg_id: i32, second_leg_id: i32, config: ModelConfig) -> Self {
        Self {
            first_leg_id,
            second_leg_id,
            config,
            booster_config: Arc::new(RwLock::new(None)),
            booster: Arc::new(RwLock::new(None)),
            features_buffer: Arc::new(Mutex::new(Vec::new())),
            prices_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn is_it_time_to_close_deal(&self, book1: &OrderBook, book2: &OrderBook, deal_data: &mut DealData) -> bool {
        let last_time = book1.timestamp.max(book2.timestamp);

        let (_ask, _bid, mid1) = OrderBookValues::calculate_ask_bid_mid(book1);
        let (_ask, _bid, mid2) = OrderBookValues::calculate_ask_bid_mid(book2);
        let cur_pnl = match deal_data.direction {
            TradeDirection::Buy => mid1 - deal_data.open_price1 + deal_data.open_price2 - mid2,
            TradeDirection::Sell => deal_data.open_price1 - mid1 + mid2 - deal_data.open_price2,
        };
        if cur_pnl < -0.5 && last_time > deal_data.open_time + TimeDelta::milliseconds(300) {
            return true;
        }

        let dt = (last_time - deal_data.prev_price_time).as_seconds_f64();
        let alpha = 1.0 - (-dt / 1.5).exp();
        deal_data.trend += alpha * (cur_pnl - deal_data.trend);

        if deal_data.max_trend < deal_data.trend {
            deal_data.max_trend = deal_data.trend;
        }

        deal_data.prev_price_time = last_time;

        if last_time > deal_data.hold_to && deal_data.trend < deal_data.max_trend - 0.5 {
            return true;
        }

        false
    }

    fn write_prices_to_history(&self, mut buffer: MutexGuard<Vec<HistoryPrices>>, prices: HistoryPrices) {
        let mut i = buffer.len() as i32 - 1;
        buffer.push(prices);
        let horizon_target = prices.time - self.config.hold_time;
        while i >= 0 {
            let prev_prices = &mut buffer[i as usize];
            if prev_prices.price1_horizon > 0.0 {
                break;
            }

            if prev_prices.time <= horizon_target {
                prev_prices.price1_horizon = prices.price1;
                prev_prices.price2_horizon = prices.price2;
            }

            i -= 1;
        }
    }

    pub fn process(&self, window_data: &mut WindowData, event: &OrderBookEvent) -> TradeSignal {
        if self.first_leg_id == event.instrument_id {
            window_data.last_book1 = Some(OrderBookValues::from_order_book(&event.order_book));
        } else if self.second_leg_id == event.instrument_id {
            window_data.last_book2 = Some(OrderBookValues::from_order_book(&event.order_book));
        } else {
            return TradeSignal::None;
        };

        let mut prediction = None;

        if let Some(book1) = &window_data.last_book1
            && let Some(book2) = &window_data.last_book2 {
            let mut features = window_data.window1.process(book1.to_features());
            features.extend(window_data.window2.process(book2.to_features()));

            if let Ok(mut lock) = self.features_buffer.lock() {
                lock.push(features.clone());
            }

            if let Ok(lock) = self.prices_buffer.lock() {
                let prices = HistoryPrices {
                    time: event.order_book.timestamp,
                    price1: book1.mid as f32,
                    price2: book2.mid as f32,
                    price1_horizon: 0.0,
                    price2_horizon: 0.0,
                };
                self.write_prices_to_history(lock, prices);
            }

            if let Ok(read) = self.booster.read()
                && let Some(booster) = read.as_ref() {
                match booster.test(&[features]) {
                    Ok(predictions) => {
                        prediction = predictions.into_iter().next();
                    }
                    Err(err) => {
                        error!("Failed to make prediction: {err}");
                    }
                }
            }
        }

        if let Some(prediction) = prediction {
            window_data.last_predictions.push(prediction);
            window_data.last_predictions_times.push(event.order_book.timestamp);

            let threshold = event.order_book.timestamp - LAST_PREDICTIONS_INTERVAL;
            while let Some(first_time) = window_data.last_predictions_times.first() && *first_time < threshold {
                window_data.last_predictions_times.remove(0);
                window_data.last_predictions.remove(0);
            }

            let p_threshold = percentile(&window_data.last_predictions, self.config.prediction_percentile);
            if prediction > p_threshold {
                return TradeSignal::Buy(self.config.hold_time.num_milliseconds() as i32)
            }
        }

        TradeSignal::None
    }

    fn get_horizons(&self, direction: TradeDirection) -> Vec<f32> {
        self.prices_buffer.lock()
            .map(|prices| {
                if prices.len() < self.config.train_min_size {
                    Vec::new()
                } else {
                    prices.iter()
                        .filter(|p| p.price1_horizon > 0.0 && p.price2_horizon > 0.0)
                        .map(|p| {
                            match direction {
                                TradeDirection::Buy => p.price1_horizon - p.price1 + p.price2 - p.price2_horizon,
                                TradeDirection::Sell => p.price1 - p.price1_horizon + p.price2_horizon - p.price2,
                            }
                        })
                        .collect::<Vec<_>>()
                }
            }).unwrap_or_default()
    }

    pub fn train(&self) -> bool {
        self.shrink_buffer();

        let mut booster = None;

        let horizons = self.get_horizons(TradeDirection::Buy);
        if horizons.len() >= self.config.train_min_size {
            let features = self.features_buffer.lock().map(|buf| buf[..horizons.len()].to_vec());
            let config = self.booster_config.read().map(|cfg| cfg.clone());
            if let Ok(Some(config)) = &config && let Ok(features) = features {
                info!("Start training on {} events", features.len());
                match GradientBoost::train(
                    PredictionType::ClassificationPositive(self.config.expected_delta),
                    config,
                    DataVec::Owned(features),
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

        let horizons = self.get_horizons(TradeDirection::Buy);
        if horizons.len() >= self.config.train_min_size {
            let train_size = horizons.len() * 4 / 5;
            let features = self.features_buffer.lock()
                .map(|buf| {
                    (buf[..train_size].to_vec(), buf[train_size..horizons.len()].to_vec())
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