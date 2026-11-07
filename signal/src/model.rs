extern crate core;

pub mod orderbook_values;
mod gradient_boost;

use crate::gradient_boost::{DataVec, GradientBoost, PredictionType, optimize_light_gbm};
use crate::orderbook_values::OrderBookValues;
use chrono::{DateTime, TimeDelta, Utc};
use log::{error, info};
use model::events::OrderBookEvent;
use model::math_util::OutputType::{DerivativeNorm, Raw, TrendDeviation, TrendDeviationDerivativeNorm};
use model::math_util::{FeaturesRollingWindow, percentile};
use model::{OrderBook, TradeDirection};
use serde_json::Value;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

const LAST_PREDICTIONS_INTERVAL: TimeDelta = TimeDelta::minutes(180);

#[derive(Copy, Clone, Default)]
pub enum TradeSignal {
    #[default]
    None,
    Sell(f64),
    Buy(f64),
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
    last_features1: Option<Vec<f32>>,
    last_features2: Option<Vec<f32>>,
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
            last_features1: None,
            last_features2: None,
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
    price: f32,
    price_horizon: f32,
}

#[derive(Clone)]
pub struct WalkForwardModel {
    first_leg_id: i32,
    second_leg_id: i32,
    config: ModelConfig,
    booster_config: Arc<RwLock<Option<Value>>>,
    booster: Arc<RwLock<Option<GradientBoost>>>,
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

    fn write_prices_to_history(&self, mut buffer: MutexGuard<Vec<PriceHistory>>, prices: PriceHistory) {
        let mut i = buffer.len() as i32 - 1;
        buffer.push(prices);
        /*let horizon_target = prices.time - self.config.hold_time;
        while i >= 0 {
            let prev_price = &mut buffer[i as usize];
            i -= 1;

            if prev_price.instrument_id != prices.instrument_id {
                continue;
            }

            if prev_price.price_horizon > 0.0 {
                break;
            }

            if prev_price.time <= horizon_target {
                prev_price.price_horizon = prices.price;
            }
        }*/
    }

    pub fn process(&self, window_data: &mut WindowData, event: &OrderBookEvent) -> TradeSignal {
        let price;

        if self.first_leg_id == event.instrument_id {
            let book = OrderBookValues::from_order_book(&event.order_book);
            price = PriceHistory {
                instrument_id: event.instrument_id,
                time: event.order_book.timestamp,
                price: book.mid as f32,
                price_horizon: 0.0,
            };
            window_data.last_features1 = Some(window_data.window1.process(book.to_features()));
            window_data.last_book1 = Some(book);
        } else if self.second_leg_id == event.instrument_id {
            let book = OrderBookValues::from_order_book(&event.order_book);
            price = PriceHistory {
                instrument_id: event.instrument_id,
                time: event.order_book.timestamp,
                price: book.mid as f32,
                price_horizon: 0.0,
            };
            window_data.last_features2 = Some(window_data.window2.process(book.to_features()));
            window_data.last_book2 = Some(book);
        } else {
            return TradeSignal::None;
        };

        let mut prediction = None;

        if let Some(book1) = &window_data.last_book1
            && let Some(book2) = &window_data.last_book2 {
            let mut features = window_data.last_features1.clone().unwrap();
            features.extend(window_data.last_features2.clone().unwrap());
            if let Ok(mut lock) = self.features_buffer.lock() {
                lock.push(features.clone());
            }
            if let Ok(lock) = self.prices_buffer.lock() {
                self.write_prices_to_history(lock, price);
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
                return TradeSignal::Buy(prediction)
            }
        }

        TradeSignal::None
    }

    fn get_price_on_horizon(&self, prices: &[PriceHistory], index: usize) -> Option<f32> {
        let mut i = index + 1;
        let target_time = prices[index].time + self.config.hold_time;
        let instrument_id = prices[index].instrument_id;
        while i < prices.len() {
            if prices[i].instrument_id == instrument_id
                && prices[i].time >= target_time {
                return Some(prices[i].price)
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
                    let (mut price1, mut price2, mut horizon1, mut horizon2) = (0.0, 0.0, 0.0, 0.0);
                    let mut horizons = Vec::with_capacity(prices.len());
                    
                    for (i, price) in prices.iter().enumerate() {
                        if let Some(horizon_price) = self.get_price_on_horizon(&prices, i) {
                            if prices[i].instrument_id == self.first_leg_id {
                                price1 = prices[i].price;
                                horizon1 = horizon_price;
                            } else {
                                price2 = prices[i].price;
                                horizon2 = horizon_price;
                            }
                        }

                        if price1 > 0.0 && price2 > 0.0 && horizon1 > 0.0 && horizon2 > 0.0 {
                            let horizon = match direction {
                                TradeDirection::Buy => horizon1 - price1 + price2 - horizon2,
                                TradeDirection::Sell => price1 - horizon1 + horizon2 - price2,
                            };
                            horizons.push(horizon);
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

        let (horizons, skipped) = self.get_horizons(TradeDirection::Buy);
        info!("Skipped {} first events", skipped);
        //let train_size = horizons.len() * 4 / 5;
        //let (horizons_train, horizons_test) = (&horizons[0..train_size], &horizons[train_size..]);
        if horizons.len() >= self.config.train_min_size {
            let features = self.features_buffer.lock()
                .map(|buf| buf[skipped..skipped + horizons.len()].to_vec());
            let config = self.booster_config.read().map(|cfg| cfg.clone());
            if let Ok(Some(config)) = &config && let Ok(features_train) = features {
                info!("Start training on {} events", features_train.len());
                match GradientBoost::train(
                    PredictionType::ClassificationPositive(self.config.expected_delta),
                    config,
                    DataVec::Owned(features_train),
                    &horizons,
                ) {
                    Ok(boost) => {
                        /*info!("Start testing on {} events", features_test.len());
                        let predictions = boost.test(&features_test).unwrap();
                        let mut predictions_ord = predictions.clone();
                        predictions_ord.sort_by(|p1, p2| p1.partial_cmp(p2).unwrap_or(Equal));

                        for percentile in [0.99, 0.995, 0.999, 0.9995, 0.99966, 0.9999] {
                            let index = ((predictions_ord.len() - 1) as f64 * percentile) as usize;
                            let threshold = predictions_ord[index];
                            let (mut correct, mut incorrect) = (0, 0);
                            let (mut commission, mut pnl) = (0.0, 0.0);
                            for (prediction, horizon) in predictions.iter().zip(horizons_test.iter()) {
                                if *prediction > threshold {
                                    if *horizon > self.config.expected_delta {
                                        correct += 1;
                                    } else {
                                        incorrect += 1;
                                    }

                                    pnl += horizon;
                                    commission += 5.0;
                                }
                            }
                            info!("Percentile {percentile} ({threshold}): {:.2}%, PnL {pnl}, commission {commission}, gross {}",
                                correct as f64 * 100.0 / (correct + incorrect) as f64, pnl - commission);
                        }*/
                        
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