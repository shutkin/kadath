use crate::executor::{KadathExecutor, LastPrices};
use crate::queue::Queue;
use abstract_broker::BrokerBox;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{error, info, warn};
use model::common::{EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, merge_events};
use model::{Instrument, OrderBook, OrderData, SignalCalculatorLog, Trade, TradeDirection};
use rust_decimal::Decimal;
use signal::{ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use std::sync::{Arc, Mutex};
use tokio_schedule::Job;
use serde::Serialize;

pub struct KadathTrader {
    queue: Arc<Queue<OrderBookEvent>>,
    instrument: Instrument,
    model: WalkForwardModel,
    db: Db,
    executor: KadathExecutor,
    last_prices: Arc<Mutex<LastPrices>>,
}

impl KadathTrader {
    pub fn new(
        broker: BrokerBox,
        queue: Arc<Queue<OrderBookEvent>>,
        instrument: Instrument,
        db: Db,
        trades: Arc<Queue<Trade>>,
    ) -> Self {
        let config = ModelConfig {
            hold_time_ms: 25000,
            derivatives_normalize_diapason_s: 325.0,
            trend_tau_s: 176.0,
            train_data_minutes: 45,
            train_interval_minutes: 1,
            decay_time: 360.0,
        };
        let model = WalkForwardModel::new_buying(instrument.id.unwrap(), config);
        
        let last_prices = Arc::new(Mutex::new(LastPrices::default()));
        let executor = KadathExecutor::new(broker, instrument.clone(), trades, last_prices.clone(), db.clone());
        
        Self {
            queue,
            instrument,
            model,
            db,
            executor,
            last_prices,
        }
    }
    
    #[inline]
    fn inst_id(&self) -> i32 {
        self.instrument.id.unwrap_or_default()
    }

    pub async fn run(&self) -> EmptyResult {
        if let Err(err) = self.run_calibration() {
            error!("Failed to run calibration task: {err}");
        }

        let mut window_data = WindowData::new();
        self.consume_history(&mut window_data).await?;
        self.model.calibrate(false);

        let mut online_events = 0;
        let mut deals = 0;

        loop {
            let event = self.queue.pop().await;

            let mut last_prices_copy = None;
            if let Ok(last_prices) = self.last_prices.lock().as_mut() {
                if event.instrument_id == self.inst_id() {
                    last_prices.time = event.order_book.timestamp;
                    last_prices.ask = lowest_price(&event.order_book.asks);
                    last_prices.bid = highest_price(&event.order_book.bids);
                }
                last_prices_copy = Some(last_prices.clone());
            }

            let signal = self.model.process(&mut window_data, &event);
            if online_events < 1024 {
                online_events += 1;
            } else {
                if online_events == 1024 {
                    info!("Start trading");
                    online_events += 1;
                }
                if !self.executor.has_active_deal() {
                    match signal {
                        TradeSignal::None => {}
                        TradeSignal::Buy(hold) => {
                            info!("Sell hold: {hold:?}");
                            if let Err(err) = self.executor.open_deal(TradeDirection::Sell, hold.max(35000)).await {
                                error!("error opening deal: {err}");
                            } else {
                                deals += 1;
                                info!("Deals: {deals}");
                            }
                        }
                        TradeSignal::Sell(hold) => {
                            info!("Sell hold: {hold:?}");
                        }
                    }
                } else if let Some(last_prices) = last_prices_copy {
                    self.executor.try_close_deal(last_prices).await;
                }
            }
        }

        /*if let Some(start_time) = start_time && let Some(end_time) = end_time {
            let diapason = TimeDiapason::new(start_time - TimeDelta::milliseconds(1), end_time - TimeDelta::milliseconds(1));
            let db_order_books1 = self.db.get_order_books_compact(self.inst1_id(), diapason).await?;
            let db_order_books2 = self.db.get_order_books_compact(self.inst2_id(), diapason).await?;

            if let Err(err) = compare_order_books(&order_books1, &db_order_books1, "order_books1_diffs.json") {
                error!("Failed to compare order books: {err}");
            }

            if let Err(err) = compare_order_books(&order_books2, &db_order_books2, "order_books2_diffs.json") {
                error!("Failed to compare order books: {err}");
            }
        }

        Ok(())*/
    }

    async fn consume_history(&self, window_data: &mut WindowData) -> EmptyResult {
        let diapason = TimeDiapason::new(
            Utc::now() - TimeDelta::hours(1),
            Utc::now(),
        );
        let ob = self.db.get_order_books(self.inst_id(), diapason).await?;
        let events = merge_events(&[self.inst_id()], &[ob]);
        info!("Got {} history events", events.len());
        for event in events {
            if !event.order_book.asks.is_empty() && !event.order_book.bids.is_empty() {
                self.model.process(window_data, &event);
            } else {
                warn!("Order book {} is empty", event.order_book.id.unwrap_or_default());
            }
        }
        info!("History events consumed");

        Ok(())
    }

    fn run_calibration(&self) -> EmptyResult {
        let model = self.model.clone();
        let db = self.db.clone();
        let calibrate_task = tokio_schedule::every(1).minute().perform(move || {
            let model_clone = model.clone();
            let db_clone = db.clone();
            async move {
                let resp = model_clone.calibrate(false);

                let down_map = resp.calculator.map(|calc| calc.to_params_map()).unwrap_or_default();

                if !down_map.is_empty() {
                    let calculator_log = SignalCalculatorLog {
                        trained_on: Utc::now(),
                        up_score: None,
                        down_score: down_map.get("score").map(|s| *s as f32),
                        up_params: None,
                        down_params: if down_map.is_empty() { None } else { serde_json::to_string(&down_map).ok() },
                        train_time_ms: 0,
                    };
                    if let Err(err) = db_clone.insert_calculator_log(&calculator_log).await {
                        error!("Failed to insert calculator log: {err}");
                    }
                }
            }
        });
        tokio::spawn(calibrate_task);
        Ok(())
    }
}

fn highest_price(data: &[OrderData]) -> Decimal {
    let max = data.iter()
        .map(|data| data.price)
        .max();
    if max.is_none() {
        warn!("Can't find highest price: {data:?}");
    }
    max.unwrap_or_default()
}

fn lowest_price(data: &[OrderData]) -> Decimal {
    let min = data.iter()
        .map(|data| data.price)
        .min();
    if min.is_none() {
        warn!("Can't find lowest price: {data:?}");
    }
    min.unwrap_or_default()
}

fn compare_order_books(obs1: &[OrderBook], obs2: &[OrderBook], filename: &str) -> EmptyResult {
    let mut diffs = Vec::new();
    for (i, ob1) in obs1.iter().enumerate() {
        let j_from = ((i as i32) - 100).max(0) as usize;
        let j_to = ((i as i32) + 100).min(obs2.len() as i32 - 1) as usize;
        let mut nearest_distance = 5000;
        let mut nearest_j = i;
        for j in j_from..j_to {
            let distance = (ob1.timestamp.timestamp_millis() - obs1[j].timestamp.timestamp_millis()).abs();
            if distance < nearest_distance {
                nearest_distance = distance;
                nearest_j = j;
            }
        }
        let asks_diff = compare_data(&ob1.asks, &obs2[nearest_j].asks);
        let bids_diff = compare_data(&ob1.asks, &obs2[nearest_j].asks);
        if !asks_diff.is_empty() || !bids_diff.is_empty() {
            diffs.push(OrderBookDiff {
                time1: ob1.timestamp,
                time2: obs2[nearest_j].timestamp,
                asks: asks_diff,
                bids: bids_diff,
            })
        }
    }
    let json = serde_json::to_string(&diffs)?;
    std::fs::write(filename, json)?;

    Ok(())
}

fn compare_data(vec1: &[OrderData], vec2: &[OrderData]) -> Vec<OrderDataDiff> {
    let mut diffs = Vec::new();

    for d1 in vec1 {
        let d2 = vec2.iter().find(|d| d.price == d1.price);
        if let Some(d2) = d2 {
            if d1.size != d2.size {
                diffs.push(OrderDataDiff {
                    price: d1.price,
                    size1: d1.size,
                    size2: d2.size,
                });
            }
        } else {
            diffs.push(OrderDataDiff {
                price: d1.price,
                size1: d1.size,
                size2: 0,
            })
        }
    }

    for d2 in vec2 {
        if !vec1.iter().any(|d| d.price == d2.price) {
            diffs.push(OrderDataDiff {
                price: d2.price,
                size1: 0,
                size2: d2.size,
            })
        }
    }

    diffs
}

#[derive(Serialize)]
struct OrderBookDiff {
    time1: DateTime<Utc>,
    time2: DateTime<Utc>,
    asks: Vec<OrderDataDiff>,
    bids: Vec<OrderDataDiff>,
}

#[derive(Serialize)]
struct OrderDataDiff {
    price: Decimal,
    size1: i32,
    size2: i32,
}