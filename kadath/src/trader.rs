use crate::executor::{KadathExecutor, LastPrices};
use crate::queue::Queue;
use abstract_broker::BrokerBox;
use chrono::Datelike;
use chrono::{TimeDelta, Utc};
use db::Db;
use log::{error, info, warn};
use model::common::{EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, merge_events};
use model::{Instrument, OrderData, SignalCalculatorLog, Trade, TradeDirection};
use rust_decimal::Decimal;
use signal::{ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use std::sync::{Arc, Mutex};
use tokio_schedule::Job;

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
            derivatives_normalize_diapason_s: 188.6128974149429,
            trend_tau_s: 80.56655630108168,
            train_data_minutes: 120,
            train_interval_minutes: 1,
            decay_time: 308.29617113815294,
            hold_time_ms: 15000,
            price_change_filter: -0.0310822185621702,
            prediction_strength: 0.0021771551776655815,
            predicted_change: -0.018042337354087265,
            prediction_probability_up_to_flat: -1.0377714753698688,
            prediction_probability_up_to_down: 0.02836358508759033
        };
        let model = WalkForwardModel::new_buying(instrument.id.unwrap(), config);
        
        let last_prices = Arc::new(Mutex::new(LastPrices::default()));
        let executor = KadathExecutor::new(broker, trades, last_prices.clone(), db.clone());
        
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
            if matches!(event.order_book.timestamp.weekday().number_from_monday(), 6 | 7) // ignore weekends
                || event.instrument_id != self.inst_id() {
                continue;
            }

            let signal = self.model.process(&mut window_data, &event);

            let mut last_prices_copy = None;
            if let Ok(last_prices) = self.last_prices.lock().as_mut() {
                last_prices.time = event.order_book.timestamp;
                last_prices.ask = lowest_price(&event.order_book.asks);
                last_prices.bid = highest_price(&event.order_book.bids);
                last_prices.signal = signal;
                last_prices_copy = Some(last_prices.clone());
            }

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
                            if let Err(err) = self.executor.open_deal(self.instrument.clone(), TradeDirection::Buy, hold).await {
                                error!("error opening UP deal: {err}");
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
    }

    async fn consume_history(&self, window_data: &mut WindowData) -> EmptyResult {
        let diapason = TimeDiapason::new(
            Utc::now() - TimeDelta::hours(2),
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
