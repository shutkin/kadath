use crate::executor::{KadathExecutor, LastPrices};
use crate::queue::Queue;
use abstract_broker::BrokerBox;
use chrono::{TimeDelta, Utc};
use db::{Db, QueryAsksOrBids};
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
    instruments: Vec<Instrument>,
    model: WalkForwardModel,
    db: Db,
    executor: KadathExecutor,
    last_prices: Arc<Mutex<LastPrices>>,
}

impl KadathTrader {
    pub fn new(
        broker: BrokerBox,
        queue: Arc<Queue<OrderBookEvent>>,
        instruments: Vec<Instrument>,
        db: Db,
        trades: Arc<Queue<Trade>>,
    ) -> Self {
        let config = ModelConfig { std_diapason_s: 1211, train_data_minutes: 42, train_interval_minutes: 1, decay_time: 12.406388913778208, fixed_hold_time: None };
        let model = WalkForwardModel::new_with_config(instruments[0].id.unwrap(), instruments[1].id.unwrap(), config);
        
        let last_prices = Arc::new(Mutex::new(LastPrices::default()));
        let executor = KadathExecutor::new(broker, &instruments, trades, last_prices.clone(), db.clone());
        
        Self {
            queue,
            instruments,
            model,
            db,
            executor,
            last_prices,
        }
    }
    
    #[inline]
    fn inst1_id(&self) -> i32 {
        self.instruments[0].id.unwrap_or_default()
    }

    #[inline]
    fn inst2_id(&self) -> i32 {
        self.instruments[1].id.unwrap_or_default()
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
                if event.instrument_id == self.inst1_id() {
                    last_prices.leg1_time = event.order_book.timestamp;
                    last_prices.leg1_ask = lowest_price(&event.order_book.asks);
                    last_prices.leg1_bid = highest_price(&event.order_book.bids);
                } else if event.instrument_id == self.inst2_id() {
                    last_prices.leg2_time = event.order_book.timestamp;
                    last_prices.leg2_ask = lowest_price(&event.order_book.asks);
                    last_prices.leg2_bid = highest_price(&event.order_book.bids);
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
                        TradeSignal::Sell1Buy2(hold) => {
                            info!("Sell1 Buy2 hold: {hold:?}");
                            if let Err(err) = self.executor.open_deal(TradeDirection::Sell, hold).await {
                                error!("error opening deal: {err}");
                            } else {
                                deals += 1;
                                info!("Deals: {deals}");
                            }
                        }
                        TradeSignal::Buy1Sell2(hold) => {
                            info!("Buy1 Sell2 hold: {hold:?}");
                            if let Err(err) = self.executor.open_deal(TradeDirection::Buy, hold).await {
                                error!("error opening deal: {err}");
                            } else {
                                deals += 1;
                                info!("Deals: {deals}");
                            }
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
            Utc::now() - TimeDelta::hours(8),
            Utc::now(),
        );
        let ob1 = self.db.get_order_books(Some(self.inst1_id()), diapason, QueryAsksOrBids::BOTH).await?;
        let ob2 = self.db.get_order_books(Some(self.inst2_id()), diapason, QueryAsksOrBids::BOTH).await?;
        let events = merge_events(&[self.inst1_id(), self.inst2_id()], &[ob1, ob2]);
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

                let up_map = resp.up.map(|calc| calc.to_params_map()).unwrap_or_default();
                let down_map = resp.down.map(|calc| calc.to_params_map()).unwrap_or_default();

                if !up_map.is_empty() || !down_map.is_empty() {
                    let calculator_log = SignalCalculatorLog {
                        trained_on: Utc::now(),
                        up_score: up_map.get("score").map(|s| *s as f32),
                        down_score: down_map.get("score").map(|s| *s as f32),
                        up_params: if up_map.is_empty() { None } else { serde_json::to_string(&up_map).ok() },
                        down_params: if down_map.is_empty() { None } else { serde_json::to_string(&down_map).ok() },
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