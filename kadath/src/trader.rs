use crate::executor::{KadathExecutor, LastPrices};
use crate::queue::Queue;
use abstract_broker::BrokerBox;
use chrono::Datelike;
use chrono::{TimeDelta, Utc};
use db::Db;
use log::{error, info, warn};
use model::common::{EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, merge_events};
use model::{Instrument, OrderData, Trade, TradeDirection};
use rust_decimal::Decimal;
use signal::{ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use std::sync::{Arc, Mutex};
use tokio_schedule::Job;

pub struct KadathTrader {
    queue: Arc<Queue<OrderBookEvent>>,
    leg1: Instrument,
    leg2: Instrument,
    model: WalkForwardModel,
    db: Db,
    executor: KadathExecutor,
    last_prices: Arc<Mutex<LastPrices>>,
}

impl KadathTrader {
    pub fn new(
        broker: BrokerBox,
        queue: Arc<Queue<OrderBookEvent>>,
        leg1: Instrument,
        leg2: Instrument,
        db: Db,
        trades: Arc<Queue<Trade>>,
    ) -> Self {
        let config = ModelConfig {
            derivatives_normalize_diapason_s: 57.28,
            trend_tau_s: 614.61,
            train_buf_size: 285000,
            train_min_size: 200000,
            optimize_iterations: 20,
            hold_time: TimeDelta::milliseconds(5000),
            expected_delta: 5.0,
            prediction_percentile: 0.99966,
        };
        let model = WalkForwardModel::new(leg1.id.unwrap(), leg2.id.unwrap(), config);
        
        let last_prices = Arc::new(Mutex::new(LastPrices::default()));
        let executor = KadathExecutor::new(broker, trades, last_prices.clone(), db.clone());
        
        Self {
            queue,
            leg1,
            leg2,
            model,
            db,
            executor,
            last_prices,
        }
    }
    
    #[inline]
    fn leg1_id(&self) -> i32 {
        self.leg1.id.unwrap()
    }

    #[inline]
    fn leg2_id(&self) -> i32 {
        self.leg2.id.unwrap()
    }

    pub async fn run(&self) -> EmptyResult {
        if let Err(err) = self.run_optimization() {
            error!("Failed to run calibration task: {err}");
        }

        let mut window_data = WindowData::new(self.model.get_config());
        self.consume_history(&mut window_data).await?;
        //self.model.optimize_config();

        let mut deals = 0;

        let mut do_not_trade_until = None;

        loop {
            let event = self.queue.pop().await;
            if matches!(event.order_book.timestamp.weekday().number_from_monday(), 6 | 7) // ignore weekends
                || (event.instrument_id != self.leg1_id() && event.instrument_id != self.leg2_id()) {
                continue;
            }

            let signal = self.model.process(&mut window_data, &event);

            let mut last_prices_copy = None;
            if let Ok(last_prices) = self.last_prices.lock().as_mut() {
                last_prices.time = event.order_book.timestamp;
                last_prices.ask = lowest_price(&event.order_book.asks);
                last_prices.bid = highest_price(&event.order_book.bids);
                last_prices.signal = signal;
                last_prices_copy = Some(**last_prices);
            }

            if !self.executor.has_active_deal() {
                let can_trade = if let Some(do_not_trade_until) = do_not_trade_until {
                    event.order_book.timestamp > do_not_trade_until
                } else {
                    true
                };
                if can_trade {
                    do_not_trade_until = None;
                    match signal {
                        TradeSignal::None => {}
                        TradeSignal::Buy(hold) => {
                            if let Err(err) = self.executor.open_deal(self.leg1.clone(), TradeDirection::Buy, hold).await {
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
                }
            } else if let Some(last_prices) = last_prices_copy 
                && let Some(delta) = self.executor.try_close_deal(last_prices).await && delta.is_sign_negative() {
                    let time = event.order_book.timestamp + TimeDelta::minutes(4);
                    do_not_trade_until = Some(time);
                info!("Deal close with delta {delta:?}, stop trading until {time}");
            }
        }
    }

    async fn consume_history(&self, window_data: &mut WindowData) -> EmptyResult {
        let diapason = TimeDiapason::new(
            Utc::now() - TimeDelta::hours(5),
            Utc::now(),
        );
        let books1 = self.db.get_order_books(self.leg1_id(), diapason).await?;
        let books2 = self.db.get_order_books(self.leg2_id(), diapason).await?;
        let events = merge_events(&[self.leg1_id(), self.leg2_id()], &[&books1, &books2]);
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

    fn run_optimization(&self) -> EmptyResult {
        let model = self.model.clone();
        let calibrate_task = tokio_schedule::every(1).hour().perform(move || {
            let model_clone = model.clone();
            async move {
                model_clone.optimize_config(None);
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
