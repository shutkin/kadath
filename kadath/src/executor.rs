use crate::queue::Queue;
use abstract_broker::BrokerBox;
use chrono::{DateTime, TimeDelta, Utc};
use db::Db;
use log::{error, info, warn};
use model::common::EmptyResult;
use model::{Deal, Instrument, OrderStatus, Trade, TradeDirection, TradeStatus};
use rust_decimal::Decimal;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use signal::TradeSignal;

#[derive(Copy, Clone, Default)]
pub struct LastPrices {
    pub time: DateTime<Utc>,
    pub ask: Decimal,
    pub bid: Decimal,
    pub signal: TradeSignal,
}

#[derive(Clone)]
pub struct KadathExecutor {
    instrument: Arc<Instrument>,
    broker: BrokerBox,
    cur_deal: Arc<Mutex<Option<Deal>>>,
    last_prices: Arc<Mutex<LastPrices>>,
    db: Db,
}

impl KadathExecutor {
    pub fn new(broker: BrokerBox, instrument: Instrument, trades: Arc<Queue<Trade>>, last_prices: Arc<Mutex<LastPrices>>, db: Db) -> Self {
        let cur_deal = Arc::new(Mutex::new(Option::<Deal>::None));

        let cur_deal_clone = cur_deal.clone();
        let db_clone = db.clone();
        tokio::spawn(async move {
            loop {
                let trade = trades.pop().await;
                if let Some(mut deal) = if let Ok(lock) = cur_deal_clone.lock() {
                    lock.clone()
                } else {
                    None
                } {
                    if let Some(order_id) = &deal.open_order_id && order_id == &trade.order_id {
                        info!("Deal open status {:?}", trade.status);
                        deal.open_status = if trade.status == TradeStatus::Executed {OrderStatus::Executed} else {OrderStatus::Failed}
                    } else if let Some(order_id) = &deal.close_order_id && order_id == &trade.order_id {
                        info!("Deal close status {:?}", trade.status);
                        deal.close_status = if trade.status == TradeStatus::Executed {OrderStatus::Executed} else {OrderStatus::Failed}
                    }

                    if let Err(err) = db_clone.update_deal(&deal).await {
                        error!("Error updating deal: {err}");
                    }

                    if let Ok(mut lock) = cur_deal_clone.lock() {
                        if deal.open_status == OrderStatus::Executed && deal.close_status == OrderStatus::Executed {
                            info!("Deal confirmed and closed");
                            *lock = None;
                        } else {
                            *lock = Some(deal.clone());
                        }
                    }
                } else {
                    warn!("Trade without order: {trade:?}");
                }
            }
        });

        Self {
            broker,
            cur_deal,
            instrument: Arc::new(instrument),
            db,
            last_prices,
        }
    }

    pub fn has_active_deal(&self) -> bool {
        let (is_some, is_expired) = if let Ok(cur_deal) = self.cur_deal.lock() {
            let is_expired = cur_deal.as_ref().map(|deal| {
                deal.open_time < Utc::now() - TimeDelta::seconds(60)
            }).unwrap_or(false);
            (cur_deal.is_some(), is_expired)
        } else {
            (false, false)
        };
        if is_expired {
            if let Ok(mut lock) = self.cur_deal.lock() {
                warn!("Deal expire!");
                *lock = None;
                return false;
            }
        }
        is_some
    }

    pub async fn open_deal(&self, direction: TradeDirection, hold: i32) -> EmptyResult {
        info!("Opening deal: {direction:?} with hold {hold}");

        let last_prices = self.last_prices.lock()
            .map_err(|err| format!("Failed to lock last prices: {err:?}"))?;
        let price = match direction {
                TradeDirection::Buy => last_prices.ask,
                TradeDirection::Sell => last_prices.bid,
        };
        drop(last_prices);

        let deal = self.db.start_deal(
            self.instrument.id.unwrap_or_default(),
            1,
            direction,
            hold,
            price,
        ).await?;
        if let Ok(cur_deal) = self.cur_deal.lock().as_mut() {
            cur_deal.replace(deal.clone());
        }

        /*let deal = self.make_orders(deal, true).await;
        if let Ok(cur_deal) = self.cur_deal.lock().as_mut() {
            cur_deal.replace(deal.clone());
        }
        self.db.update_deal(&deal).await?;*/

        Ok(())
    }

    pub async fn try_close_deal(&self, last_prices: LastPrices) {
        let deal = if let Ok(lock) = self.cur_deal.lock() {
            lock.clone()
        } else {
            None
        };

        if let Some(mut deal) = deal {
            let price = match deal.direction {
                TradeDirection::Buy => last_prices.bid,
                TradeDirection::Sell => last_prices.ask,
            };

            let mut close = false;
            
            match deal.direction {
                TradeDirection::Buy => {
                    if price < deal.expected_open_price - Decimal::ONE && last_prices.time > deal.open_time + TimeDelta::milliseconds(300) {
                        close = true;
                    } else {
                        if last_prices.time > deal.open_time + TimeDelta::milliseconds(deal.max_hold as i64)
                            || price > deal.expected_open_price + Decimal::TWO {
                            close = true;
                        }
                    }
                }
                TradeDirection::Sell => {
                    if price > deal.expected_open_price + Decimal::ONE && last_prices.time > deal.open_time + TimeDelta::milliseconds(300) {
                        close = true;
                    } else {
                        if last_prices.time > deal.open_time + TimeDelta::milliseconds(deal.max_hold as i64)
                            || price < deal.expected_open_price - Decimal::TWO {
                            close = true;
                        }
                    }
                }
            }

            if close {
                info!("Closing deal");
                deal.expected_close_price = Some(price);
                deal.close_time = Some(Utc::now());

                //let deal = self.make_orders(deal, false).await;
                if let Ok(mut cur_deal) = self.cur_deal.lock() {
                    //cur_deal.replace(deal.clone());
                    *cur_deal = None;
                }
                if let Err(err) = self.db.update_deal(&deal).await {
                    error!("Error updating deal: {err:?}");
                }
            }
        } else {
            info!("Deal already closed");
        }
    }

    async fn make_orders(&self, mut deal: Deal, is_opening: bool) -> Deal {
        let direction = if is_opening {deal.direction} else {deal.direction.reverse()};

        let id = Uuid::new_v4().to_string();
        if is_opening {
            deal.open_order_id = Some(id.clone());
        } else {
            deal.close_order_id = Some(id.clone());
        }
        if let Ok(cur_deal) = self.cur_deal.lock().as_mut() {
            cur_deal.replace(deal.clone());
        }
        if let Err(err) = self.db.update_deal(&deal).await {
            error!("Error updating deal on make orders: {err:?}");
        }

        if is_opening && deal.open_status == OrderStatus::NotStarted
            || !is_opening && deal.close_status == OrderStatus::NotStarted {
            match self.broker.make_order(&id, &self.instrument, direction, 1).await {
                Ok(order_id) => {
                    if is_opening {
                        deal.open_status = OrderStatus::Started;
                        deal.open_order_id = Some(order_id);
                    } else {
                        deal.close_status = OrderStatus::Started;
                        deal.close_order_id = Some(order_id);
                    }
                }
                Err(err) => {
                    if is_opening {
                        deal.open_status = OrderStatus::Failed;
                    } else {
                        deal.close_status = OrderStatus::Failed;
                    }
                    error!("Failed to create leg1 order: {err}");
                }
            }
        } else {
            warn!("Order already sent");
        };
        deal
    }
}