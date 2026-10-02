use crate::queue::Queue;
use abstract_broker::BrokerBox;
use db::Db;
use log::{error, info, warn};
use model::common::EmptyResult;
use model::{Deal, DealStatus, Instrument, Trade, TradeDirection};
use rust_decimal::Decimal;
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct LastPrices {
    pub leg1_ask: Decimal,
    pub leg1_bid: Decimal,
    pub leg2_ask: Decimal,
    pub leg2_bid: Decimal,
}

#[derive(Clone)]
pub struct KadathExecutor {
    instruments: Arc<Vec<Instrument>>,
    broker: BrokerBox,
    cur_deal: Arc<Mutex<Option<Deal>>>,
    last_prices: Arc<Mutex<LastPrices>>,
    db: Db,
}

impl KadathExecutor {
    pub fn new(broker: BrokerBox, instruments: &[Instrument], trades: Arc<Queue<Trade>>, last_prices: Arc<Mutex<LastPrices>>, db: Db) -> Self {
        let cur_deal = Arc::new(Mutex::new(Option::<Deal>::None));

        let cur_deal_clone = cur_deal.clone();
        let db_clone = db.clone();
        tokio::spawn(async move {
            loop {
                let trade = trades.pop().await;
                info!("{trade:?}");
                if let Some(mut deal) = if let Ok(lock) = cur_deal_clone.lock() {
                    lock.clone()
                } else {
                    None
                } {
                    if let Some(order_id) = &deal.leg1_open_order_id && order_id == &trade.order_id {
                        deal.leg1_status = if trade.quantity > 0 {DealStatus::OpenConfirmed} else {DealStatus::OpenFailed}
                    } else if let Some(order_id) = &deal.leg2_open_order_id && order_id == &trade.order_id {
                        deal.leg2_status = if trade.quantity > 0 {DealStatus::OpenConfirmed} else {DealStatus::OpenFailed}
                    } else if let Some(order_id) = &deal.leg1_close_order_id && order_id == &trade.order_id {
                        deal.leg1_status = if trade.quantity > 0 {DealStatus::CloseConfirmed} else {DealStatus::CloseFailed}
                    } else if let Some(order_id) = &deal.leg2_close_order_id && order_id == &trade.order_id {
                        deal.leg2_status = if trade.quantity > 0 {DealStatus::CloseConfirmed} else {DealStatus::CloseFailed}
                    } else {
                        warn!("Unknown order id: {}", trade.order_id);
                    }

                    if let Err(err) = db_clone.update_deal(&deal).await {
                        error!("Error updating deal: {err}");
                    }

                    if let Ok(mut lock) = cur_deal_clone.lock() {
                        if deal.leg1_status == DealStatus::CloseConfirmed && deal.leg2_status == DealStatus::CloseConfirmed {
                            *lock = None;
                        } else {
                            *lock = Some(deal);
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
            instruments: Arc::new(instruments.to_vec()),
            db,
            last_prices,
        }
    }

    pub fn has_active_deal(&self) -> bool {
        if let Ok(cur_deal) = self.cur_deal.lock() {
            cur_deal.is_some()
        } else {
            false
        }
    }

    pub async fn open_deal(&self, direction: TradeDirection, hold: u16) -> EmptyResult {
        info!("Opening deal: {direction:?} with hold {hold}");

        let last_prices = self.last_prices.lock()
            .map_err(|err| format!("Failed to lock last prices: {err:?}"))?;
        let (leg1_price, leg2_price) = match direction {
                TradeDirection::Buy => (last_prices.leg1_ask, last_prices.leg2_bid),
                TradeDirection::Sell => (last_prices.leg1_bid, last_prices.leg2_ask),
        };
        drop(last_prices);

        let deal = self.db.start_deal(
            self.instruments[0].id.unwrap_or_default(),
            self.instruments[1].id.unwrap_or_default(),
            direction,
            hold,
            leg1_price,
            leg2_price,
        ).await?;

        let deal = self.make_orders(deal, true).await;
        self.db.update_deal(&deal).await?;

        if let Ok(cur_deal) = self.cur_deal.lock().as_mut() {
            cur_deal.replace(deal);
        }

        let self_clone = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(hold as u64)).await;
            self_clone.close_deal().await;
        });

        Ok(())
    }

    async fn close_deal(&self) {
        let deal = if let Ok(lock) = self.cur_deal.lock() {
            lock.clone()
        } else {
            None
        };

        if let Some(mut deal) = deal {
            info!("Closing deal");
            if let Ok(last_prices) = self.last_prices.lock() {
                let (leg1_price, leg2_price) = match deal.direction {
                    TradeDirection::Buy => (last_prices.leg1_bid, last_prices.leg2_ask),
                    TradeDirection::Sell => (last_prices.leg1_ask, last_prices.leg2_bid),
                };
                deal.leg1_close_price_expected = Some(leg1_price);
                deal.leg2_close_price_expected = Some(leg2_price);
            }

            let deal = self.make_orders(deal, false).await;
            if let Err(err) = self.db.update_deal(&deal).await {
                error!("Error updating deal: {err:?}");
            }
        } else {
            info!("Deal already closed");
        }
    }

    async fn make_orders(&self, deal: Deal, is_opening: bool) -> Deal {
        let direction = if is_opening {deal.direction} else {deal.direction.reverse()};
        let original_deal = deal.clone();
        let deal_mutex = Arc::new(Mutex::new(deal));

        let h1 = if is_opening || original_deal.leg1_status == DealStatus::OpenConfirmed {
            let broker = self.broker.clone();
            let instrument = self.instruments[0].clone();
            let deal_clone = deal_mutex.clone();
            Some(tokio::spawn(async move {
                match broker.make_order(&instrument, direction, 1).await {
                    Ok(order_id) => {
                        if let Ok(deal) = deal_clone.lock().as_mut() {
                            if is_opening {
                                deal.leg1_open_order_id = Some(order_id);
                            } else {
                                deal.leg1_close_order_id = Some(order_id);
                            }
                        }
                    }
                    Err(err) => {
                        if let Ok(deal) = deal_clone.lock().as_mut() {
                            deal.leg1_status = if is_opening { DealStatus::OpenFailed } else { DealStatus::CloseFailed };
                        }
                        error!("Failed to create leg1 order: {err}");
                    }
                }
            }))
        } else {
            None
        };

        let h2 = if is_opening || original_deal.leg2_status == DealStatus::OpenConfirmed {
            let broker = self.broker.clone();
            let instrument = self.instruments[1].clone();
            let deal_clone = deal_mutex.clone();
            Some(tokio::spawn(async move {
                match broker.make_order(&instrument, direction.reverse(), 1).await {
                    Ok(order_id) => {
                        if let Ok(deal) = deal_clone.lock().as_mut() {
                            deal.leg2_open_order_id = Some(order_id);
                        }
                    }
                    Err(err) => {
                        if let Ok(deal) = deal_clone.lock().as_mut() {
                            deal.leg2_status = if is_opening { DealStatus::OpenFailed } else { DealStatus::CloseFailed };
                        }
                        error!("Failed to create leg2 order: {err}");
                    }
                }
            }))
        } else {
            None
        };

        if let Some(h1) = h1 {
            if let Err(err) = h1.await {
                if let Ok(deal) = deal_mutex.lock().as_mut() {
                    deal.leg1_status = if is_opening { DealStatus::OpenFailed } else { DealStatus::CloseFailed };
                    error!("Failed to get response on leg1 order: {err}");
                }
            }
        }

        if let Some(h2) = h2 {
            if let Err(err) = h2.await {
                if let Ok(deal) = deal_mutex.lock().as_mut() {
                    deal.leg2_status = if is_opening { DealStatus::OpenFailed } else { DealStatus::CloseFailed };
                    error!("Failed to get response on leg2 order: {err}");
                }
            }
        }

        if let Ok(deal) = deal_mutex.lock() {
            deal.clone()
        } else {
            error!("Failed to lock result deal");
            original_deal
        }
    }
}