use crate::queue::Queue;
use abstract_broker::{AbstractBroker, BrokerBox};
use chrono::Utc;
use db::Db;
use log::{debug, error};
use model::common::EmptyResult;
use model::events::OrderBookEvent;
use model::{Instrument, Trade};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Clone)]
struct ListenerHandler {
    db: Db,
    instruments: HashMap<i32, Instrument>,
    ticker: Arc<AtomicI64>,
    order_book_queue: Arc<Queue<OrderBookEvent>>,
}

impl ListenerHandler {
    async fn on_order_book(&self, event: OrderBookEvent) {
        self.ticker.store(Utc::now().timestamp(), Relaxed);
        if let Some(instrument) = self.instruments.get(&event.instrument_id) {
            self.order_book_queue.push(event.clone());

            debug!("Save snapshot order book {:?}", event.order_book);
            if let Err(err) = self.db.insert_order_book_compact(&event.order_book, instrument).await {
                error!("Failed to save order book: {}", err);
            }
        }
    }
}

pub async fn listen(
    db: Db,
    instruments: Vec<Instrument>,
    order_book_queue: Arc<Queue<OrderBookEvent>>,
    trade_queue: Arc<Queue<Trade>>,
    broker: BrokerBox,
) -> EmptyResult {
    let mut instruments_map = HashMap::new();
    instruments.into_iter().for_each(|instrument| {
        if let Some(id) = instrument.id {
            instruments_map.insert(id, instrument);
        }
    });

    let db_clone = db.clone();
    let broker_clone = broker.clone();
    let trade_clone = trade_queue.clone();
    tokio::spawn(async move {
        loop {
            if let Err(err) = listen_trades(db_clone.clone(), broker_clone.clone(), trade_clone.clone()).await {
                error!("Error listening trades: {}", err);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    let ticker = Arc::new(AtomicI64::new(0));
    loop {
        let now = Utc::now().timestamp();
        if ticker.load(Relaxed) < now - 5 * 60 {
            start_listener(db.clone(), instruments_map.clone(), ticker.clone(), order_book_queue.clone(), broker.clone());
            ticker.store(Utc::now().timestamp(), Relaxed);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn listen_trades(db: Db, broker: Arc<Box<dyn AbstractBroker + Sync + Send>>, trade_queue: Arc<Queue<Trade>>) -> EmptyResult {
    let db_clone = db.clone();
    let (trades_sender, mut trades_receiver) = mpsc::channel(64);
    let h1 = tokio::spawn(async move {
        while let Some(mut trade) = trades_receiver.recv().await {
            if let Err(err) = db_clone.insert_trade(&mut trade).await {
                error!("Failed to insert trade: {err}");
            }
            trade_queue.push(trade);
        }
    });
    let h2 = tokio::spawn(async move {
        loop {
            if let Err(err) = broker.listen_trades(trades_sender.clone()).await {
                error!("Error listening trades: {}", err);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });
    h1.await?;
    h2.await?;
    Ok(())
}

fn start_listener(
    db: Db,
    instruments: HashMap<i32, Instrument>,
    ticker: Arc<AtomicI64>,
    order_book_queue: Arc<Queue<OrderBookEvent>>,
    broker: Arc<Box<dyn AbstractBroker + Sync + Send>>,
) {
    let (order_book_sender,
        mut order_book_receiver) = mpsc::channel::<OrderBookEvent>(64);
    let listener_handler = ListenerHandler {
        db: db.clone(),
        instruments: instruments.clone(),
        ticker: ticker.clone(),
        order_book_queue,
    };
    let listener_handler_clone = listener_handler.clone();
    tokio::spawn(async move {
        while let Some(event) = order_book_receiver.recv().await {
            listener_handler_clone.on_order_book(event).await;
        }
    });

    let instruments = instruments.values().cloned().collect::<Vec<_>>();
    tokio::spawn(async move {
        if let Err(err) = broker.listen_order_books(
            &instruments,
            order_book_sender,
        ).await {
            error!("Listener error: {}", err);
        }
    });
}
