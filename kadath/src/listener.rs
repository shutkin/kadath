use crate::queue::Queue;
use abstract_broker::BrokerListener;
use chrono::Utc;
use db::Db;
use log::{debug, error, warn};
use model::common::EmptyResult;
use model::events::{OrderBookDeltaEvent, OrderBookEvent};
use model::utils::handle_delta;
use model::{Instrument, OrderBook, Trade};
use std::collections::HashMap;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

const FULL_ORDER_BOOK_QUANT_MINUTES: i64 = 10;

#[derive(Clone)]
struct ListenerHandler {
    db: Db,
    instruments: HashMap<i32, Instrument>,
    order_books: Arc<Mutex<HashMap<i32, OrderBook>>>,
    ticker: Arc<AtomicI64>,
    order_book_queue: Arc<Queue<OrderBookEvent>>,
}

impl ListenerHandler {
    async fn on_order_book(&self, event: OrderBookEvent) {
        self.ticker.store(Utc::now().timestamp(), Relaxed);
        if let Some(instrument) = self.instruments.get(&event.instrument_id) {
            self.order_book_queue.push(event.clone());

            debug!("Save snapshot order book {:?}", event.order_book);
            if let Err(err) = self.db.insert_order_book_full(&event.order_book, instrument).await {
                error!("Failed to save order book: {}", err);
            }
            if let Ok(mut lock) = self.order_books.lock() {
                lock.insert(event.instrument_id, event.order_book);
            } else {
                error!("Failed to lock order books");
            }
        }
    }

    async fn on_order_book_delta(&self, event: OrderBookDeltaEvent) {
        self.ticker.store(Utc::now().timestamp(), Relaxed);
        if let Some(instrument) = self.instruments.get(&event.instrument_id) {
            let mut updated_order_book = None;
            let mut save_full = false;

            if let Ok(mut lock) = self.order_books.lock()
                && let Some(order_book) = lock.get_mut(&event.instrument_id) {
                let prev_timestamp_quant = order_book.timestamp.timestamp_micros() / (FULL_ORDER_BOOK_QUANT_MINUTES * 60 * 1000 * 1000);
                order_book.timestamp = event.timestamp;
                handle_delta(&event.asks, &mut order_book.asks);
                handle_delta(&event.bids, &mut order_book.bids);
                updated_order_book = Some(order_book.clone());
                let cur_timestamp_quant = order_book.timestamp.timestamp_micros() / (FULL_ORDER_BOOK_QUANT_MINUTES * 60 * 1000 * 1000);
                if cur_timestamp_quant > prev_timestamp_quant {
                    save_full = true;
                }
            } else {
                error!("Failed to lock order books");
            }

            if let Some(order_book) = updated_order_book {
                self.order_book_queue.push(OrderBookEvent {
                    instrument_id: event.instrument_id,
                    order_book: order_book.clone(),
                });

                if save_full {
                    debug!("Save full order book {order_book:?}");
                    if let Err(err) = self.db.insert_order_book_full(&order_book, instrument).await {
                        error!("Failed to save order book: {}", err);
                    }
                } else {
                    debug!("Save order book delta asks {:?}, bids {:?}", event.asks, event.bids);
                    if let Err(err) = self.db.insert_order_book_delta(instrument, event.timestamp, &event.asks, &event.bids).await {
                        error!("Failed to save order book delta: {}", err);
                    }
                }
            }
        } else {
            warn!("Skipping order book for unknown instrument {}", event.instrument_id);
        }
    }
}

pub async fn listen(
    db: Db,
    instruments: Vec<Instrument>,
    order_book_queue: Arc<Queue<OrderBookEvent>>,
    trade_queue: Arc<Queue<Trade>>,
    broker: Arc<Box<dyn BrokerListener + Sync + Send>>,
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

async fn listen_trades(db: Db, broker: Arc<Box<dyn BrokerListener + Sync + Send>>, trade_queue: Arc<Queue<Trade>>) -> EmptyResult {
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
    broker: Arc<Box<dyn BrokerListener + Sync + Send>>,
) {
    let (order_book_sender,
        mut order_book_receiver) = mpsc::channel::<OrderBookEvent>(64);
    let (order_book_delta_sender,
        mut order_book_delta_receiver) = mpsc::channel::<OrderBookDeltaEvent>(64);
    let listener_handler = ListenerHandler {
        db: db.clone(),
        instruments: instruments.clone(),
        order_books: Arc::new(Mutex::new(HashMap::new())),
        ticker: ticker.clone(),
        order_book_queue,
    };
    let listener_handler_clone = listener_handler.clone();
    tokio::spawn(async move {
        while let Some(event) = order_book_receiver.recv().await {
            listener_handler_clone.on_order_book(event).await;
        }
    });
    tokio::spawn(async move {
        while let Some(event) = order_book_delta_receiver.recv().await {
            listener_handler.on_order_book_delta(event).await;
        }
    });

    let instruments = instruments.values().cloned().collect::<Vec<_>>();
    tokio::spawn(async move {
        if let Err(err) = broker.listen_order_books(
            &instruments,
            order_book_sender,
            order_book_delta_sender,
        ).await {
            error!("Listener error: {}", err);
        }
    });
}
