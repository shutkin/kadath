use alor::Alor;
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
    order_book_event_sender: mpsc::Sender<OrderBookEvent>,
}

impl ListenerHandler {
    async fn on_order_book(&self, event: OrderBookEvent) {
        self.ticker.store(Utc::now().timestamp(), Relaxed);
        if let Some(instrument) = self.instruments.get(&event.instrument_id) {
            if let Err(err) = self.order_book_event_sender.send(event.clone()).await {
                error!("Error sending order book event: {}", err);
            }

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
                if let Err(err) = self.order_book_event_sender.send(OrderBookEvent {
                    instrument_id: event.instrument_id,
                    order_book: order_book.clone(),
                }).await {
                    error!("Error sending order book event on delta: {}", err);
                }

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
    order_book_event_sender: mpsc::Sender<OrderBookEvent>,
    trades_event_sender: mpsc::Sender<Trade>,
    alor: &Alor,
) -> EmptyResult {
    let mut instruments_map = HashMap::new();
    instruments.into_iter().for_each(|instrument| {
        if let Some(id) = instrument.id {
            instruments_map.insert(id, instrument);
        }
    });
    
    let db_clone = db.clone();
    let (trades_sender, mut trades_receiver) = mpsc::channel(64);
    tokio::spawn(async move {
        while let Some(mut trade) = trades_receiver.recv().await {
            if let Err(err) = db_clone.insert_trade(&mut trade).await {
                error!("Failed to insert trade: {err}");
            }
            
            if let Err(err) = trades_event_sender.send(trade).await {
                error!("Error sending trades event: {err}");
            }
        }
    });
    
    let alor_clone = alor.clone();
    tokio::spawn(async move {
        loop {
            if let Err(err) = alor_clone.listen_trades(trades_sender.clone()).await {
                error!("Error listening trades: {}", err);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    let db_clone = db.clone();
    let alor_clone = alor.clone();
    let instruments_map_clone = instruments_map.clone();
    let ticker = Arc::new(AtomicI64::new(0));
    loop {
        let now = Utc::now().timestamp();
        if ticker.load(Relaxed) < now - 5 * 60 {
            start_listener(db_clone.clone(), instruments_map_clone.clone(), ticker.clone(), order_book_event_sender.clone(), alor_clone.clone());
            ticker.store(Utc::now().timestamp(), Relaxed);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn start_listener(
    db: Db,
    instruments: HashMap<i32, Instrument>,
    ticker: Arc<AtomicI64>,
    order_book_event_sender: mpsc::Sender<OrderBookEvent>,
    alor: Alor,
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
        order_book_event_sender,
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
        if let Err(err) = alor.listen_order_books(
            &instruments,
            order_book_sender,
            order_book_delta_sender,
        ).await {
            error!("Listener error: {}", err);
        }
    });
}
