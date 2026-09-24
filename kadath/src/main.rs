use alor::Alor;
use chrono::Utc;
use db::Db;
use log::{debug, error, info, warn};
use model::common::EmptyResult;
use model::events::{OrderBookDeltaEvent, OrderBookEvent};
use model::utils::handle_delta;
use model::{Instrument, OrderBook};
use simplelog::{LevelFilter, SimpleLogger};
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
}

impl ListenerHandler {
    async fn on_order_book(&self, event: OrderBookEvent) {
        self.ticker.store(Utc::now().timestamp(), Relaxed);
        if let Some(instrument) = self.instruments.get(&event.instrument_id) {
            info!("Save snapshot order book {:?}", event.order_book);
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
            if let Ok(mut lock) = self.order_books.lock()
                && let Some(order_book) = lock.get_mut(&event.instrument_id) {
                let prev_timestamp_quant = order_book.timestamp.timestamp_micros() / (FULL_ORDER_BOOK_QUANT_MINUTES * 60 * 1000 * 1000);
                order_book.timestamp = event.timestamp;
                handle_delta(&event.asks, &mut order_book.asks);
                handle_delta(&event.bids, &mut order_book.bids);
                let cur_timestamp_quant = order_book.timestamp.timestamp_micros() / (FULL_ORDER_BOOK_QUANT_MINUTES * 60 * 1000 * 1000);
                if cur_timestamp_quant > prev_timestamp_quant {
                    updated_order_book = Some(order_book.clone());
                }
            } else {
                error!("Failed to lock order books");
            }
            if let Some(order_book) = updated_order_book {
                info!("Save full order book {order_book:?}");
                if let Err(err) = self.db.insert_order_book_full(&order_book, instrument).await {
                    error!("Failed to save order book: {}", err);
                }
            } else {
                debug!("Save order book delta asks {:?}, bids {:?}", event.asks, event.bids);
                if let Err(err) = self.db.insert_order_book_delta(instrument, event.timestamp, &event.asks, &event.bids).await {
                    error!("Failed to save order book delta: {}", err);
                }
            }
        } else {
            warn!("Skipping order book for unknown instrument {}", event.instrument_id);
        }
    }
}

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();

    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    let alor = Alor::default();

    let mut instruments = alor.get_instruments().await?;
    db.insert_instruments(&mut instruments).await?;
    let tickers = ["GLZ6", "GLH7", "GLM7", "GLU7", "BMV6", "BMX6", "BMZ6", "SiZ6", "SiH7"];
    let mut instruments_map = HashMap::new();
    instruments.iter().for_each(|instrument| {
        if tickers.contains(&instrument.ticker.as_str()) && let Some(id) = instrument.id {
            instruments_map.insert(id, instrument.clone());
        }
    });

    let db_clone = db.clone();
    let instruments_map_clone = instruments_map.clone();

    let ticker = Arc::new(AtomicI64::new(0));
    loop {
        let now = Utc::now().timestamp();
        if ticker.load(Relaxed) < now - 5 * 60 {
            start_listener(db_clone.clone(), instruments_map_clone.clone(), ticker.clone());
            ticker.store(Utc::now().timestamp(), Relaxed);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn start_listener(db: Db, instruments: HashMap<i32, Instrument>, ticker: Arc<AtomicI64>) {
    let (order_book_sender,
        mut order_book_receiver) = mpsc::channel::<OrderBookEvent>(64);
    let (order_book_delta_sender,
        mut order_book_delta_receiver) = mpsc::channel::<OrderBookDeltaEvent>(64);
    let listener_handler = ListenerHandler {
        db: db.clone(),
        instruments: instruments.clone(),
        order_books: Arc::new(Mutex::new(HashMap::new())),
        ticker: ticker.clone(),
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
        let alor = Alor::default();
        if let Err(err) = alor.listen(
            &instruments,
            order_book_sender,
            order_book_delta_sender,
        ).await {
            error!("Listener error: {}", err);
        }
    });
}
