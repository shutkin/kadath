mod listener;
mod queue;
mod executor;
mod trader;

use crate::listener::listen;
use crate::queue::Queue;
use crate::trader::KadathTrader;
use abstract_broker::AbstractBroker;
use alor::Alor;
use db::Db;
use log::{error, info};
use model::common::EmptyResult;
use model::events::OrderBookEvent;
use model::{DataSource, Trade};
use simplelog::{LevelFilter, SimpleLogger};
use std::sync::Arc;
use tinvest::TInvest;

const QUEUE_CAPACITY: usize = 65536;

const LISTEN_TICKERS: [&str; 9] = ["GLZ6", "GLH7", "GLM7", "GLU7", "BMV6", "BMX6", "BMZ6", "SiZ6", "SiH7"];
const TRADE_TICKER: &str = "GLZ6";

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();

    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    info!("Initialize DB");
    let db = Db::new(&db_url).await?;

    let data_source_code = std::env::var("DATA_SOURCE").expect("DATA_SOURCE is not set");
    let data_source = DataSource::from_code(&data_source_code);
    
    let broker: Box<dyn AbstractBroker + Send + Sync> = match data_source {
        DataSource::Alor => Box::new(Alor::default()),
        DataSource::TInvest => Box::new(TInvest::new().await?),
    };
    let broker = Arc::new(broker);

    if std::env::var("UPDATE_INSTRUMENTS_LIST").map_or(false, |s| s == "true") {
        let mut instruments = broker.get_instruments().await?;
        db.insert_instruments(&mut instruments).await?;
        return Ok(());
    }

    let instruments = db.get_instruments(data_source).await?;
    let order_book_queue = Arc::new(Queue::<OrderBookEvent>::new(QUEUE_CAPACITY));
    let trade_queue = Arc::new(Queue::<Trade>::new(QUEUE_CAPACITY));

    let listen_instruments = instruments.iter()
        .filter(|instrument| LISTEN_TICKERS.contains(&instrument.ticker.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let db_clone = db.clone();
    let order_book_queue_clone = order_book_queue.clone();
    let trade_queue_clone = trade_queue.clone();
    let broker_clone = broker.clone();
    tokio::spawn(async move {
        if let Err(err) = listen(db_clone, listen_instruments, order_book_queue_clone, trade_queue_clone, broker_clone).await {
            error!("Listener failed: {err}");
        }
        error!("Listener stopped");
    });

    let trade_instrument = instruments.iter()
        .find(|instrument| TRADE_TICKER == instrument.ticker.as_str())
        .cloned();
    if let Some(trade_instrument) = trade_instrument {
        let trader = KadathTrader::new(broker, order_book_queue, trade_instrument, db, trade_queue);
        if let Err(err) = trader.run().await {
            error!("Trader failed: {err}");
        }
    }
    error!("Trader stopped");

    Ok(())
}
