mod listener;
mod queue;

use crate::listener::listen;
use crate::queue::Queue;
use abstract_broker::BrokerListener;
use alor::Alor;
use chrono::{TimeDelta, Utc};
use db::{Db, QueryAsksOrBids};
use log::{error, info, warn};
use model::common::{EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, OrderResponseEvent, merge_events};
use model::{DataSource, OrderBook, OrderData, SignalCalculatorLog, SpreadDeal, Trade, TradeDirection};
use rust_decimal::Decimal;
use signal::{ModelConfig, WalkForwardModel, WindowData};
use simplelog::{LevelFilter, SimpleLogger};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_schedule::Job;
use tinvest::TInvest;

const QUEUE_CAPACITY: usize = 65536;

const LISTEN_TICKERS: [&str; 9] = ["GLZ6", "GLH7", "GLM7", "GLU7", "BMV6", "BMX6", "BMZ6", "SiZ6", "SiH7"];
const SIGNAL_TICKERS: [&str; 2] = ["GLZ6", "GLH7"];
const TRADE_TICKER: &str = "GLZ6";//GLH7";

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();

    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    info!("Initialize DB");
    let db = Db::new(&db_url).await?;

    let data_source_code = std::env::var("DATA_SOURCE").expect("DATA_SOURCE is not set");
    let data_source = DataSource::from_code(&data_source_code);
    
    let broker: Box<dyn BrokerListener + Send + Sync> = match data_source {
        DataSource::Alor => Box::new(Alor::default()),
        DataSource::TInvest => Box::new(TInvest::new().await?),
    };
    let broker = Arc::new(broker);

    let alor = Alor::default();
    let positions_text = alor.client_positions().await?;
    info!("Client positons: {positions_text}");

    if std::env::var("UPDATE_INSTRUMENTS_LIST").map_or(false, |s| s == "true") {
        let mut instruments = alor.get_instruments().await?;
        db.insert_instruments(&mut instruments).await?;
        return Ok(());
    }

    let instruments = db.get_instruments(data_source).await?;
    let order_book_queue = Arc::new(Queue::<OrderBookEvent>::new(QUEUE_CAPACITY));
    let trade_queue = Arc::new(Queue::<Trade>::new(QUEUE_CAPACITY));

    let signal_instruments = instruments.iter()
        .filter(|instrument| SIGNAL_TICKERS.contains(&instrument.ticker.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let trade_instrument = instruments.iter()
        .find(|instrument| instrument.ticker == TRADE_TICKER)
        .cloned()
        .ok_or(format!("Instrument {TRADE_TICKER} not found"))?;

    let inst1_id = signal_instruments.get(0)
        .map(|instrument| instrument.id)
        .flatten().ok_or(format!("Signal instrument {} not found", SIGNAL_TICKERS[0]))?;
    let inst2_id = signal_instruments.get(1)
        .map(|instrument| instrument.id)
        .flatten().ok_or(format!("Signal instrument {} not found", SIGNAL_TICKERS[1]))?;

    let config = ModelConfig { std_diapason_s: 1184, train_data_minutes: 22, train_interval_minutes: 1, decay_time: 40.90421962773701, fixed_hold_time: None };
    let model = WalkForwardModel::new_with_config(inst1_id, inst2_id, config);
    let mut window_data = WindowData::new();
    consume_history(data_source, inst1_id, inst2_id, &db, &model, &mut window_data).await?;
    model.calibrate(false);

    let model_clone = model.clone();
    let db_clone = db.clone();
    let calibrate_task = tokio_schedule::every(1).minute().perform(move || {
        let model_clone = model_clone.clone();
        let db_clone = db_clone.clone();
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

    let listen_instruments = instruments.iter()
        .filter(|instrument| LISTEN_TICKERS.contains(&instrument.ticker.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    listen(db, listen_instruments, order_book_queue, trade_queue, broker).await?;

    Ok(())
}

async fn consume_history(data_source: DataSource, inst1: i32, inst2: i32, db: &Db, model: &WalkForwardModel, window_data: &mut WindowData) -> EmptyResult {
    let diapason = TimeDiapason::new(
        Utc::now() - TimeDelta::hours(24),
        Utc::now(),
    );
    let ob1 = db.get_order_books(Some(inst1), diapason, QueryAsksOrBids::BOTH).await?;
    let ob2 = db.get_order_books(Some(inst2), diapason, QueryAsksOrBids::BOTH).await?;
    let events = merge_events(&[inst1, inst2], &[ob1, ob2]);
    info!("Got {} history events", events.len());
    for event in events {
        model.process(window_data, &event);
    }

    Ok(())
}

async fn handle_order_responses(mut order_response_receiver: mpsc::Receiver<OrderResponseEvent>) {
    while let Some(event) = order_response_receiver.recv().await {
        info!("Received order event {event:?}");
    }
    warn!("OrderBookEvent channel is closed");
}

async fn handle_trade(mut trade_receiver: mpsc::Receiver<Trade>, deal: Arc<Mutex<Option<SpreadDeal>>>, db: Db) {
    while let Some(event) = trade_receiver.recv().await {
        let mut deal_id = 0;
        let (mut open_order_id, mut close_order_id) = (None, None);

        if let Ok(mut lock) = deal.lock()
            && let Some(deal) = lock.as_mut() {
            deal_id = deal.id;

            if event.direction == deal.direction {
                if deal.open_order_id.is_none() {
                    deal.open_order_id = Some(event.order_id.clone());
                    open_order_id = Some(event.order_id.clone());
                }
            } else {
                if deal.close_order_id.is_none() {
                    deal.close_order_id = Some(event.order_id.clone());
                    close_order_id = Some(event.order_id.clone());
                }
            }

            if deal.open_order_id.is_some() && deal.close_order_id.is_some() {
                info!("Deal closed");
                *lock = None;
            }
        }

        if open_order_id.is_some() || close_order_id.is_some() {
            if let Err(err) = db.update_deal(deal_id, open_order_id, close_order_id).await {
                error!("Update deal error {err}");
            }
        }
    }
    warn!("TradeEvent channel is closed");

}

fn get_single_close_price(ob: &OrderBook, direction: TradeDirection) -> Option<Decimal> {
    match direction {
        TradeDirection::Buy => highest_price(&ob.bids),
        TradeDirection::Sell => lowest_price(&ob.asks),
    }
}

fn get_spread_open_price(ob1: &OrderBook, ob2: &OrderBook, direction: TradeDirection) -> Option<Decimal> {
    match direction {
        TradeDirection::Buy => {
            if let Some(p2) = lowest_price(&ob2.asks) && let Some(p1) = highest_price(&ob1.bids) {
                if p1 > p2 {
                    warn!("i1 bids: {:?}", ob1.bids);
                    warn!("i2 asks: {:?}", ob2.asks);
                }
                Some(p2 - p1)
            } else {
                None
            }
        }
        TradeDirection::Sell => {
            if let Some(p2) = highest_price(&ob2.bids) && let Some(p1) = lowest_price(&ob1.asks) {
                if p1 > p2 {
                    warn!("i1 asks: {:?}", ob1.asks);
                    warn!("i2 bids: {:?}", ob2.bids);
                }
                Some(p2 - p1)
            } else {
                None
            }
        }
    }
}

fn get_spread_close_price(ob1: &OrderBook, ob2: &OrderBook, direction: TradeDirection) -> Option<Decimal> {
    match direction {
        TradeDirection::Buy => {
            if let Some(p2) = highest_price(&ob2.bids) && let Some(p1) = lowest_price(&ob1.asks) {
                if p1 > p2 {
                    warn!("i1 asks: {:?}", ob1.asks);
                    warn!("i2 bids: {:?}", ob2.bids);
                }
                Some(p2 - p1)
            } else {
                None
            }
        }
        TradeDirection::Sell => {
            if let Some(p2) = lowest_price(&ob2.asks) && let Some(p1) = highest_price(&ob1.bids) {
                if p1 > p2 {
                    warn!("i1 bids: {:?}", ob1.bids);
                    warn!("i2 asks: {:?}", ob2.asks);
                }
                Some(p2 - p1)
            } else {
                None
            }
        }
    }
}

fn highest_price(data: &[OrderData]) -> Option<Decimal> {
    let max = data.iter()
        .map(|data| data.price)
        .max();
    if max.is_none() {
        warn!("Can't find highest price: {data:?}");
    }
    max
}

fn lowest_price(data: &[OrderData]) -> Option<Decimal> {
    let min = data.iter()
        .map(|data| data.price)
        .min();
    if min.is_none() {
        warn!("Can't find lowest price: {data:?}");
    }
    min
}