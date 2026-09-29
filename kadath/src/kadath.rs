mod listener;

use crate::listener::listen;
use alor::Alor;
use chrono::{TimeDelta, Utc};
use db::{Db, QueryAsksOrBids};
use log::{error, info, warn};
use model::common::{EmptyResult, TimeDiapason};
use model::events::{merge_events, OrderBookEvent, OrderEvent, OrderResponseEvent};
use model::{Deal, OrderBook, OrderData, SignalCalculatorLog, Trade, TradeDirection};
use signal::{ModelConfig, TradeSignal, WalkForwardModel, WindowData};
use simplelog::{LevelFilter, SimpleLogger};
use std::sync::{Arc, Mutex};
use rust_decimal::Decimal;
use tokio::sync::mpsc;
use tokio_schedule::Job;

const CHANNEL_CAPACITY: usize = 65536;

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

    let alor = Alor::default();
    let positions_text = alor.client_positions().await?;
    info!("Client positons: {positions_text}");

    if std::env::var("UPDATE_INSTRUMENTS_LIST").map_or(false, |s| s == "true") {
        let mut instruments = alor.get_instruments().await?;
        db.insert_instruments(&mut instruments).await?;
        return Ok(());
    }

    let instruments = db.get_instruments().await?;
    let (order_book_sender, order_book_receiver) = mpsc::channel::<OrderBookEvent>(CHANNEL_CAPACITY);
    let (trades_sender, trades_receiver) = mpsc::channel::<Trade>(CHANNEL_CAPACITY);

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
    consume_history(inst1_id, inst2_id, &db, &model, &mut window_data).await?;
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

    let db_clone = db.clone();
    let alor_clone = alor.clone();
    tokio::spawn(async move {
        if let Err(err) = trade(
            order_book_receiver,
            trades_receiver,
            model,
            window_data,
            inst1_id,
            inst2_id,
            trade_instrument.id.unwrap(),
            trade_instrument.ticker,
            &db_clone, &alor_clone).await {
            error!("Trade stopped with error {err}");
        }
    });

    let listen_instruments = instruments.iter()
        .filter(|instrument| LISTEN_TICKERS.contains(&instrument.ticker.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    listen(db.clone(), listen_instruments, order_book_sender, trades_sender, &alor).await?;

    Ok(())
}

async fn consume_history(inst1: i32, inst2: i32, db: &Db, model: &WalkForwardModel, window_data: &mut WindowData) -> EmptyResult {
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

async fn handle_trade(mut trade_receiver: mpsc::Receiver<Trade>, deal: Arc<Mutex<Option<Deal>>>, db: Db) {
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

async fn trade(
    mut order_book_receiver: mpsc::Receiver<OrderBookEvent>,
    trade_receiver: mpsc::Receiver<Trade>,
    model: WalkForwardModel,
    mut window_data: WindowData,
    instrument1_id: i32,
    instrument2_id: i32,
    trade_instrument_id: i32,
    trade_instrument_ticker: String,
    db: &Db,
    alor: &Alor,
) -> EmptyResult {
    let (order_response_sender, order_response_receiver) = mpsc::channel(64);
    let order_event_sender = alor.start_commands_stream(order_response_sender).await?;

    info!("Start trading");
    let active_deal = Arc::new(Mutex::new(Option::<Deal>::None));

    tokio::spawn(async move {
        handle_order_responses(order_response_receiver).await;
    });

    let deal_clone = active_deal.clone();
    let db_clone = db.clone();
    tokio::spawn(async move {
        handle_trade(trade_receiver, deal_clone, db_clone).await;
    });

    let (mut last_ob1, mut last_ob2) = (None, None);
    let mut close_deal_sent = false;
    let mut cnt = 0;

    while let Some(event) = order_book_receiver.recv().await {
        if event.instrument_id == instrument1_id {
            last_ob1 = Some(event.order_book.clone());
        } else if event.instrument_id == instrument2_id {
            last_ob2 = Some(event.order_book.clone());
        } else {
            continue;
        }

        cnt += 1;
        if cnt == 128 {
            order_event_sender.send(OrderEvent {
                deal_id: 0,
                direction: TradeDirection::Buy,
                instrument_id: trade_instrument_id,
                quantity: 1,
                ticker: trade_instrument_ticker.clone(),
                expected_price: lowest_price(&last_ob1.as_ref().unwrap().asks).unwrap(),
            }).await?;
        }

        let signal = model.process(&mut window_data, &event);

        let cur_deal = if let Ok(lock) = active_deal.lock() {
            lock.clone()
        } else { None };

        if let Some(deal) = cur_deal {
            let close_time = deal.started + TimeDelta::milliseconds(deal.hold as i64);
            if let Some(ob1) = &last_ob1 && let Some(ob2) = &last_ob2
                && (close_time < ob1.timestamp || close_time < ob2.timestamp)
                && let Some(expected_price) = get_single_close_price(ob1, deal.direction) {
                if !close_deal_sent {
                    if let Err(err) = order_event_sender.send(OrderEvent {
                        deal_id: deal.id,
                        direction: deal.direction.reverse(),
                        instrument_id: trade_instrument_id,
                        quantity: 1,
                        ticker: trade_instrument_ticker.clone(),
                        expected_price,
                    }).await {
                        error!("Close order event send error {err}");
                    } else {
                        close_deal_sent = true;
                    }
                } else if ob1.timestamp.max(ob2.timestamp) > close_time + TimeDelta::seconds(10)
                    && let Ok(mut deal) = active_deal.lock() {
                    *deal = None;
                }
            }
        } else {
            if let Some((dir, hold)) = match signal {
                TradeSignal::Sell1Buy2(hold) => Some((TradeDirection::Sell, hold)),
                TradeSignal::Buy1Sell2(hold) => Some((TradeDirection::Buy, hold)),
                _ => None,
            } && let Some(ob1) = &last_ob1 && let Some(ob2) = &last_ob2
                && let Some(expected_price) = get_single_open_price(ob1, dir) {
                match db.start_deal(trade_instrument_id, dir, hold, Some(expected_price)).await {
                    Ok(deal) => {
                        info!("Start {deal:?}");
                        close_deal_sent = false;
                        let deal_id = deal.id;
                        if let Ok(mut lock) = active_deal.lock() {
                            *lock = Some(deal);
                        }
                        if let Err(err) = order_event_sender.send(OrderEvent {
                            deal_id,
                            direction: dir,
                            instrument_id: trade_instrument_id,
                            quantity: 1,
                            ticker: trade_instrument_ticker.clone(),
                            expected_price,
                        }).await {
                            error!("Open order event send error {err}");
                        }
                    }
                    Err(err) => {
                        error!("Failed to create deal: {err}");
                    }
                }
            }
        }
    }
    warn!("OrderBookEvent channel is closed");

    Ok(())
}

fn get_single_open_price(ob: &OrderBook, direction: TradeDirection) -> Option<Decimal> {
    match direction {
        TradeDirection::Buy => lowest_price(&ob.asks),
        TradeDirection::Sell => highest_price(&ob.bids),
    }
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