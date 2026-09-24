mod listener;

use crate::listener::listen;
use alor::Alor;
use chrono::{TimeDelta, Utc};
use db::{Db, QueryAsksOrBids};
use log::{error, info, warn};
use model::common::{EmptyResult, TimeDiapason};
use model::events::{OrderBookEvent, OrderEvent, merge_events};
use model::{Deal, Instrument, Trade, TradeDirection};
use signal::{TradeSignal, WalkForwardModel, WindowData};
use simplelog::{LevelFilter, SimpleLogger};
use tokio::sync::mpsc;
use tokio_schedule::Job;

const LISTEN_TICKERS: [&str; 9] = ["GLZ6", "GLH7", "GLM7", "GLU7", "BMV6", "BMX6", "BMZ6", "SiZ6", "SiH7"];
const SIGNAL_TICKERS: [&str; 2] = ["GLZ6", "GLH7"];
const TRADE_TICKER: &str = "GLZ6GLH7";

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();

    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    info!("Initialize DB");
    let db = Db::new(&db_url).await?;

    let alor = Alor::default();

    if std::env::var("UPDATE_INSTRUMENTS_LIST").map_or(false, |s| s == "true") {
        let mut instruments = alor.get_instruments().await?;
        db.insert_instruments(&mut instruments).await?;
        return Ok(());
    }

    let instruments = db.get_instruments().await?;
    let (order_book_sender, order_book_receiver) = mpsc::channel::<OrderBookEvent>(64);
    let (trades_sender, trades_receiver) = mpsc::channel(64);

    let signal_instruments = instruments.iter()
        .filter(|instrument| SIGNAL_TICKERS.contains(&instrument.ticker.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let trade_instrument = instruments.iter()
        .find(|instrument| instrument.ticker == TRADE_TICKER)
        .cloned()
        .ok_or(format!("Instrument {TRADE_TICKER} not found"))?;
    let db_clone = db.clone();
    let alor_clone = alor.clone();
    tokio::spawn(async move {
        if let Err(err) = trade(
            order_book_receiver,
            trades_receiver,
            signal_instruments,
            trade_instrument,
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
    let events = merge_events(inst1, inst2, &ob1, &ob2);
    info!("Got {} history events", events.len());
    for event in events {
        model.process(window_data, &event);
    }

    Ok(())
}

async fn trade(
    mut order_book_receiver: mpsc::Receiver<OrderBookEvent>,
    mut trade_receiver: mpsc::Receiver<Trade>,
    order_book_instruments: Vec<Instrument>,
    trade_instrument: Instrument,
    db: &Db,
    alor: &Alor,
) -> EmptyResult {
    let inst1_id = order_book_instruments.get(0).map(|instrument| instrument.id).flatten();
    let inst2_id = order_book_instruments.get(1).map(|instrument| instrument.id).flatten();
    if let Some(inst1_id) = inst1_id && let Some(inst2_id) = inst2_id && let Some(trade_inst_id) = trade_instrument.id {
        let model = WalkForwardModel::new(inst1_id, inst2_id);
        let mut window_data = WindowData::new();
        consume_history(inst1_id, inst2_id, &db, &model, &mut window_data).await?;
        model.calibrate(false);

        let model_clone = model.clone();
        let calibrate_task = tokio_schedule::every(1).minute().perform(move || {
            let model_clone = model_clone.clone();
            async move {
                model_clone.calibrate(false);
            }
        });
        tokio::spawn(calibrate_task);

        let (order_response_sender, mut order_response_receiver) = mpsc::channel(64);
        let order_event_sender = alor.start_commands_stream(order_response_sender).await?;

        info!("Start trading");
        let mut active_deal = Option::<Deal>::None;

        loop {
            tokio::select! {
                order_book_event = order_book_receiver.recv() => {
                    if let Some(event) = order_book_event {
                        if active_deal.is_none() {
                            if let Some((dir, hold)) = match model.process(&mut window_data, &event) {
                                TradeSignal::Sell1Buy2(hold) => Some((TradeDirection::Buy, hold)),
                                TradeSignal::Buy1Sell2(hold) => Some((TradeDirection::Sell, hold)),
                                _ => None,
                            } {
                                if let Ok(deal) = db.start_deal(trade_inst_id, dir).await {
                                    info!("Start {deal:?}");
                                    let deal_id = deal.id;
                                    active_deal = Some(deal);
                                    if let Err(err) = order_event_sender.send(OrderEvent {
                                        deal_id,
                                        direction: dir,
                                        instrument_id: trade_inst_id,
                                        quantity: 1,
                                        ticker: trade_instrument.ticker.clone(),
                                    }).await {
                                        error!("Open order event send error {err}");
                                    }

                                    let ticker = trade_instrument.ticker.clone();
                                    let sender = order_event_sender.clone();
                                    tokio::spawn(async move {
                                        tokio::time::sleep(tokio::time::Duration::from_millis(hold as u64)).await;
                                        if let Err(err) = sender.send(OrderEvent {
                                            deal_id,
                                            direction: dir.reverse(),
                                            instrument_id: trade_inst_id,
                                            quantity: 1,
                                            ticker,
                                        }).await {
                                            error!("Close order event send error {err}");
                                        }
                                    });
                                }
                            }
                        }
                    } else {
                        warn!("OrderBookEvent channel is closed");
                        break;
                    }
                }

                order_response_event = order_response_receiver.recv() => {
                    if let Some(event) = order_response_event {
                        info!("Received order event {event:?}");
                    } else {
                        warn!("OrderBookEvent channel is closed");
                        break;
                    }
                }

                trade_event = trade_receiver.recv() => {
                    if let Some(event) = trade_event {
                        if let Some(deal) = active_deal.as_mut() {
                            if event.direction == deal.direction {
                                deal.open_order_id = Some(event.order_id.clone());
                                if let Err(err) = db.update_deal(deal.id, Some(event.order_id.clone()), None).await {
                                    error!("Update deal error {err}");
                                }
                            } else {
                                deal.close_order_id = Some(event.order_id.clone());
                                if let Err(err) = db.update_deal(deal.id, None, Some(event.order_id.clone())).await {
                                    error!("Update deal error {err}");
                                }
                            }
                            if deal.open_order_id.is_some() && deal.close_order_id.is_some() {
                                active_deal = None;
                            }
                        }
                    } else {
                        warn!("TradeEvent channel is closed");
                        break;
                    }
                }
            }
        }

        while let Some(event) = order_book_receiver.recv().await {

            let _signal = model.process(&mut window_data, &event);
        }
        Ok(())
    } else {
        Err("Not enough signal instruments".into())
    }
}