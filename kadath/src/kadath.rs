mod listener;

use chrono::{TimeDelta, Utc};
use log::{error, info};
use crate::listener::listen;
use alor::Alor;
use db::{Db, QueryAsksOrBids};
use model::common::{EmptyResult, TimeDiapason};
use model::events::{merge_events, OrderBookEvent};
use simplelog::{LevelFilter, SimpleLogger};
use tokio::sync::mpsc;
use tokio_schedule::Job;
use model::Instrument;
use signal::{WalkForwardModel, WindowData};

const LISTEN_TICKERS: [&str; 9] = ["GLZ6", "GLH7", "GLM7", "GLU7", "BMV6", "BMX6", "BMZ6", "SiZ6", "SiH7"];
const SIGNAL_TICKERS: [&str; 2] = ["GLZ6", "GLH7"];

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
    let (order_book_sender,
        mut order_book_receiver) = mpsc::channel::<OrderBookEvent>(64);

    let signal_instruments = instruments.iter()
        .filter(|instrument| SIGNAL_TICKERS.contains(&instrument.ticker.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let db_clone = db.clone();
    tokio::spawn(async move {
        //while let Some(_event) = order_book_receiver.recv().await {
            // do nothing
        //}
        if let Err(err) = trade(order_book_receiver, signal_instruments, db_clone).await {
            error!("Trade stopped with error {err}");
        }
    });

    let listen_instruments = instruments.iter()
        .filter(|instrument| LISTEN_TICKERS.contains(&instrument.ticker.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    //listen(db.clone(), listen_instruments, order_book_sender).await?;
    loop {}
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

async fn trade(mut order_book_receiver: mpsc::Receiver<OrderBookEvent>, instruments: Vec<Instrument>, db: Db) -> EmptyResult {
    let inst1_id = instruments.get(0).map(|instrument| instrument.id).flatten();
    let inst2_id = instruments.get(1).map(|instrument| instrument.id).flatten();
    if let Some(inst1_id) = inst1_id && let Some(inst2_id) = inst2_id {
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

        info!("Start trading");
        while let Some(event) = order_book_receiver.recv().await {
            let _signal = model.process(&mut window_data, &event);
        }
        Ok(())
    } else {
        Err("Not enough signal instruments".into())
    }
}