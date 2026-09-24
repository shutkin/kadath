mod orderbook_values;
mod signal_calculator;

use crate::orderbook_values::OrderBookValues;
use crate::signal_calculator::{SignalCalculator, trade_sim};
use chrono::{DateTime, TimeDelta, Utc};
use db::{Db, QueryAsksOrBids};
use log::{LevelFilter, info};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::{Instrument, OrderBook, order_book_cache};
use simplelog::SimpleLogger;

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = db::Db::new(&db_url).await?;

    let ticker = "GLU6";

    let all_instruments = db.get_instruments().await?;
    if let Some(inst_id) = find_instrument_id(&all_instruments, ticker) {
        let train_diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-09-04T05:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-09-04T15:00:00Z")?.to_utc(),
        );
        let test_diapason = TimeDiapason::new(
            DateTime::parse_from_rfc3339("2026-09-04T15:00:00Z")?.to_utc(),
            DateTime::parse_from_rfc3339("2026-09-04T20:00:00Z")?.to_utc(),
        );

        let values = get_order_books_values(ticker, inst_id, train_diapason, &db).await?;

        let mut horizon_results = Vec::new();
        for i in 100_000 .. 110_000 {
            let v_from = &values[i];
            if let Some(v_to) = find_value_on_horizon(&values, i, v_from.time + TimeDelta::milliseconds(1000)) {
                horizon_results.push(v_to.bid - v_from.ask);
            }
        }
        let csv = generate_csv(&values[100_000 .. 110_000], &horizon_results);
        std::fs::write("values.csv", &csv)?;

        if let Some(calculator) = SignalCalculator::optimize(&values, 1000) {
            let values = get_order_books_values(ticker, inst_id, test_diapason, &db).await?;
            let result = trade_sim(&values, &calculator, 1000);
            info!("Test trade profit {}, commission {}", result.income - result.outcome - result.commission, result.commission);
        }
    }
    Ok(())
}

fn find_value_on_horizon(values: &[OrderBookValues], index: usize, target_time: DateTime<Utc>) -> Option<OrderBookValues> {
    let mut i = index;
    while i < values.len() {
        if values[i].time >= target_time {
            return Some(values[i]);
        }
        i += 1;
    }
    None
}

fn generate_csv(values: &[OrderBookValues], results: &[f64]) -> String {
    let mut lines = Vec::new();
    lines.push("d,t,i1,i2,i3,i4,r".to_string());
    values.iter().enumerate().for_each(|(i, v)| {
        let line = format!("{},{:.4}", v.to_csv(), results[i]);
        lines.push(line);
    });
    lines.join("\n")
}

fn find_instrument_id(instruments: &[Instrument], ticker: &str) -> Option<i32> {
    instruments
        .iter()
        .find(|instrument| instrument.ticker == ticker)
        .and_then(|instrument| instrument.id)
}

async fn get_order_books_values(ticker: &str, id: i32, diapason: TimeDiapason, db: &Db)
    -> Result<Vec<OrderBookValues>, CommonError> {
    let order_books = get_order_books(ticker, id, diapason, db).await?;
    let mut values = order_books.iter().filter_map(OrderBookValues::calculate).collect::<Vec<_>>();
    info!("Calculate {} window values", values.len());
    OrderBookValues::calculate_window_values(&mut values, 1000, 30.0);
    Ok(values)
}

async fn get_order_books(ticker: &str, id: i32, diapason: TimeDiapason, db: &Db)
                         -> Result<Vec<OrderBook>, CommonError> {
    if let Some(order_books1) =
        order_book_cache::read(ticker, diapason, true)? {
        Ok(order_books1)
    } else {
        let order_books1 = db.get_order_books(Some(id), diapason, QueryAsksOrBids::BOTH).await?;
        order_book_cache::write(ticker, diapason, &order_books1, true)?;
        Ok(order_books1)
    }
}
