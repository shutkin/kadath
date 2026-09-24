use chrono::{NaiveDate, NaiveTime, TimeDelta, TimeZone, Utc};
use log::{error, info, LevelFilter};
use simplelog::SimpleLogger;
use db::Db;
use model::common::EmptyResult;

const DAYS: i64 = 65;
const IDS_BATCH_SIZE: i32 = 8192;
const SLEEP: u64 = 5;

#[tokio::main]
async fn main() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let db = Db::new(&db_url).await?;

    loop {
        if let Err(err) = clean(&db).await {
            error!("{err}");
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(SLEEP)).await;
    }
}

async fn clean(db: &Db) -> EmptyResult {
    let dt = NaiveDate::from_ymd_opt(2025, 12, 15)
        .unwrap_or_default().and_time(NaiveTime::default());
    let from = Utc.from_utc_datetime(&dt);
    let to = Utc::now() - TimeDelta::days(DAYS);
    info!("Query interval {from} - {to}");
    let ids = db.get_order_books_ids(from, to, IDS_BATCH_SIZE).await?;
    if !ids.is_empty() {
        db.remove_order_books(&ids, to).await?;
    }
    Ok(())
}