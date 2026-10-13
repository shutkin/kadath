use chrono::{DateTime, Utc};
use log::{debug, info, warn};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::{DataSource, Deal, Instrument, OrderBook, OrderData, OrderDataCompact, OrderStatus, SignalCalculatorLog, Trade, TradeDirection};
use rust_decimal::Decimal;
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres, Row};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
struct Prices {
    prices_map: Arc<RwLock<HashMap<Decimal, i32>>>,
}

impl Prices {
    fn new() -> Self {
        Self {
            prices_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn init(&self, pool: &Pool<Postgres>) -> EmptyResult {
        info!("Fetch prices");
        let rows = sqlx::query("SELECT id, price FROM price")
            .fetch_all(pool)
            .await?;
        info!("handle {} rows", rows.len());
        let mut write = self.prices_map.write().map_err(|err| format!("{err}"))?;
        rows.iter().for_each(|row| {
            write.insert(row.get(1), row.get(0));
        });
        info!("Prices init with {} values", write.len());
        Ok(())
    }

    async fn get_id(&self, price: Decimal, pool: &Pool<Postgres>) -> Result<i32, CommonError> {
        if let Some(id) = self.prices_map.read().map_err(|err| format!("{err}"))?.get(&price) {
            Ok(*id)
        } else {
            debug!("Insert new price value {price}");
            let id_row = sqlx::query("INSERT INTO price (price) VALUES ($1) RETURNING id")
                .bind(price)
                .fetch_one(pool)
                .await?;
            let id = id_row.get(0);
            self.prices_map.write().map_err(|err| format!("{err}"))?.insert(price, id);
            Ok(id)
        }
    }

    fn get_prices(&self, map: &mut HashMap<i32, Decimal>) {
        map.clear();
        if let Ok(read) = self.prices_map.read() {
            for (price, id) in read.iter() {
                map.insert(*id, *price);
            }
        }
    }
}

#[derive(Clone)]
pub struct Db {
    pool: Pool<Postgres>,
    prices: Prices,
}

impl Db {
    pub async fn new(url: &str) -> Result<Self, CommonError> {
        let pool = PgPoolOptions::new().max_connections(8).connect(url).await?;
        let prices = Prices::new();
        prices.init(&pool).await?;
        Ok(Self { pool, prices })
    }

    pub async fn get_instruments(&self, source: DataSource) -> Result<Vec<Instrument>, CommonError> {
        let rows = sqlx::query("SELECT id, ticker, name, lot_size, external_id  FROM instrument i WHERE source = $1")
            .bind(source.get_code())
            .fetch_all(&self.pool).await?;
        Ok(rows
            .iter()
            .map(|row| Instrument {
                id: row.get(0),
                source,
                ticker: row.get(1),
                name: row.get(2),
                lot_size: row.get(3),
                external_id: row.get(4),
            })
            .collect())
    }

    pub async fn insert_instruments(&self, instruments: &mut [Instrument]) -> Result<(), CommonError> {
        let source = instruments[0].source;
        let existing_instruments = self.get_instruments(source).await?;

        for instrument in instruments {
            if let Some(existing_instrument) = existing_instruments.iter()
                .find(|existing_instrument| existing_instrument.ticker == instrument.ticker) {
                instrument.id = existing_instrument.id;
                continue;
            }

            info!("Insert {:?}", instrument);
            let id_row = sqlx::query("INSERT INTO instrument (source, ticker, name, lot_size, external_id) VALUES ($1, $2, $3, $4, $5) RETURNING id")
                .bind(instrument.source.get_code())
                .bind(instrument.ticker.clone())
                .bind(instrument.name.clone())
                .bind(instrument.lot_size)
                .bind(instrument.external_id.clone())
                .fetch_one(&self.pool).await?;
            let id = id_row.get(0);
            instrument.id = Some(id);
        }

        Ok(())
    }

    async fn serialize_compact(&self, data: &[OrderData]) -> Result<Vec<u8>, CommonError> {
        let mut delta_vec = Vec::with_capacity(data.len());
        for data in data {
            delta_vec.push(OrderDataCompact {
                price_id: self.prices.get_id(data.price, &self.pool).await?,
                size: data.size,
            });
        }
        let res = bincode::serde::encode_to_vec(&delta_vec, bincode::config::standard())?;
        Ok(res)
    }

    pub async fn get_order_books(&self, instrument_id: i32, diapason: TimeDiapason) -> Result<Vec<OrderBook>, CommonError> {
        info!("Query order books instrument {instrument_id} {diapason:?}");
        let rows = sqlx::query("SELECT id, created, ask, bid FROM order_book WHERE instrument_id = $1 AND created BETWEEN $2 AND $3 ORDER BY id")
            .bind(instrument_id)
            .bind(diapason.from)
            .bind(diapason.to)
            .fetch_all(&self.pool).await?;

        let mut prices_map = HashMap::new();
        self.prices.get_prices(&mut prices_map);

        let result = rows.iter().filter_map(|row| {
            if let Ok(asks) = Self::deserialize_compact_data(row.get(2), &prices_map)
                && let Ok(bids) = Self::deserialize_compact_data(row.get(3), &prices_map) {
                Some(OrderBook {
                    id: Some(row.get(0)),
                    timestamp: row.get(1),
                    asks,
                    bids,
                })
            } else {
                None
            }
        }).collect();

        Ok(result)
    }

    fn deserialize_compact_data(data: Vec<u8>, prices_map: &HashMap<i32, Decimal>) -> Result<Vec<OrderData>, CommonError> {
        let (delta_vec, _): (Vec<OrderDataCompact>, usize) = bincode::serde::decode_from_slice(&data, bincode::config::standard())?;
        let mut result = Vec::with_capacity(delta_vec.len());
        for delta in delta_vec {
            if let Some(price) = prices_map.get(&delta.price_id).copied() {
                result.push(OrderData { price, size: delta.size });
            } else {
                warn!("Price ID {} not found in map", delta.price_id);
            }
        }
        Ok(result)
    }

    pub async fn insert_order_book(&self, order_book: &OrderBook, instrument: &Instrument) -> Result<(), CommonError> {
        let instrument_id = instrument
            .id
            .ok_or::<CommonError>(format!("Instrument {} has no id", instrument.ticker).into())?;

        let ask_serialized = self.serialize_compact(&order_book.asks).await?;
        let bid_serialized = self.serialize_compact(&order_book.bids).await?;

        sqlx::query(
            "INSERT INTO order_book (created, instrument_id, ask, bid) VALUES ($1, $2, $3, $4)",
        )
            .bind(order_book.timestamp)
            .bind(instrument_id)
            .bind(ask_serialized)
            .bind(bid_serialized)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn start_deal(&self, instrument_id: i32, lots: u16, direction: TradeDirection, hold: i32, cur_price: Decimal) -> Result<Deal, CommonError> {
        let open_time = Utc::now();

        let id_row = sqlx::query("INSERT INTO deal (instrument_id, lots, direction, max_hold, open_time, expected_open_price, open_status, close_status) values ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id")
            .bind(instrument_id)
            .bind(lots as i16)
            .bind(direction.to_short_string())
            .bind(hold)
            .bind(open_time)
            .bind(cur_price)
            .bind(OrderStatus::NotStarted.to_short_string())
            .bind(OrderStatus::NotStarted.to_short_string())
            .fetch_one(&self.pool).await?;
        let id = id_row.get(0);

        Ok(Deal {
            id,
            instrument_id,
            lots,
            direction,
            max_hold: hold,

            open_time,
            open_status: OrderStatus::NotStarted,
            expected_open_price: cur_price,
            open_order_id: None,

            close_time: None,
            close_status: OrderStatus::NotStarted,
            expected_close_price: None,
            close_order_id: None,
        })
    }

    pub async fn update_deal(&self, deal: &Deal) -> EmptyResult {
        sqlx::query("UPDATE deal SET open_order_id = $2, open_status = $3, close_time = $4, expected_close_price = $5, close_order_id = $6, close_status = $7  WHERE id = $1")
            .bind(deal.id)
            .bind(deal.open_order_id.clone())
            .bind(deal.open_status.to_short_string())
            .bind(deal.close_time)
            .bind(deal.expected_close_price)
            .bind(deal.close_order_id.clone())
            .bind(deal.close_status.to_short_string())
            .execute(&self.pool).await?;

        Ok(())
    }

    pub async fn insert_trade(&self, trade: &mut Trade) -> EmptyResult {
        let id_row = sqlx::query("INSERT INTO trade (order_id, executed, direction, price, quantity, json) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id")
            .bind(trade.order_id.clone())
            .bind(trade.executed)
            .bind(trade.direction.to_short_string())
            .bind(trade.price)
            .bind(trade.quantity)
            .bind(trade.json.clone())
            .fetch_one(&self.pool).await?;
        let id = id_row.get(0);
        trade.id = Some(id);
        Ok(())
    }
    
    pub async fn insert_calculator_log(&self, log: &SignalCalculatorLog) -> EmptyResult {
        sqlx::query("INSERT INTO signal_calculator (trained_on, up_score, up_params, down_score, down_params, train_time_ms) VALUES ($1, $2, $3, $4, $5, $6)")
            .bind(log.trained_on)
            .bind(log.up_score)
            .bind(log.up_params.clone())
            .bind(log.down_score)
            .bind(log.down_params.clone())
            .bind(log.train_time_ms)
            .execute(&self.pool).await?;
        Ok(())
    }

    pub async fn get_signal_calculators(&self, diapason: TimeDiapason) -> Result<Vec<SignalCalculatorLog>, CommonError> {
        let rows = sqlx::query("SELECT trained_on, up_score, up_params, down_score, down_params, train_time_ms FROM signal_calculator WHERE trained_on BETWEEN $1 AND $2")
            .bind(diapason.from)
            .bind(diapason.to)
            .fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|row| SignalCalculatorLog {
            trained_on: row.get(0),
            up_score: row.get(1),
            up_params: row.get(2),
            down_score: row.get(3),
            down_params: row.get(4),
            train_time_ms: row.get(5),
        }).collect())
    }

    pub async fn get_order_books_ids(&self, from: DateTime<Utc>, to: DateTime<Utc>, limit: i32)
                                     -> Result<Vec<i64>, CommonError> {
        let ids = sqlx::query("SELECT id FROM order_book WHERE created BETWEEN $1 AND $2 LIMIT $3")
            .bind(from)
            .bind(to)
            .bind(limit)
            .fetch_all(&self.pool).await?
            .into_iter()
            .map(|row| row.get(0))
            .collect();
        Ok(ids)
    }
    
    pub async fn get_order_books_time_diapason(&self, ids: &[i64]) -> Result<(DateTime<Utc>, DateTime<Utc>), CommonError> {
        let row = sqlx::query("SELECT min(created), max(created) FROM order_book WHERE id IN (SELECT * FROM unnest($1))")
            .bind(ids)
            .fetch_one(&self.pool).await?;
        Ok((row.get(0), row.get(1)))
    }

    pub async fn remove_order_books(&self, ids: &[i64], to: DateTime<Utc>) -> Result<(), CommonError> {
        let res = sqlx::query("DELETE FROM order_book WHERE id IN (SELECT * FROM unnest($1)) AND created < $2")
            .bind(ids)
            .bind(to)
            .execute(&self.pool).await?;
        info!("removed order books {}", res.rows_affected());
        Ok(())
    }
}
