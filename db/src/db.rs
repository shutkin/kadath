use chrono::{DateTime, Utc};
use log::{debug, info, warn};
use model::common::{CommonError, EmptyResult, TimeDiapason};
use model::utils::handle_delta;
use model::{Deal, Instrument, OrderBook, OrderData, OrderDataDelta, SignalCalculatorLog, Trade, TradeDirection};
use rust_decimal::Decimal;
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres, Row};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub enum QueryAsksOrBids {
    ASKS,
    BIDS,
    BOTH,
}

impl QueryAsksOrBids {
    fn is_asks(&self) -> bool {
        match self {
            QueryAsksOrBids::ASKS => true,
            QueryAsksOrBids::BIDS => false,
            QueryAsksOrBids::BOTH => true,
        }
    }

    fn is_bids(&self) -> bool {
        match self {
            QueryAsksOrBids::ASKS => false,
            QueryAsksOrBids::BIDS => true,
            QueryAsksOrBids::BOTH => true,
        }
    }
}

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
            info!("Insert new price value {price}");
            let id_row = sqlx::query("INSERT INTO price (price) VALUES ($1) RETURNING id")
                .bind(price)
                .fetch_one(pool)
                .await?;
            let id = id_row.get(0);
            self.prices_map.write().map_err(|err| format!("{err}"))?.insert(price, id);
            Ok(id)
        }
    }

    async fn get_prices(&self, map: &mut HashMap<i32, Decimal>) {
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

    pub async fn get_instruments(&self) -> Result<Vec<Instrument>, CommonError> {
        let rows = sqlx::query("SELECT i.id, i.ticker, i.name, i.lot_size FROM instrument i")
            .fetch_all(&self.pool).await?;
        Ok(rows
            .iter()
            .map(|row| Instrument {
                id: row.get(0),
                ticker: row.get(1),
                name: row.get(2),
                lot_size: row.get(3),
            })
            .collect())
    }

    pub async fn insert_instruments(&self, instruments: &mut [Instrument]) -> Result<(), CommonError> {
        let existing_instruments = self.get_instruments().await?;

        for instrument in instruments {
            if let Some(existing_instrument) = existing_instruments.iter()
                .find(|existing_instrument| existing_instrument.ticker == instrument.ticker) {
                instrument.id = existing_instrument.id;
                continue;
            }

            info!("Insert {:?}", instrument);
            let id_row = sqlx::query("INSERT INTO instrument (ticker, name, lot_size) VALUES ($1, $2, $3) RETURNING id")
                .bind(instrument.ticker.clone())
                .bind(instrument.name.clone())
                .bind(instrument.lot_size)
                .fetch_one(&self.pool).await?;
            let id = id_row.get(0);
            instrument.id = Some(id);
        }

        Ok(())
    }

    async fn serialize_delta(&self, delta: &[OrderData]) -> Result<Vec<u8>, CommonError> {
        let mut delta_vec = Vec::with_capacity(delta.len());
        for data in delta {
            delta_vec.push(OrderDataDelta {
                price_id: self.prices.get_id(data.price, &self.pool).await?,
                size: data.size,
            });
        }
        let res = bincode::serde::encode_to_vec(&delta_vec, bincode::config::standard())?;
        Ok(res)
    }

    fn deserialize_and_handle(data: &[u8], cur_data: &mut Vec<OrderData>, prices_map: &HashMap<i32, Decimal>) -> Result<bool, CommonError> {
        let (delta_vec, _): (Vec<OrderDataDelta>, usize) = bincode::serde::decode_from_slice(data, bincode::config::standard())?;
        let mut deltas = Vec::with_capacity(delta_vec.len());
        for delta in delta_vec {
            if let Some(price) = prices_map.get(&delta.price_id).copied() {
                deltas.push(OrderData { price, size: delta.size });
            } else {
                warn!("Price ID {} not found in map", delta.price_id);
                return Ok(false);
            }
        }
        handle_delta(&deltas, cur_data);
        Ok(true)
    }

    pub async fn get_order_books(
        &self,
        instrument_id: Option<i32>,
        diapason: TimeDiapason,
        ab: QueryAsksOrBids,
    ) -> Result<Vec<OrderBook>, CommonError> {
        let instrument_id = instrument_id.ok_or::<CommonError>("Instrument has no id".into())?;

        info!("Query order books instrument {instrument_id} {diapason:?}");
        let rows = sqlx::query("SELECT id, created, ask_delta, bid_delta FROM order_book WHERE instrument_id = $1 AND created BETWEEN $2 AND $3 ORDER BY created")
            .bind(instrument_id)
            .bind(diapason.from)
            .bind(diapason.to)
            .fetch_all(&self.pool).await?;

        let full_ids = rows
            .iter()
            .filter(|row| {
                let v: Option<&[u8]> = row.get(2);
                v.is_none()
            })
            .map(|row| {
                let id: i64 = row.get(0);
                id
            })
            .collect::<Vec<i64>>();
        let mut full_asks = if ab.is_asks() {
            self.get_order_data(&full_ids, "order_book_ask").await?
        } else {
            HashMap::default()
        };
        let mut full_bids = if ab.is_bids() {
            self.get_order_data(&full_ids, "order_book_bid").await?
        } else {
            HashMap::default()
        };

        let mut prices_map = HashMap::new();
        self.prices.get_prices(&mut prices_map).await;
        let mut cur_asks = Vec::new();
        let mut cur_bids = Vec::new();

        let mut result = Vec::new();
        let mut prev_percentage = 0;
        for (i, row) in rows.iter().enumerate() {
            let id = row.get(0);
            let asks_encoded: Option<Vec<u8>> = row.get(2);
            let bids_encoded: Option<Vec<u8>> = row.get(3);
            let mut order_book = OrderBook {
                id: Some(id),
                timestamp: row.get(1),
                asks: Vec::default(),
                bids: Vec::default(),
            };

            if ab.is_asks() {
                if let Some(asks) = full_asks.remove(&id) {
                    cur_asks = asks.clone();
                    order_book.asks = asks;
                } else if let Some(asks_encoded) = asks_encoded {
                    loop {
                        if Self::deserialize_and_handle(&asks_encoded, &mut cur_asks, &prices_map)? {
                            break;
                        }
                        self.prices.init(&self.pool).await?;
                        self.prices.get_prices(&mut prices_map).await;
                    }
                    order_book.asks = cur_asks.clone();
                }
            }
            if ab.is_bids() {
                if let Some(bids) = full_bids.remove(&id) {
                    cur_bids = bids.clone();
                    order_book.bids = bids;
                } else if let Some(bids_encoded) = bids_encoded {
                    loop {
                        if Self::deserialize_and_handle(&bids_encoded, &mut cur_bids, &prices_map)? {
                            break;
                        }
                        self.prices.init(&self.pool).await?;
                        self.prices.get_prices(&mut prices_map).await;
                    }
                    order_book.bids = cur_bids.clone();
                }
            }

            if !order_book.asks.is_empty() || !order_book.bids.is_empty() {
                result.push(order_book);
            }

            let percentage = i * 100 / rows.len();
            if percentage > prev_percentage {
                debug!("Process {percentage}%");
                prev_percentage = percentage;
            }
        }
        Ok(result)
    }

    async fn get_order_data(
        &self,
        order_book_ids: &[i64],
        table_name: &str,
    ) -> Result<HashMap<i64, Vec<OrderData>>, CommonError> {
        let rows = sqlx::query(
            &format!("SELECT o.order_book_id, p.price, o.size FROM {table_name} o JOIN price p ON o.price_id = p.id WHERE order_book_id IN (SELECT * FROM unnest($1))"))
            .bind(order_book_ids)
            .fetch_all(&self.pool).await?;
        let mut result: HashMap<i64, Vec<OrderData>> = HashMap::with_capacity(order_book_ids.len());
        for row in rows {
            let order_book_id = row.get(0);
            let data = OrderData {
                price: row.get(1),
                size: row.get(2),
            };
            if let Some(vec) = result.get_mut(&order_book_id) {
                vec.push(data);
            } else {
                result.insert(order_book_id, vec![data]);
            }
        }
        Ok(result)
    }

    async fn insert_order_book(
        &self,
        order_book: &OrderBook,
        instrument: &Instrument,
    ) -> Result<i64, CommonError> {
        let instrument_id = instrument
            .id
            .ok_or::<CommonError>(format!("Instrument {} has no id", instrument.ticker).into())?;
        let res = sqlx::query(
            "INSERT INTO order_book (created, instrument_id) VALUES ($1, $2) RETURNING id",
        )
        .bind(order_book.timestamp)
        .bind(instrument_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(res.get(0))
    }

    pub async fn insert_order_book_delta(
        &self,
        instrument: &Instrument,
        timestamp: DateTime<Utc>,
        ask_delta: &[OrderData],
        bid_delta: &[OrderData],
    ) -> Result<(), CommonError> {
        let instrument_id = instrument
            .id
            .ok_or::<CommonError>(format!("Instrument {} has no id", instrument.ticker).into())?;

        let ask_serialized = self.serialize_delta(ask_delta).await?;
        let bid_serialized = self.serialize_delta(bid_delta).await?;

        sqlx::query("INSERT INTO order_book (created, instrument_id, ask_delta, bid_delta) VALUES ($1, $2, $3, $4)")
            .bind(timestamp)
            .bind(instrument_id)
            .bind(ask_serialized)
            .bind(bid_serialized)
            .execute(&self.pool).await?;
        Ok(())
    }

    pub async fn insert_order_book_full(
        &self,
        order_book: &OrderBook,
        instrument: &Instrument,
    ) -> Result<(), CommonError> {
        let id = self.insert_order_book(order_book, instrument).await?;

        let mut asks_ids = Vec::with_capacity(order_book.asks.len());
        let mut asks_sizes = Vec::with_capacity(order_book.asks.len());
        for ask in &order_book.asks {
            let price_id = self.prices.get_id(ask.price, &self.pool).await?;
            asks_ids.push(price_id);
            asks_sizes.push(ask.size);
        }
        let mut bids_ids = Vec::with_capacity(order_book.bids.len());
        let mut bids_sizes = Vec::with_capacity(order_book.bids.len());
        for bid in &order_book.bids {
            let price_id = self.prices.get_id(bid.price, &self.pool).await?;
            bids_ids.push(price_id);
            bids_sizes.push(bid.size);
        }

        sqlx::query("INSERT INTO order_book_ask (order_book_id, price_id, size) SELECT * FROM unnest($1, $2, $3)")
            .bind(vec![id; order_book.asks.len()])
            .bind(asks_ids)
            .bind(asks_sizes)
            .execute(&self.pool).await?;

        sqlx::query("INSERT INTO order_book_bid (order_book_id, price_id, size) SELECT * FROM unnest($1, $2, $3)")
            .bind(vec![id; order_book.bids.len()])
            .bind(bids_ids)
            .bind(bids_sizes)
            .execute(&self.pool).await?;

        Ok(())
    }
    
    pub async fn start_deal(&self, instrument_id: i32, direction: TradeDirection) -> Result<Deal, CommonError> {
        let started = Utc::now();
        let id_row = sqlx::query("INSERT INTO spread_deal (started, instrument_id, direction) VALUES ($1, $2, $3)")
            .bind(started)
            .bind(instrument_id)
            .bind(direction.to_short_string())
            .fetch_one(&self.pool).await?;
        let id = id_row.get(0);
        Ok(Deal {
            id,
            instrument_id,
            started,
            direction,
            open_order_id: None,
            close_order_id: None,
        })
    }
    
    pub async fn update_deal(&self, deal_id: i32, open_order_id: Option<String>, close_order_id: Option<String>) -> EmptyResult {
        if let Some(open_order_id) = open_order_id {
            sqlx::query("UPDATE spread_deal SET open_order_id = $1 WHERE id = $2")
                .bind(open_order_id)
                .bind(deal_id)
                .execute(&self.pool).await?;
        }
        if let Some(close_order_id) = close_order_id {
            sqlx::query("UPDATE spread_deal SET close_order_id = $1 WHERE id = $2")
                .bind(close_order_id)
                .bind(deal_id)
                .execute(&self.pool).await?;
        }
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
        sqlx::query("INSERT INTO signal_calculator (trained_on, up_score, up_params, down_score, down_params) VALUES ($1, $2, $3, $4, $5)")
            .bind(log.trained_on)
            .bind(log.up_score)
            .bind(log.up_params.clone())
            .bind(log.down_score)
            .bind(log.down_params.clone())
            .execute(&self.pool).await?;
        Ok(())
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
        let res = sqlx::query("DELETE FROM order_book_ask WHERE order_book_id IN (SELECT * FROM unnest($1))")
            .bind(ids)
            .execute(&self.pool).await?;
        info!("removed order books asks {}", res.rows_affected());

        let res = sqlx::query("DELETE FROM order_book_bid WHERE order_book_id IN (SELECT * FROM unnest($1))")
            .bind(ids)
            .execute(&self.pool).await?;
        info!("removed order books bids {}", res.rows_affected());

        let res = sqlx::query("DELETE FROM order_book WHERE id IN (SELECT * FROM unnest($1)) AND created < $2")
            .bind(ids)
            .bind(to)
            .execute(&self.pool).await?;
        info!("removed order books {}", res.rows_affected());
        Ok(())
    }
}
