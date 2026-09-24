use chrono::{TimeZone, Utc};
use futures_util::{SinkExt, StreamExt};
use jsonwebtoken::dangerous::insecure_decode_claims;
use log::{error, info, warn};
use model::common::{CommonError, EmptyResult};
use model::events::{OrderBookDeltaEvent, OrderBookEvent};
use model::{Instrument, OrderBook, OrderData};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use uuid::Uuid;
use yawc::{Frame, OpCode, WebSocket};

const ACCESS_TOKEN_URL: &str = "https://oauthdev.alor.ru/refresh";
const API_ROOT: &str = "https://apidev.alor.ru/md/v2";
const WS_URL: &str = "wss://apidev.alor.ru/ws";

const INSTRUMENTS_LIMIT: usize = 100;

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: i64, // Expiration time as a Unix timestamp
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AccessTokenRequest {
    token: String,
    allowed_portfolios: Vec<String>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct AccessTokenResponse {
    access_token: Option<String>,
}

#[derive(Deserialize, Debug)]
struct InstrumentResponse {
    desc: String,
    t: String,
    lot: i32,
}

#[derive(Serialize)]
struct SubscribeRequest {
    opcode: String,
    exchange: String,
    code: String,
    depth: i32,
    format: String,
    frequency: i32,
    guid: String,
    token: String,
}

#[derive(Deserialize, Debug)]
struct OrderBookMessage {
    /// Блок данных от информационного канала
    data: OrderBookData,
    guid: String,
}

#[derive(Deserialize, Debug)]
struct OrderBookData {
    /// Данные об асках
    a: Vec<OrderBookPosition>,
    /// Данные о бидах
    b: Vec<OrderBookPosition>,
    /// Время (UTC) в формате Unix Time Milliseconds
    t: i64,
    /// True — для данных из снепшота, то есть из истории. False — для новых событий
    h: bool,
}

#[derive(Deserialize, Debug)]
struct OrderBookPosition {
    /// Цена
    #[serde(deserialize_with = "rust_decimal::serde::arbitrary_precision::deserialize")]
    p: Decimal,
    /// Объём
    #[serde(deserialize_with = "rust_decimal::serde::arbitrary_precision::deserialize")]
    v: Decimal,
}

#[derive(Default)]
pub struct Alor {
    access_token: Arc<RwLock<String>>,
}

impl Alor {
    pub async fn get_instruments(&self) -> Result<Vec<Instrument>, CommonError> {
        let token = self.get_actual_access_token().await?;

        let client = reqwest::Client::builder()
            .build()?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Accept", "application/json".parse()?);
        headers.insert("Authorization", format!("Bearer {token}").parse()?);

        let mut all_instruments = Vec::new();
        let mut offset = 0;
        loop {
            let response = client.request(reqwest::Method::GET, format!("{API_ROOT}/Securities"))
                .query(&[
                    ("limit", format!("{INSTRUMENTS_LIMIT}").as_str()),
                    ("offset", format!("{offset}").as_str()),
                    ("sector", "FORTS"),
                    ("exchange", "MOEX"),
                    ("format", "Slim")
                ])
                .headers(headers.clone())
                .send().await?;
            match response.json::<Vec<InstrumentResponse>>().await {
                Err(err) => {
                    error!("{err}");
                    break;
                },
                Ok(instruments) => {
                    for instrument in &instruments {
                        info!("{instrument:?}");
                    }
                    let len = instruments.len();
                    all_instruments.extend(instruments);
                    if len < INSTRUMENTS_LIMIT {
                        break;
                    }
                    offset += len;
                    info!("{} instruments", all_instruments.len());
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }

        Ok(all_instruments.into_iter().map(|i| Instrument {
            id: None,
            ticker: i.desc,
            name: i.t,
            lot_size: i.lot as i16,
        }).collect())
    }

    pub async fn listen(
        &self,
        instruments: &[Instrument],
        order_book_sender: mpsc::Sender<OrderBookEvent>,
        order_book_delta_sender: mpsc::Sender<OrderBookDeltaEvent>,
    ) -> EmptyResult {
        let token = self.get_actual_access_token().await?;

        let mut ws = WebSocket::connect(WS_URL.parse()?)
            .with_options(yawc::Options::default().with_compression_level(yawc::CompressionLevel::best()))
            .await?;

        info!("Listening on {}", instruments.iter().map(|i| i.ticker.clone()).collect::<Vec<_>>().join(", "));
        let mut guid_to_instrument_id = HashMap::new();
        for instrument in instruments {
            let guid = Uuid::new_v4().to_string();
            guid_to_instrument_id.insert(guid.clone(), instrument.id.unwrap());
            let rq = SubscribeRequest {
                opcode: "OrderBookGetAndSubscribe".to_string(),
                exchange: "MOEX".to_string(),
                code: instrument.ticker.clone(),
                depth: 50,
                format: "Slim".to_string(),
                frequency: 50,
                guid: guid.clone(),
                token: token.clone(),
            };
            let json = serde_json::to_string(&rq)?;
            ws.send(Frame::text(json)).await?;
        }

        let mut context = ListenerContext {
            order_books: HashMap::new(),
            order_book_sender,
            order_book_delta_sender,
        };

        while let Some(frame) = ws.next().await {
            match frame.opcode() {
                OpCode::Text => {
                    match serde_json::from_str::<OrderBookMessage>(frame.as_str()) {
                        Ok(msg) => {
                            if let Some(id) = guid_to_instrument_id.get(&msg.guid) {
                                if let Err(err) = handle_order_book(&mut context, &msg.data, *id).await {
                                    error!("Failed to handle {msg:?} {err:?}");
                                }
                            } else {
                                warn!("Unknown guid {}", msg.guid);
                            }
                        },
                        Err(err) => {
                            error!("{err}: {}", frame.as_str());
                        }
                    }
                }
                OpCode::Close => {
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn check_token(&self) -> Option<String> {
        if let Ok(token) = self.access_token.read()
            && !token.is_empty()
            && let Ok(claims) = insecure_decode_claims::<Claims>(token.as_str())
            && claims.exp > Utc::now().timestamp() {
            Some(token.clone())
        } else {
            None
        }
    }

    async fn get_actual_access_token(&self) -> Result<String, CommonError> {
        if let Some(access_token) = self.check_token() {
            Ok(access_token)
        } else {
            self.refresh_access_token().await
        }
    }

    async fn refresh_access_token(&self) -> Result<String, CommonError> {
        let refresh_token = env::var("ALOR_REFRESH_TOKEN")?;

        let client = reqwest::Client::builder()
            .build()?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        headers.insert("Accept", "application/json".parse()?);

        let data = AccessTokenRequest {
            token: refresh_token,
            allowed_portfolios: vec!["D00013".to_string()],
        };
        let request = client.request(reqwest::Method::POST, ACCESS_TOKEN_URL)
            .headers(headers)
            .json(&data);

        let response = request.send().await?;
        let resp_object = response.json::<AccessTokenResponse>().await?;
        info!("'{resp_object:?}'");

        if let Some(token) = resp_object.access_token &&
            let Ok(mut lock) = self.access_token.write() {
            *lock = token.clone();
            return Ok(token)
        }

        Err("Unknown error".into())
    }
}

struct ListenerContext {
    order_books: HashMap<i32, OrderBook>,
    order_book_sender: mpsc::Sender<OrderBookEvent>,
    order_book_delta_sender: mpsc::Sender<OrderBookDeltaEvent>,
}

fn convert_order_book(api_order_book: &OrderBookData) -> Option<OrderBook> {
    if api_order_book.h {
        None
    } else {
        Some(OrderBook {
            id: None,
            timestamp: Utc.timestamp_millis_opt(api_order_book.t).single().unwrap_or(Utc::now()),
            asks: api_order_book.a.iter().map(|a| OrderData { price: a.p, size: a.v }).collect(),
            bids: api_order_book.b.iter().map(|b| OrderData { price: b.p, size: b.v }).collect(),
        })
    }
}

async fn handle_order_book(context: &mut ListenerContext, api_order_book: &OrderBookData, instrument_id: i32) -> EmptyResult {
    if let Some(new_order_book) = convert_order_book(api_order_book) {
        let mut prev_order_book = OrderBook {
            id: None,
            timestamp: new_order_book.timestamp,
            asks: Vec::new(),
            bids: Vec::new(),
        };
        if let Some(order_books) = context.order_books.get(&instrument_id) {
            prev_order_book = order_books.clone();
        }
        if prev_order_book.asks.is_empty() || prev_order_book.bids.is_empty() {
            if let Err(err) = context.order_book_sender.send(OrderBookEvent {
                instrument_id,
                order_book: new_order_book.clone(),
            }).await {
                error!("Failed to send order book event: {err}");
            }
        } else {
            let ask_delta = calculate_delta(&prev_order_book.asks, &new_order_book.asks);
            let bid_delta = calculate_delta(&prev_order_book.bids, &new_order_book.bids);
            if let Err(err) = context.order_book_delta_sender.send(OrderBookDeltaEvent {
                instrument_id,
                timestamp: new_order_book.timestamp,
                asks: ask_delta,
                bids: bid_delta,
            }).await {
                error!("Failed to send order book delta event: {err}");
            }
        }
        context.order_books.insert(instrument_id, new_order_book);
    }
    Ok(())
}

fn calculate_delta(prev_orders: &[OrderData], new_orders: &[OrderData]) -> Vec<OrderData> {
    let mut delta = new_orders.iter()
        .filter(|new| {
            if let Some(prev) = prev_orders.iter().find(|prev| prev.price == new.price) {
                prev.size != new.size
            } else {
                true
            }
        })
        .cloned()
        .collect::<Vec<_>>();
    delta.extend(prev_orders.iter()
        .filter_map(|prev| {
            if new_orders.iter().any(|new| new.price == prev.price) {
                None
            } else {
                Some(OrderData {
                    price: prev.price,
                    size: Decimal::ZERO,
                })
            }
        }));
    delta
}
