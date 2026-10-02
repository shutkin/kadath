use abstract_broker::{calculate_delta, AbstractBroker};
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use futures_util::{SinkExt, StreamExt};
use jsonwebtoken::dangerous::insecure_decode_claims;
use log::{error, info, warn};
use model::common::{CommonError, EmptyResult};
use model::events::{OrderBookDeltaEvent, OrderBookEvent, OrderEvent};
use model::{DataSource, Instrument, OrderBook, OrderData, Trade, TradeDirection};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use uuid::Uuid;
use yawc::{Frame, MaybeTlsStream, OpCode, WebSocket};

const ACCESS_TOKEN_URL: &str = "https://oauthdev.alor.ru/refresh";
const API_ROOT: &str = "https://apidev.alor.ru/md/v2";
const WS_SUBSCRIBE_URL: &str = "wss://apidev.alor.ru/ws";
const WS_COMMAND_URL: &str = "wss://apidev.alor.ru/cws";

const INSTRUMENTS_LIMIT: usize = 100;

const PORTFOLIO: &str = "7500013";

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
struct OrderBookSubscribeRequest {
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
    v: i32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TradeSubscribeRequest {
    opcode: String,
    exchange: String,
    portfolio: String,
    skip_history: bool,
    format: String,
    guid: String,
    token: String,
}

#[derive(Deserialize, Debug)]
struct TradeMessage {
    data: TradeData,
    guid: String,
}

#[derive(Deserialize, Debug)]
struct TradeData {
    /// Уникальный идентификатор сделки
    id: String,
    /// Уникальный идентификатор заявки
    eid: String,
    /// Пользовательский комментарий к заявке. Максимальная длина комментария — 100 символов
    cmt: Option<String>,
    /// Тикер (Код финансового инструмента). [N/A] используется, если symbol не определен
    sym: String,
    ///  Пара Биржа:Тикер. Пример: "MOEX:ALRS-3.24"
    tic: String,
    /// Биржа: MOEX — Московская Биржа, SPBX — СПБ Биржа
    ex: String,
    /// Дата и время (UTC) закрытия заявки
    d: DateTime<Utc>,
    /// Количество (штуки)
    q: i32,
    /// Количество (лоты)
    qb: i32,
    /// Цена
    px: Decimal,
    /// Начислено (НКД)
    ai: Decimal,
    /// Направление сделки: buy — Покупка, sell — Продажа
    s: String,
    /// True — для данных из снепшота, то есть из истории. False — для новых событий
    h: bool,
    /// Суммарная комиссия (null для Срочного рынка)
    cms: Option<String>,
    /// Объём сделки в валюте расчётов, определённый по действовавшей на момент заключения цене лота
    v: Decimal,
    /// Только для фондового рынка MOEX и SPBX. Точная стоимость сделки в валюте расчётов, равная произведению цены одной штуки (price/px) в валюте расчётов на объём лота (qtyUnits/q)
    val: Decimal,
}

#[derive(Serialize)]
struct AuthRequest {
    opcode: String,
    guid: String,
    token: String,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CreateOrderRequest {
    opcode: String,
    guid: String,
    /// Направление сделки: buy — Покупка, sell — Продажа
    side: String,
    /// Количество (лоты)
    quantity: i32,
    #[serde(with = "rust_decimal::serde::arbitrary_precision")]
    price: Decimal,
    comment: Option<String>,
    user: CreateOrderUser,
    instrument: CreateOrderInstrument,
    /// ImmediateOrCancel — Исполнение немедленно на весь доступный объём встречных заявок. Неисполненный остаток отменяется
    time_in_force: String,
    /// Флаг, подтверждающий согласие клиента с начальным уровнем риска (КНУР) на выставление или изменение заявки, исполнение которой может привести к возникновению или увеличению непокрытой позиции.
    /// Флаг не переносится между запросами — в каждом запросе, требующем подтверждения, должен быть свой флаг allowMargin: true
    allow_margin: bool,
}

#[derive(Serialize, Debug)]
struct CreateOrderUser {
    /// Идентификатор клиентского портфеля
    portfolio: String,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
struct CreateOrderInstrument {
    /// Тикер (Код финансового инструмента)
    symbol: String,
    /// Биржа: MOEX — Московская Биржа, SPBX — СПБ Биржа
    exchange: String,
    /// RFUD
    instrument_group: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateOrderResponse {
    request_guid: String,
    http_code: i32,
    message: Option<String>,
    order_number: i64,
}

#[derive(Default)]
pub struct Alor {
    access_token: RwLock<String>,
    access_token_updating: AtomicBool,
}

#[async_trait]
impl AbstractBroker for Alor {
    async fn get_instruments(&self) -> Result<Vec<Instrument>, CommonError> {
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
            source: DataSource::Alor,
            ticker: i.desc,
            name: i.t,
            lot_size: i.lot as i16,
            external_id: None,
        }).collect())
    }

    async fn listen_trades(&self, order_book_sender: mpsc::Sender<Trade>) -> EmptyResult {
        let token = self.get_actual_access_token().await?;

        let mut ws = WebSocket::connect(WS_SUBSCRIBE_URL.parse()?)
            .with_options(yawc::Options::default().with_compression_level(yawc::CompressionLevel::best()))
            .await?;

        let rq = TradeSubscribeRequest {
            opcode: "TradesGetAndSubscribeV2".to_string(),
            exchange: "MOEX".to_string(),
            portfolio: PORTFOLIO.to_string(),
            skip_history: true,
            format: "Slim".to_string(),
            guid: Uuid::new_v4().to_string(),
            token,
        };
        let json = serde_json::to_string(&rq)?;
        ws.send(Frame::text(json)).await?;

        while let Some(frame) = ws.next().await {
            match frame.opcode() {
                OpCode::Text => {
                    match serde_json::from_str::<TradeMessage>(frame.as_str()) {
                        Ok(msg) => {
                            let trade = Trade {
                                id: None,
                                order_id: msg.data.eid,
                                executed: msg.data.d,
                                direction: if msg.data.s == "buy" {TradeDirection::Buy} else {TradeDirection::Sell},
                                quantity: msg.data.q as i16,
                                price: msg.data.px,
                                json: frame.as_str().to_string(),
                            };
                            if let Err(err) = order_book_sender.send(trade).await {
                                error!("Failed to send trade event {err}");
                            }
                        },
                        Err(err) => {
                            info!("{err}: {}", frame.as_str());
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

    async fn listen_order_books(
        &self,
        instruments: &[Instrument],
        order_book_sender: mpsc::Sender<OrderBookEvent>,
        order_book_delta_sender: mpsc::Sender<OrderBookDeltaEvent>,
    ) -> EmptyResult {
        let token = self.get_actual_access_token().await?;

        let mut ws = WebSocket::connect(WS_SUBSCRIBE_URL.parse()?)
            .with_options(yawc::Options::default().with_compression_level(yawc::CompressionLevel::best()))
            .await?;

        info!("Listening on {}", instruments.iter().map(|i| i.ticker.clone()).collect::<Vec<_>>().join(", "));
        let mut guid_to_instrument_id = HashMap::new();
        for instrument in instruments {
            let guid = Uuid::new_v4().to_string();
            guid_to_instrument_id.insert(guid.clone(), instrument.id.unwrap());
            let rq = OrderBookSubscribeRequest {
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
                            info!("{err}: {}", frame.as_str());
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

    async fn make_order(&self, _instrument: &Instrument, _direction: TradeDirection, _quantity: u16) -> Result<String, CommonError> {
        Err("Not implemented yet".into())
    }
}

impl Alor {
    pub async fn client_positions(&self) -> Result<String, CommonError> {
        let token = self.get_actual_access_token().await?;

        let client = reqwest::Client::builder()
            .build()?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Accept", "application/json".parse()?);
        headers.insert("Authorization", format!("Bearer {token}").parse()?);

        let response = client.request(reqwest::Method::GET, format!("{API_ROOT}/Clients/P052863/positions"))
            .headers(headers).send().await?;
        let text = response.text().await?;
        Ok(text)
    }

    async fn send_auth_command(ws: &mut WebSocket<MaybeTlsStream<TcpStream>>, token: String) {
        let rq = AuthRequest {
            opcode: "authorize".to_string(),
            guid: Uuid::new_v4().to_string(),
            token,
        };
        if let Ok(json) = serde_json::to_string(&rq) {
            if let Err(err) = ws.send(Frame::text(json)).await {
                error!("Failed to send authorize message: {err}");
            } else {
                info!("Successfully sent authorize message");
            }
        }
    }

    async fn handle_command(
        event: OrderEvent,
        ws: &mut WebSocket<MaybeTlsStream<TcpStream>>,
    ) {
        let guid = Uuid::new_v4().to_string();
        let symbol = if event.ticker == "GLZ6" {"GOLD-12.26".to_string()} else {event.ticker.clone()};
        let rq = CreateOrderRequest {
            opcode: "create:limit".to_string(),
            comment: None,
            user: CreateOrderUser { portfolio: PORTFOLIO.to_string() },
            instrument: CreateOrderInstrument { exchange: "MOEX".to_string(), symbol, instrument_group: "RFUD".to_string() },
            side: event.direction.to_full_string(),
            quantity: event.quantity,
            price: event.expected_price,
            time_in_force: "OneDay".to_string(),
            allow_margin: true,
            guid,
        };
        if let Ok(json) = serde_json::to_string(&rq) {
            info!("Send command {json}");
            if let Err(err) = ws.send(Frame::text(json)).await {
                error!("Error sending command: {err}");
                return;
            }
        } else {
            error!("Failed to serialize event: {rq:?}");
            return;
        }
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
        let mut wait_count = 0;
        while self.access_token_updating.load(Ordering::Relaxed) && wait_count < 1000 {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            wait_count += 1;
        }

        if wait_count > 0
            && let Ok(token) = self.access_token.read()
            && !token.is_empty() {
            return Ok(token.clone());
        }
        self.access_token_updating.store(true, Ordering::Relaxed);

        match Self::request_token().await {
            Ok(resp_object) => {
                if let Some(token) = resp_object.access_token &&
                    let Ok(mut lock) = self.access_token.write() {
                    *lock = token.clone();
                    self.access_token_updating.store(false, Ordering::Relaxed);
                    return Ok(token)
                }
            },
            Err(err) => {
                self.access_token_updating.store(false, Ordering::Relaxed);
                return Err(err)
            }
        }

        self.access_token_updating.store(false, Ordering::Relaxed);
        Err("Unknown error".into())
    }

    async fn request_token() -> Result<AccessTokenResponse, CommonError> {
        let refresh_token = env::var("ALOR_REFRESH_TOKEN")?;

        let client = reqwest::Client::builder()
            .build()?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        headers.insert("Accept", "application/json".parse()?);

        let data = AccessTokenRequest {
            token: refresh_token,
            allowed_portfolios: vec![PORTFOLIO.to_string()],
        };
        let request = client.request(reqwest::Method::POST, ACCESS_TOKEN_URL)
            .headers(headers)
            .json(&data);

        let response = request.send().await?;
        let resp_object = response.json::<AccessTokenResponse>().await?;
        info!("'{resp_object:?}'");
        Ok(resp_object)
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

