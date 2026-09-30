use chrono::{DateTime, TimeZone, Utc};
use log::{error, info, warn};
use model::common::{CommonError, EmptyResult};
use model::events::{OrderBookDeltaEvent, OrderBookEvent};
use model::{DataSource, Instrument, OrderBook, OrderData, Trade};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::ops::Add;
use async_trait::async_trait;
use t_invest_sdk::TInvestSdk;
use t_invest_sdk::api::market_data_response::Payload;
use t_invest_sdk::api::{market_data_request, InstrumentStatus, InstrumentsRequest, MarketDataRequest, OrderBookInstrument, OrderBookType, PingRequest, Quotation, SubscribeOrderBookRequest, SubscribeTradesRequest, SubscriptionAction, TradeInstrument, TradeSourceType, TradeDirection};
use tokio::sync::mpsc;
use tokio::sync::mpsc::Sender;
use tokio_stream::wrappers::UnboundedReceiverStream;
use abstract_broker::BrokerListener;

#[derive(Clone)]
pub struct TInvest {
    sdk: TInvestSdk,
}

#[async_trait]
impl BrokerListener for TInvest {
    async fn get_instruments(&self) -> Result<Vec<Instrument>, CommonError> {
        let futures_response = self.sdk
            .instruments()
            .futures(InstrumentsRequest {
                instrument_status: Some(InstrumentStatus::Base as i32),
                instrument_exchange: None,
            }).await?
            .into_inner();
        info!("Got {} futures", futures_response.instruments.len());
        let instruments = futures_response.instruments.into_iter()
            .filter(|future| future.api_trade_available_flag && future.short_enabled_flag)
            .map(|future| Instrument {
                id: None,
                source: DataSource::TInvest,
                ticker: future.ticker,
                external_id: Some(future.uid),
                name: future.name,
                lot_size: future.lot as i16,
            }).collect();
        Ok(instruments)
    }

    async fn listen_trades(&self, trade_sender: Sender<Trade>) -> EmptyResult {
        todo!()
    }

    async fn listen_order_books(&self, instruments: &[Instrument], order_book_sender: Sender<OrderBookEvent>, order_book_delta_sender: Sender<OrderBookDeltaEvent>) -> EmptyResult {
        let mut uid_id_map = HashMap::with_capacity(instruments.len());
        for instrument in instruments {
            if let Some(uid) = &instrument.external_id && let Some(id) = instrument.id {
                uid_id_map.insert(uid.clone(), id);
            }
        }
        let subs = uid_id_map.keys()
            .map(|uid| {
                OrderBookInstrument {
                    instrument_id: uid.clone(),
                    depth: 50,
                    order_book_type: OrderBookType::OrderbookTypeAll as i32,
                    figi: "".to_string(),
                }
            })
            .collect::<Vec<_>>();
        let request = MarketDataRequest {
            payload: Some(market_data_request::Payload::SubscribeOrderBookRequest(
                SubscribeOrderBookRequest {
                    subscription_action: SubscriptionAction::Subscribe as i32,
                    instruments: subs,
                },
            )),
        };
        let (tx, rx) = mpsc::unbounded_channel::<MarketDataRequest>();
        tx.send(request)?;

        let response = self.sdk
            .market_data_stream()
            .market_data_stream(UnboundedReceiverStream::new(rx))
            .await?;

        let mut context = ListenerContext {
            uid_id_map,
            order_book_sender,
            order_book_delta_sender,
            order_books: HashMap::new(),
        };
        let mut streaming = response.into_inner();
        while let Some(message) = streaming.message().await? {
            if let Some(payload) = message.payload {
                match payload {
                    Payload::SubscribeOrderBookResponse(response) => {
                        info!("Order book subscribe response: {response:?}");
                    }
                    Payload::Orderbook(order_book) => {
                        if let Err(err) = handle_order_book(&mut context, order_book).await {
                            error!("Error handling orderbook: {err}");
                        }
                    }
                    Payload::Ping(ping) => {
                        let request = MarketDataRequest {
                            payload: Some(market_data_request::Payload::Ping(
                                PingRequest {
                                    time: ping.time,
                                }
                            )),
                        };
                        tx.send(request)?;
                    }
                    _ => {
                        warn!("Ignore message: {:?}", payload);
                    }
                }
            }
        }
        Ok(())
    }
}

impl TInvest {
    pub async fn new() -> Result<Self, CommonError> {
        let token = std::env::var("T_INVEST_TOKEN").expect("Expected a T_INVEST_TOKEN in the environment");
        let sdk = TInvestSdk::new_production(&token).await?;
        info!("SDK initialized");
        Ok(Self { sdk })
    }
}

struct ListenerContext {
    uid_id_map: HashMap<String, i32>,
    order_books: HashMap<String, OrderBook>,
    order_book_sender: mpsc::Sender<OrderBookEvent>,
    order_book_delta_sender: mpsc::Sender<OrderBookDeltaEvent>,
}

async fn handle_order_book(context: &mut ListenerContext, api_order_book: t_invest_sdk::api::OrderBook) -> EmptyResult {
    if !api_order_book.is_consistent {
        warn!("Order book is not consistent: {api_order_book:?}");
    }
    let new_order_book = convert_order_book(&api_order_book);
    let mut prev_order_book = OrderBook {
        id: None,
        timestamp: new_order_book.timestamp,
        asks: Vec::new(),
        bids: Vec::new(),
    };
    if let Some(order_books) = context.order_books.get(&api_order_book.instrument_uid) {
        prev_order_book = order_books.clone();
    }
    let instrument_id = context.uid_id_map.get(&api_order_book.instrument_uid).copied().unwrap_or_default();
    if prev_order_book.asks.is_empty() || prev_order_book.bids.is_empty() {
        context.order_book_sender.send(OrderBookEvent {
            instrument_id,
            order_book: new_order_book.clone(),
        }).await?;
    } else {
        let ask_delta = calculate_delta(&prev_order_book.asks, &new_order_book.asks);
        let bid_delta = calculate_delta(&prev_order_book.bids, &new_order_book.bids);
        context.order_book_delta_sender.send(OrderBookDeltaEvent {
            instrument_id,
            timestamp: new_order_book.timestamp,
            asks: ask_delta,
            bids: bid_delta,
        }).await?;
    }
    context.order_books.insert(api_order_book.instrument_uid, new_order_book);
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
                    size: 0,
                })
            }
        }));
    delta
}

fn convert_order_book(order_book: &t_invest_sdk::api::OrderBook) -> OrderBook {
    OrderBook {
        id: None,
        timestamp: convert_timestamp(order_book.time.map(|t|  (t.seconds, t.nanos))),
        asks: convert_orders(&order_book.asks),
        bids: convert_orders(&order_book.bids),
    }
}

fn convert_orders(order: &[t_invest_sdk::api::Order]) -> Vec<OrderData> {
    order.iter()
        .filter_map(|order| {
            order.price.as_ref().map(|price| {
                OrderData {
                    price: convert_quotation(price),
                    size: order.quantity as i32,
                }
            })
        }).collect::<Vec<_>>()
}

fn convert_timestamp(time: Option<(i64, i32)>) -> DateTime<Utc> {
    time.and_then(|(s, ns)| Utc.timestamp_opt(s, ns as u32).earliest())
        .unwrap_or(Utc::now())
}

fn convert_quotation(quotation: &Quotation) -> Decimal {
    Decimal::from(quotation.units)
        .add(Decimal::from_i128_with_scale(quotation.nano as i128, 9))
}

fn decode_trade_direction(dir: i32) -> Option<char> {
    if TradeDirection::Buy as i32 == dir {
        Some('b')
    } else if TradeDirection::Sell as i32 == dir {
        Some('s')
    } else {
        None
    }
}