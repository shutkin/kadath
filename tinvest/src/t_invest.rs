mod debug_to_json;

use crate::debug_to_json::debug_value_to_json;
use abstract_broker::{AbstractBroker, calculate_delta};
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use log::{debug, error, info, warn};
use model::common::{CommonError, EmptyResult};
use model::events::{OrderBookDeltaEvent, OrderBookEvent};
use model::{DataSource, Instrument, OrderBook, OrderData, Trade};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::ops::Add;
use t_invest_sdk::TInvestSdk;
use t_invest_sdk::api::{AccountStatus, GetAccountsRequest, GetInfoRequest, GetMarginAttributesRequest, GetTradingStatusRequest, InstrumentStatus, InstrumentsRequest, MarketDataRequest, MoneyValue, OrderBookInstrument, OrderBookType, OrderExecutionReportStatus, OrderStateStreamRequest, OrderType, PingRequest, PostOrderAsyncRequest, PriceType, Quotation, SubscribeOrderBookRequest, SubscriptionAction, TimeInForceType, TradeDirection, market_data_request};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Clone)]
pub struct TInvest {
    sdk: TInvestSdk,
    account_id: String,
}

#[async_trait]
impl AbstractBroker for TInvest {
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

    async fn listen_trades(&self, trade_sender: mpsc::Sender<Trade>) -> EmptyResult {

        let request = OrderStateStreamRequest {
            accounts: vec![self.account_id.clone()],
            ping_delay_millis: None,
        };
        let mut streaming = self.sdk.orders_stream()
            .order_state_stream(request).await?
            .into_inner();

        while let Some(message) = streaming.message().await? {
            if let Some(payload) = message.payload {
                match payload {
                    t_invest_sdk::api::order_state_stream_response::Payload::OrderState(state) => {
                        if let Some(direction) = decode_trade_direction(state.direction) && let Some(status) = decode_trade_status(state.execution_report_status) {
                            let value = heuristic_debug_parser::parse(&format!("{state:?}"))?;
                            let json_value = debug_value_to_json(value);
                            let json = serde_json::to_string(&json_value)?;
                            let trade = Trade {
                                id: None,
                                direction,
                                status,
                                executed: convert_timestamp(state.completion_time.map(|t|  (t.seconds, t.nanos))),
                                price: state.executed_order_price.map(|m| convert_money(&m)).unwrap_or_default(),
                                quantity: state.lots_executed as i16,
                                order_id: state.order_request_id.unwrap_or_default(),
                                json,
                            };
                            if let Err(err) = trade_sender.send(trade).await {
                                warn!("Failed to send trade: {err}");
                                break;
                            }
                        } else {
                            debug!("Order State unknown direction {} or status {}", state.direction, state.execution_report_status);
                        }
                    }
                    t_invest_sdk::api::order_state_stream_response::Payload::Ping(ping) => {
                        debug!("Order State ping {ping:?}");
                    }
                    t_invest_sdk::api::order_state_stream_response::Payload::Subscription(subs) => {
                        info!("Order State subscription response {subs:?}");
                    }
                    t_invest_sdk::api::order_state_stream_response::Payload::StopOrderState(state) => {
                        info!("Stop Order state {state:?}");
                    }
                }
            }
        }

        Ok(())
    }

    async fn listen_order_books(
        &self,
        instruments: &[Instrument],
        order_book_sender: mpsc::Sender<OrderBookEvent>,
        order_book_delta_sender: mpsc::Sender<OrderBookDeltaEvent>
    ) -> EmptyResult {
        let mut uid_id_map = HashMap::with_capacity(instruments.len());
        for instrument in instruments {
            if let Some(uid) = &instrument.external_id && let Some(id) = instrument.id {
                uid_id_map.insert(uid.clone(), id);
            }
        }
        let subs = uid_id_map.keys()
            .map(|uid| {
                #[allow(deprecated)]
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
                    t_invest_sdk::api::market_data_response::Payload::SubscribeOrderBookResponse(response) => {
                        info!("Order book subscribe response: {response:?}");
                    }
                    t_invest_sdk::api::market_data_response::Payload::Orderbook(order_book) => {
                        if let Err(err) = handle_order_book(&mut context, order_book).await {
                            error!("Error handling orderbook: {err}");
                        }
                    }
                    t_invest_sdk::api::market_data_response::Payload::Ping(ping) => {
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
    
    async fn make_order(&self, internal_id: &str, instrument: &Instrument, direction: model::TradeDirection, quantity: u16) -> Result<String, CommonError> {
        let direction = match direction {
            model::TradeDirection::Buy => TradeDirection::Buy as i32,
            model::TradeDirection::Sell => TradeDirection::Sell as i32,
        };

        let rq = PostOrderAsyncRequest {
            account_id: self.account_id.clone(),
            instrument_id: instrument.external_id.clone().ok_or("Instrument has no external id")?,
            order_id: internal_id.to_string(),
            quantity: quantity as i64,
            direction,
            price: None,
            price_type: Some(PriceType::Unspecified as i32),
            order_type: OrderType::Market as i32,
            time_in_force: Some(TimeInForceType::TimeInForceUnspecified as i32),
            confirm_margin_trade: true,
        };

        info!("Post request {rq:?}");
        let resp = self.sdk.orders().post_order_async(rq).await?.into_inner();
        info!("Response {resp:?}");
        match resp.execution_report_status {
            2 => Err("Order status REJECTED".into()),
            3 => Err("Order status CANCELED".into()),
            _ => Ok(resp.order_request_id),
        }
    }
}

impl TInvest {
    pub async fn new() -> Result<Self, CommonError> {
        let token = std::env::var("T_INVEST_TOKEN").expect("Expected a T_INVEST_TOKEN in the environment");
        let sdk = TInvestSdk::new_production(&token).await?;
        info!("T-Invest SDK initialized");

        let accounts_rq = GetAccountsRequest {
            status: Some(AccountStatus::Open as i32),
        };
        let accounts_resp = sdk.users().get_accounts(accounts_rq)
            .await?.into_inner();
        info!("Account {accounts_resp:?}");
        let account_id = accounts_resp.accounts.first().ok_or("Active account not found")?.id.clone();

        let info_rq = GetInfoRequest {};
        let info = sdk.users().get_info(info_rq).await?.into_inner();
        info!("Info {info:?}");

        #[allow(deprecated)]
        let status_rq = GetTradingStatusRequest {
            instrument_id: Some("312c293a-462d-4684-a8e2-882255c44f2e".to_string()),
            figi: None,
        };
        let status = sdk.market_data().get_trading_status(status_rq).await?.into_inner();
        info!("Trading status: {status:?}");

        let margin_rq = GetMarginAttributesRequest {
            account_id: account_id.clone(),
        };
        let margin = sdk.users().get_margin_attributes(margin_rq).await?.into_inner();
        info!("Margin attributes: {margin:?}");

        Ok(Self { sdk, account_id })
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

fn convert_money(money_value: &MoneyValue) -> Decimal {
    Decimal::from(money_value.units)
        .add(Decimal::from_i128_with_scale(money_value.nano as i128, 9))
}

fn decode_trade_direction(dir: i32) -> Option<model::TradeDirection> {
    if TradeDirection::Buy as i32 == dir {
        Some(model::TradeDirection::Buy)
    } else if TradeDirection::Sell as i32 == dir {
        Some(model::TradeDirection::Sell)
    } else {
        None
    }
}

fn decode_trade_status(status: i32) -> Option<model::TradeStatus> {
    if OrderExecutionReportStatus::ExecutionReportStatusFill as i32 == status {
        Some(model::TradeStatus::Executed)
    } else if OrderExecutionReportStatus::ExecutionReportStatusRejected as i32 == status || OrderExecutionReportStatus::ExecutionReportStatusCancelled as i32 == status {
        Some(model::TradeStatus::Failed)
    } else {
        None
    }
}