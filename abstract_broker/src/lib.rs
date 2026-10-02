use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::mpsc;
use model::common::{CommonError, EmptyResult};
use model::{Instrument, OrderData, Trade, TradeDirection};
use model::events::{OrderBookDeltaEvent, OrderBookEvent};

pub type BrokerBox = Arc<Box<dyn AbstractBroker + Send + Sync>>;

#[async_trait]
pub trait AbstractBroker {
    async fn get_instruments(&self) -> Result<Vec<Instrument>, CommonError>;
    async fn listen_trades(&self, trade_sender: mpsc::Sender<Trade>) -> EmptyResult;
    async fn listen_order_books(
        &self,
        instruments: &[Instrument],
        order_book_sender: mpsc::Sender<OrderBookEvent>,
        order_book_delta_sender: mpsc::Sender<OrderBookDeltaEvent>,
    ) -> EmptyResult;

    async fn make_order(&self, instrument: &Instrument, direction: TradeDirection, quantity: u16) -> Result<String, CommonError>;
}

pub fn calculate_delta(prev_orders: &[OrderData], new_orders: &[OrderData]) -> Vec<OrderData> {
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
