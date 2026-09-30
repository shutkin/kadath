use async_trait::async_trait;
use tokio::sync::mpsc;
use model::common::{CommonError, EmptyResult};
use model::{Instrument, Trade};
use model::events::{OrderBookDeltaEvent, OrderBookEvent};

#[async_trait]
pub trait BrokerListener {
    async fn get_instruments(&self) -> Result<Vec<Instrument>, CommonError>;
    async fn listen_trades(&self, trade_sender: mpsc::Sender<Trade>) -> EmptyResult;
    async fn listen_order_books(
        &self,
        instruments: &[Instrument],
        order_book_sender: mpsc::Sender<OrderBookEvent>,
        order_book_delta_sender: mpsc::Sender<OrderBookDeltaEvent>,
    ) -> EmptyResult;
}