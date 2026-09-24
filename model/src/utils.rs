use crate::OrderData;

fn map_delta(data1: Option<&OrderData>, data2: Option<&OrderData>) -> Option<OrderData> {
    if let (Some(data1), Some(data2)) = (data1, data2) {
        Some(OrderData {
            price: data2.price - data1.price,
            size: data1.size.min(data2.size),
        })
    } else {
        None
    }
}

pub fn handle_delta(deltas: &[OrderData], order_book_data: &mut Vec<OrderData>) {
    for delta_data in deltas {
        if delta_data.size == 0 {
            order_book_data.retain(|data| data.price != delta_data.price);
        } else {
            let mut found = false;
            for data in order_book_data.as_mut_slice() {
                if data.price == delta_data.price {
                    data.size = delta_data.size;
                    found = true;
                    break;
                }
            }
            if !found {
                order_book_data.push(delta_data.clone());
            }
        }
    }
}
