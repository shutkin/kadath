use crate::{OrderBookValues, get_price_on_horizon};
use log::info;
use model::events::OrderBookEvent;
use model::math_util::{mean, median, p05_p95};
use std::collections::HashMap;

pub trait StatisticsProvider {
    fn variants(&self) -> Vec<String>;
    fn run(&mut self, variant: u8) -> Vec<(String, String)>;
}

fn collect_to_table(provider: &mut dyn StatisticsProvider) -> String {
    let variants = provider.variants();
    let mut table_rows = Vec::new();
    table_rows.push(format!("v|{}", variants.join("|")));
    table_rows.push((0..=variants.len()).map(|_| "-").collect::<Vec<_>>().join("|"));
    for i in 0..variants.len() as u8 {
        info!("Run variant {}", variants[i as usize]);
        let rows = provider.run(i);
        if i == 0 {
            for (name, value) in rows {
                table_rows.push(format!("{name}|{value}"));
            }
        } else {
            for (i, (_, value)) in rows.into_iter().enumerate() {
                table_rows[i + 2].push_str(&format!("|{value}"));
            }
        }
    }
    table_rows.join("\n")
}

pub fn future_spreads_stats(events: &[OrderBookEvent], first_leg_id: i32, second_leg_id: i32) -> String {
    struct Runner<'a> {
        first_leg_id: i32,
        second_leg_id: i32,
        events: &'a [OrderBookEvent],
    }

    impl<'a> StatisticsProvider for Runner<'a> {
        fn variants(&self) -> Vec<String> {
            vec!["mean".to_string(), "median".to_string(), "p95".to_string()]
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let mut daily = HashMap::new();
            let (mut v1, mut v2) = (None, None);
            for (i, event) in self.events.iter().enumerate() {
                if event.instrument_id == self.first_leg_id {
                    v1 = OrderBookValues::from_order_book(&event.order_book);
                } else if event.instrument_id == self.second_leg_id {
                    v2 = OrderBookValues::from_order_book(&event.order_book);
                }
                if let Some(v1) = v1 && let Some(v2) = v2
                    && let Some(h_mid1) = get_price_on_horizon(self.events, self.first_leg_id, i)
                    && let Some(h_mid2) = get_price_on_horizon(self.events, self.second_leg_id, i) {
                    let spread = v2.mid - v1.mid;
                    let hspread = h_mid2 - h_mid1;
                    let change = hspread - spread;
                    daily.entry(event.order_book.timestamp.date_naive()).or_insert(Vec::new()).push(change);
                }
            }
            let mut days = daily.keys().copied().collect::<Vec<_>>();
            days.sort();

            let mut result = Vec::with_capacity(days.len());
            for day in days {
                let changes = daily.remove(&day).unwrap_or_default();
                let value = match variant {
                    0 => mean(&changes),
                    1 => median(&changes),
                    _ => p05_p95(&changes).1
                };
                result.push((day.to_string(), format!("{value:.4}")));
            }
            result
        }
    }

    let mut runner = Runner { events, first_leg_id, second_leg_id };
    collect_to_table(&mut runner)
}
