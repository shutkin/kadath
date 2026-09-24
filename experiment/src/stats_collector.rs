use std::io::Write;
use crate::deal::{Deal, DealDirection};
use crate::math_utils::{mean, median, p05_p95, positive_percentile};
use crate::signal_optimization::{calibrate_params, collect_signal_to_pnl_values, CostFunctionImpl, Signal, SignalParams, SignalPnL};
//use crate::signals::{signal_huber_09_07, signal_spearman_09_07, signal_trading_09_07, signal_unknown_09_03};
use crate::simulation::{DealHandler, find_order_books_on_horizon, run_stats};
use crate::{commission, MarketEvent, DEFAULT_HOLD_MS};
use chrono::{NaiveDate, TimeDelta};
use log::info;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::ops::Range;

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

pub fn signal_contributions(events: &[MarketEvent], params: &SignalParams) -> String {
    const PERCENTILES: [f64; 15] = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

    struct Runner {
        values: SignalPnL,
        signal_sorted: Vec<f64>,
    }

    impl Runner {
        pub fn new(events: &[MarketEvent], params: &SignalParams) -> Self {
            let values = collect_signal_to_pnl_values(events, &params);
            let mut signal_sorted = values.signal.clone();
            signal_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            Self { values, signal_sorted }
        }
    }

    impl StatisticsProvider for Runner {
        fn variants(&self) -> Vec<String> {
            PERCENTILES.iter().map(|p| format!("{p}%")).collect()
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let offset_from_top = (self.signal_sorted.len() as f64 * PERCENTILES[variant as usize] / 100.0).round() as usize;
            let threshold = self.signal_sorted[self.signal_sorted.len() - 1 - offset_from_top];
            let mut contributions = Vec::new();
            for (signal, contribution) in self.values.signal.iter().zip(self.values.contributions.iter()) {
                if *signal >= threshold {
                    contributions.push(*contribution);
                }
            }
            let size = contributions.len() as f64;
            let d = contributions.iter().map(|c| c[0]).sum::<f64>() / size;
            let i1 = contributions.iter().map(|c| c[1]).sum::<f64>() / size;
            let i2 = contributions.iter().map(|c| c[2]).sum::<f64>() / size;
            vec![
                ("D".to_string(), format!("{d}")),
                ("I1".to_string(), format!("{i1}")),
                ("I2".to_string(), format!("{i2}")),
            ]
        }
    }

    let mut runner = Runner::new(events, params);
    collect_to_table(&mut runner)
}

pub fn future_spreads_correlations(events: &[MarketEvent]) -> String {
    struct Runner<'a> {
        events: &'a [MarketEvent],
    }

    const HORIZONS: [i64; 5] = [500, 1000, 1500, 3000, 6000];

    impl<'a> StatisticsProvider for Runner<'a> {
        fn variants(&self) -> Vec<String> {
            HORIZONS.iter().map(|v| v.to_string()).collect()
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let mut c = Vec::new();
            let mut d1 = Vec::new();
            let mut d2 = Vec::new();
            let mut i1_l3 = Vec::new();
            let mut i1_l5 = Vec::new();
            let mut i1_l10 = Vec::new();
            let mut i1_l20 = Vec::new();
            let mut i1_l50 = Vec::new();
            let mut i2_l3 = Vec::new();
            let mut i2_l5 = Vec::new();
            let mut i2_l10 = Vec::new();
            let mut i2_l20 = Vec::new();
            let mut i2_l50 = Vec::new();

            let (mut v1, mut v2) = (None, None);
            for (i, event) in self.events.iter().enumerate() {
                match event {
                    MarketEvent::OrderBook1(v) => v1 = Some(v),
                    MarketEvent::OrderBook2(v) => v2 = Some(v),
                    _ => {}
                }
                if let Some(v1) = v1 && let Some(v2) = v2 &&
                    let Some((hv1, hv2)) =
                        find_order_books_on_horizon(self.events, i, event.event_datetime() + TimeDelta::milliseconds(HORIZONS[variant as usize])) {
                    let spread = v2.mid - v1.mid;
                    let hspread = hv2.mid - hv1.mid;
                    let change = if v1.std_derivative > 0.0 {
                        hspread - spread
                    } else {
                        spread - hspread
                    };

                    c.push(change);
                    d1.push(v1.std_derivative.abs());
                    d2.push(v2.std_derivative);
                    i1_l3.push(v1.imbalances[0]);
                    i1_l5.push(v1.imbalances[1]);
                    i1_l10.push(v1.imbalances[2]);
                    i1_l20.push(v1.imbalances[3]);
                    i1_l50.push(v1.imbalances[4]);
                    i2_l3.push(v2.imbalances[0]);
                    i2_l5.push(v2.imbalances[1]);
                    i2_l10.push(v2.imbalances[2]);
                    i2_l20.push(v2.imbalances[3]);
                    i2_l50.push(v2.imbalances[4]);
                }
            }

            vec![
                ("d1".to_string(), format!("{:.4}", correlation::spearmanr(&d1, &c))),
                ("d2".to_string(), format!("{:.4}", correlation::spearmanr(&d2, &c))),
                ("i1_l3".to_string(), format!("{:.4}", correlation::spearmanr(&i1_l3, &c))),
                ("i1_l5".to_string(), format!("{:.4}", correlation::spearmanr(&i1_l5, &c))),
                ("i1_l10".to_string(), format!("{:.4}", correlation::spearmanr(&i1_l10, &c))),
                ("i1_l20".to_string(), format!("{:.4}", correlation::spearmanr(&i1_l20, &c))),
                ("i1_l50".to_string(), format!("{:.4}", correlation::spearmanr(&i1_l50, &c))),
                ("i2_l3".to_string(), format!("{:.4}", correlation::spearmanr(&i2_l3, &c))),
                ("i2_l5".to_string(), format!("{:.4}", correlation::spearmanr(&i2_l5, &c))),
                ("i2_l10".to_string(), format!("{:.4}", correlation::spearmanr(&i2_l10, &c))),
                ("i2_l20".to_string(), format!("{:.4}", correlation::spearmanr(&i2_l20, &c))),
                ("i2_l50".to_string(), format!("{:.4}", correlation::spearmanr(&i2_l50, &c))),
            ]
        }
    }

    let mut runner = Runner { events };
    collect_to_table(&mut runner)
}

pub fn future_spreads_stats(events: &[MarketEvent], delta: i64) -> String {
    struct Runner<'a> {
        events: &'a [MarketEvent],
        delta: i64,
    }

    impl<'a> StatisticsProvider for Runner<'a> {
        fn variants(&self) -> Vec<String> {
            vec!["mean".to_string(), "median".to_string(), "p95".to_string()]
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let mut daily = HashMap::new();
            let (mut v1, mut v2) = (None, None);
            for (i, event) in self.events.iter().enumerate() {
                match event {
                    MarketEvent::OrderBook1(v) => v1 = Some(v),
                    MarketEvent::OrderBook2(v) => v2 = Some(v),
                    _ => {}
                }
                if let Some(v1) = v1 && let Some(v2) = v2 &&
                    let Some((hv1, hv2)) = find_order_books_on_horizon(self.events, i, event.event_datetime() + TimeDelta::milliseconds(self.delta)) {
                    let spread = v2.mid - v1.mid;
                    let hspread = hv2.mid - hv1.mid;
                    let change = if v1.std_derivative > 0.0 {
                        hspread - spread
                    } else {
                        spread - hspread
                    };
                    daily.entry(event.event_datetime().date_naive()).or_insert(Vec::new()).push(change);
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

    let mut runner = Runner { events, delta };
    collect_to_table(&mut runner)
}

pub struct DailyDataWithSignalParams {
    pub day: NaiveDate,
    pub events: Vec<MarketEvent>,
    pub params: SignalParams,
}

pub fn trend_ranges(events: &[MarketEvent]) -> String {
    let mut up_spread_movements = Vec::new();
    let mut down_spread_movements = Vec::new();

    let (mut v1, mut v2) = (None, None);
    for (i, event) in events.iter().enumerate() {
        match event {
            MarketEvent::OrderBook1(value) => v1 = Some(value),
            MarketEvent::OrderBook2(value) => v2 = Some(value),
            _ => {}
        }

        if let Some(v1) = v1 && let Some(v2) = v2 {
            let horizon_time = v1.time.max(v2.time) + TimeDelta::milliseconds(DEFAULT_HOLD_MS as i64);
            if let Some((hv1, hv2)) = find_order_books_on_horizon(events, i, horizon_time) {
                let c = (v1.trend + v2.trend) * 0.5;
                let a = (v1.trend + v2.trend) / (v1.trend.abs() + v2.trend.abs());
                let a = if a.is_nan() {-1.0} else {a};
                if a.abs() < 0.95 {
                    let spread1 = v2.mid - v1.mid;
                    let spread2 = hv2.mid - hv1.mid;
                    if v1.std_derivative > 0.0 {
                        up_spread_movements.push((c, a, spread2 - spread1));
                    } else {
                        down_spread_movements.push((c, a, spread1 - spread2));
                    }
                }
            }
        }
    }

    let mut result = Vec::new();

    let movements_up = up_spread_movements.iter().map(|(_, _, m)| *m).collect::<Vec<f64>>();
    let movements_down = down_spread_movements.iter().map(|(_, _, m)| *m).collect::<Vec<f64>>();
    let (_, threshold_up) = p05_p95(&movements_up);
    let (_, threshold_down) = p05_p95(&movements_down);

    let positive_c = up_spread_movements.iter()
        .filter(|(_, _, m)| *m > threshold_up).map(|(c, _, _)| *c).collect::<Vec<_>>();
    let (p05, p95) = p05_p95(&positive_c);
    result.push(format!("Positive Up C: {p05} - {p95}"));

    let negative_c = up_spread_movements.iter()
        .filter(|(_, _, m)| *m < threshold_up).map(|(c, _, _)| *c).collect::<Vec<_>>();
    let (p05, p95) = p05_p95(&negative_c);
    result.push(format!("Negative Up C: {p05} - {p95}"));

    let positive_c = down_spread_movements.iter()
        .filter(|(_, _, m)| *m > threshold_down).map(|(c, _, _)| *c).collect::<Vec<_>>();
    let (p05, p95) = p05_p95(&positive_c);
    result.push(format!("Positive Down C: {p05} - {p95}"));

    let negative_c = down_spread_movements.iter()
        .filter(|(_, _, m)| *m < threshold_down).map(|(c, _, _)| *c).collect::<Vec<_>>();
    let (p05, p95) = p05_p95(&negative_c);
    result.push(format!("Negative Down C: {p05} - {p95}"));

    let positive_a = up_spread_movements.iter()
        .filter(|(_, _, m)| *m > threshold_up).map(|(_, a, _)| *a).collect::<Vec<_>>();
    let (p05, p95) = p05_p95(&positive_a);
    result.push(format!("Positive Up A: {p05} - {p95}"));

    let negative_a = up_spread_movements.iter()
        .filter(|(_, _, m)| *m < threshold_up).map(|(_, a, _)| *a).collect::<Vec<_>>();
    let (p05, p95) = p05_p95(&negative_a);
    result.push(format!("Negative Up A: {p05} - {p95}"));

    let positive_a = down_spread_movements.iter()
        .filter(|(_, _, m)| *m > threshold_down).map(|(_, a, _)| *a).collect::<Vec<_>>();
    let (p05, p95) = p05_p95(&positive_a);
    result.push(format!("Positive Down A: {p05} - {p95}"));

    let negative_a = down_spread_movements.iter()
        .filter(|(_, _, m)| *m < threshold_down).map(|(_, a, _)| *a).collect::<Vec<_>>();
    let (p05, p95) = p05_p95(&negative_a);
    result.push(format!("Negative Down A: {p05} - {p95}"));

    result.join("\n")
}

pub fn trend_to_pnl(data: &[DailyDataWithSignalParams]) -> String {
    const C_BUCKETS: [Range<f64>; 6] = [-1000.0..-0.25, -0.25..-0.0625, -0.0625..0.0, 0.0..0.0625, 0.0625..0.25, 0.25..1000.0];
    const A_BUCKETS: [Range<f64>; 6] = [-1000.0..-0.95, -0.95..-0.75, -0.75..0.0, 0.0..0.75, 0.75..0.95, 0.95..1000.0];

    struct Runner<'a> {
        data: &'a [DailyDataWithSignalParams],
    }

    impl<'a> StatisticsProvider for Runner<'a> {
        fn variants(&self) -> Vec<String> {
            A_BUCKETS.iter().map(|c| format!("{c:?}")).collect()
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let a_range = &A_BUCKETS[variant as usize];
            let mut pnl = [0.0, 0.0];
            let mut counts = [0, 0];

            for day_data in self.data {
                let signal_up = SignalParams {
                    hold_ms: day_data.params.hold_ms,
                    up: day_data.params.up,
                    down: None,
                };
                let signal_down = SignalParams {
                    hold_ms: day_data.params.hold_ms,
                    up: None,
                    down: day_data.params.down,
                };
                let signals_calculators = [signal_up, signal_down];
                let mut prev_signals = [Signal::None, Signal::None];
                let (mut v1, mut v2) = (None, None);

                for (event_index, event) in day_data.events.iter().enumerate() {
                    match event {
                        MarketEvent::OrderBook1(v) => v1 = Some(v),
                        MarketEvent::OrderBook2(v) => v2 = Some(v),
                        _ => {}
                    }
                    if let Some(v1) = v1 && let Some(v2) = v2 {
                        //let c = (v1.trend + v2.trend) * 0.5;
                        let a = (v1.trend + v2.trend) / (v1.trend.abs() + v2.trend.abs());
                        for (signal_index, signal_calc) in signals_calculators.iter().enumerate() {
                            let (signal, _) = signal_calc.signal(v1, v2);
                            if signal != prev_signals[signal_index] && a_range.contains(&a) {
                                let deal = match signal {
                                    Signal::Sell1Buy2 => Some(Deal::sell1_buy2(v1, v2)),
                                    Signal::Buy1Sell2 => Some(Deal::buy1_sell2(v1, v2)),
                                    Signal::None => None,
                                };
                                if let Some(mut deal) = deal &&
                                    let Some((hv1, hv2)) = find_order_books_on_horizon(
                                        &day_data.events, event_index, deal.get_open_time() + TimeDelta::milliseconds(day_data.params.hold_ms as i64)) {
                                    match deal.get_direction() {
                                        DealDirection::Sell1Buy2 => {
                                            deal.close_instrument1(hv1.ask, hv1.time);
                                            deal.close_instrument2(hv2.bid, hv2.time);
                                        },
                                        DealDirection::Buy1Sell2 => {
                                            deal.close_instrument1(hv1.bid, hv1.time);
                                            deal.close_instrument2(hv2.ask, hv2.time);
                                        }
                                    }
                                    let (_, revenue, cost) = deal.close(false);
                                    pnl[signal_index] += revenue - cost - commission(revenue, cost);
                                    counts[signal_index] += 1;
                                }
                            }
                            prev_signals[signal_index] = signal;
                        }
                    }
                }
            }
            vec![
                ("Up".to_string(), format!("{:.3} ({})", pnl[0], counts[0])),
                ("Down".to_string(), format!("{:.3} ({})", pnl[1], counts[1])),
            ]
        }
    }

    let mut runner = Runner { data };
    collect_to_table(&mut runner)
}

pub fn trend_buckets(daily_events: &[(String, Vec<MarketEvent>)]) -> String {
    const BUCKETS: [Range<f64>; 10] = [-1000.0..-0.5, -0.5..-0.25, -0.25..-0.125, -0.125..-0.0625, -0.0625..0.0, 0.0..0.0625, 0.0625..0.125, 0.125..0.25, 0.25..0.5, 0.5..1000.0];

    struct Runner<'a> {
        daily_events: &'a [(String, Vec<MarketEvent>)],
    }

    impl StatisticsProvider for Runner<'_> {
        fn variants(&self) -> Vec<String> {
            BUCKETS.iter().map(|r| format!("{r:?}")).collect()
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let bucket_range = &BUCKETS[variant as usize];
            let mut rows = Vec::new();
            for (day, day_events) in self.daily_events {
                let (mut bucket_c, mut bucket_a) = (0, 0);
                let (mut v1, mut v2) = (None, None);
                for event in day_events {
                    match event {
                        MarketEvent::OrderBook1(v) => v1 = Some(v),
                        MarketEvent::OrderBook2(v) => v2 = Some(v),
                        _ => {}
                    }
                    if let Some(v1) = v1 && let Some(v2) = v2 {
                        let c = (v1.trend + v2.trend) * 0.5;
                        let a = (v1.trend + v2.trend) / (v1.trend.abs() + v2.trend.abs());

                        if bucket_range.contains(&c) {
                            bucket_c += 1;
                        }
                        if bucket_range.contains(&a) {
                            bucket_a += 1;
                        }
                    }
                }
                rows.extend(vec![
                    (format!("{day}: C"), format!("{bucket_c}")),
                    (format!("{day}: A"), format!("{bucket_a}"))]);
            }
            rows
        }
    }

    let mut runner = Runner { daily_events };
    collect_to_table(&mut runner)
}

pub fn deviation_to_spread_movement(events: &[MarketEvent], params: SignalParams) -> String {
    const HORIZONS: [i64; 5] = [100, 250, 500, 1000, 2000];
    const D_RANGES: [Range<f64>; 8] = [-123456.0..-2.0, -2.0..-1.0, -1.0..0.0, 0.0..1.0, 1.0..2.0, 2.0..5.0, 5.0..10.0, 10.0..123456.0];

    struct Runner<'a> {
        events: &'a [MarketEvent],
        params: SignalParams,
    }

    impl StatisticsProvider for Runner<'_> {
        fn variants(&self) -> Vec<String> {
            let mut variants = Vec::new();
            variants.push("N".to_string());
            for horizon in HORIZONS {
                variants.push(format!("{horizon}ms mean spread move"));
            }
            variants
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let mut buckets = vec![0.0; D_RANGES.len()];
            let mut buckets_sizes = vec![0; D_RANGES.len()];
            let (mut v1, mut v2) = (None, None);
            for (i, event) in self.events.iter().enumerate() {
                match event {
                    MarketEvent::OrderBook1(v) => v1 = Some(v),
                    MarketEvent::OrderBook2(v) => v2 = Some(v),
                    _ => {}
                }
                if let Some(v1) = v1 && let Some(v2) = v2 {
                    let horizon_time = v1.time.max(v2.time) + TimeDelta::milliseconds(if variant > 0 {HORIZONS[variant as usize - 1]} else {0});
                    if let Some((hv1, hv2)) = find_order_books_on_horizon(self.events, i, horizon_time) &&
                        let Some(params) = if v1.std_derivative > 0.0 {&self.params.up} else {&self.params.down} {
                            let d = v1.std_derivative.abs() * params.derivative1_weight + v2.std_derivative * params.derivative2_weight;
                            let bucket = D_RANGES.iter().position(|r| r.contains(&d)).unwrap();
                            let cur_spread = v2.mid - v1.mid;
                            let spread_horizon = hv2.mid - hv1.mid;
                            let spread_delta = if v1.std_derivative > 0.0 {
                                spread_horizon - cur_spread
                            } else {
                                cur_spread - spread_horizon
                            };
                            buckets[bucket] += spread_delta;
                            buckets_sizes[bucket] += 1;
                    }
                }
            }
            D_RANGES.iter().enumerate().map(|(i, range)| {
                (format!("{range:?}"),
                 if variant == 0 {
                     format!("{}", buckets_sizes[i])
                 } else {
                     format!("{:.4}", buckets[i] / buckets_sizes[i] as f64)
                 })
            }).collect()
        }
    }

    let mut runner = Runner { events, params };
    collect_to_table(&mut runner)
}

pub fn threshold_horizon_probabilities(events: &[MarketEvent], params: SignalParams) -> String {
    const HORIZONS: [u32; 11] = [250, 500, 1000, 2000, 5000, 10000, 30000, 60000, 300000, 600000, 1800000];

    struct Handler {
        map: HashMap<u32, Vec<bool>>,
    }

    impl DealHandler for Handler {
        fn handle_deal(&mut self, events: &[MarketEvent], event_index: usize, deal: Deal) {
            for horizon in HORIZONS {
                if let Some((v1, v2)) = find_order_books_on_horizon(
                    events, event_index, deal.get_open_time() + TimeDelta::milliseconds(horizon as i64)
                ) {
                    let mut deal_copy = deal;
                    match deal_copy.get_direction() {
                        DealDirection::Sell1Buy2 => {
                            deal_copy.close_instrument1(v1.ask, v1.time);
                            deal_copy.close_instrument2(v2.bid, v2.time);
                        },
                        DealDirection::Buy1Sell2 => {
                            deal_copy.close_instrument1(v1.bid, v1.time);
                            deal_copy.close_instrument2(v2.ask, v2.time);
                        }
                    }
                    let (_, revenue, cost) = deal_copy.close(false);
                    let commission = commission(revenue, cost);
                    self.map.get_mut(&horizon).unwrap().push(revenue - cost > commission);
                }
            }
        }

        fn get_stats(&self) -> Vec<(String, String)> {
            HORIZONS.iter().map(|horizon| {
                let v = self.map.get(horizon).unwrap();
                let positive = v.iter().filter(|x| **x).count();
                let probability = 100.0 * positive as f64 / v.len() as f64;
                (horizon.to_string(), format!("{probability:.3}% of {}", v.len()))
            }).collect()
        }

        fn init(&mut self, _variant: u8) {
            for horizon in HORIZONS {
                self.map.insert(horizon, Vec::<bool>::new());
            }
        }
    }

    let mut handler = Handler { map: HashMap::new() };

    struct Runner<'a> {
        pub events: &'a [MarketEvent],
        pub params: SignalParams,
        pub deal_handler: &'a mut dyn DealHandler,
    }

    impl StatisticsProvider for Runner<'_> {
        fn variants(&self) -> Vec<String> {
            vec!["threshold 0".to_string(), "threshold 5".to_string(), "threshold 10".to_string()]
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            /*self.params.signal_threshold = match variant {
                1 => 5.0,
                2 => 10.0,
                _ => 0.0,
            };*/
            self.deal_handler.init(variant);
            run_stats(self.events, &self.params, self.deal_handler)
        }
    }

    let mut runner = Runner {
        events,
        params,
        deal_handler: &mut handler,
    };

    collect_to_table(&mut runner)
}

pub fn signal_after_deal(events: &[MarketEvent], params: SignalParams) -> String {
    const HORIZONS: [u32; 9] = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000];

    enum StatVariant {
        Mean, Median,
    }

    struct Handler {
        map: HashMap<u32, Vec<f64>>,
        params: SignalParams,
        variant: StatVariant,
    }

    impl DealHandler for Handler {
        fn handle_deal(&mut self, events: &[MarketEvent], event_index: usize, deal: Deal) {
            for horizon in HORIZONS {
                if let Some((v1, v2)) = find_order_books_on_horizon(
                    events, event_index, deal.get_open_time() + TimeDelta::milliseconds(horizon as i64)
                ) {
                    let score = self.params.signal_score(&v1, &v2).unwrap_or(0.0);
                    self.map.get_mut(&horizon).unwrap().push(score);
                }
            }
        }

        fn get_stats(&self) -> Vec<(String, String)> {
            HORIZONS.iter().map(|horizon| {
                let mut values = self.map.get(horizon).cloned().unwrap_or_default();
                let v = match self.variant {
                    StatVariant::Mean => {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                    StatVariant::Median => {
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        values[values.len() / 2]
                    }
                };
                (horizon.to_string(), format!("{v:.2}"))
            }).collect()
        }

        fn init(&mut self, variant: u8) {
            self.variant = match variant {
                1 | 3 | 5 => StatVariant::Median,
                _ => StatVariant::Mean
            };
            for horizon in HORIZONS {
                self.map.insert(horizon, Vec::new());
            }
        }
    }

    let mut handler = Handler {
        map: HashMap::new(),
        params,
        variant: StatVariant::Mean,
    };

    struct Runner<'a> {
        pub events: &'a [MarketEvent],
        pub params: SignalParams,
        pub deal_handler: &'a mut dyn DealHandler,
    }

    impl StatisticsProvider for Runner<'_> {
        fn variants(&self) -> Vec<String> {
            vec!["threshold 0 mean".to_string(), "threshold 0 median".to_string(),
                 "threshold 5 mean".to_string(), "threshold 5 median".to_string(),
                 "threshold 10 mean".to_string(), "threshold 10 median".to_string(),]
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            /*self.params.signal_threshold = match variant {
                2 | 3 => 5.0,
                4 | 5 => 10.0,
                _ => 0.0,
            };*/
            self.deal_handler.init(variant);
            run_stats(self.events, &self.params, self.deal_handler)
        }
    }

    let mut runner = Runner {
        events,
        params,
        deal_handler: &mut handler,
    };

    collect_to_table(&mut runner)
}

pub fn daily_signal_to_pnl(daily_events: Vec<Vec<MarketEvent>>, dir: DealDirection) -> String {
    const PERCENTILES: [f64; 5] = [0.05, 0.02, 0.01, 0.005, 0.002];

    struct Runner {
        day_titles: Vec<String>,
        values: Vec<SignalPnL>,
        signal_sorted: Vec<Vec<f64>>,
    }

    impl Runner {
        pub fn new(daily_events: Vec<Vec<MarketEvent>>, dir: DealDirection) -> Self {
            let mut day_titles = Vec::new();
            let mut daily_values = Vec::new();
            let mut daily_signal_sorted = Vec::new();

            for i in 1..daily_events.len() {
                let mut train_events = Vec::new();
                for back in 0..3 {
                    let train_day = i as i32 + back - 3;
                    if train_day >= 0 {
                        train_events.extend_from_slice(&daily_events[train_day as usize]);
                    }
                }
                let day_title = daily_events[i][0].event_datetime().date_naive().to_string();

                let params = calibrate_params(&train_events, CostFunctionImpl::HuberLoss, dir, DEFAULT_HOLD_MS);

                let mut file = OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open("params.txt").unwrap();
                writeln!(file, "{day_title} {dir:?} {params:?}");

                let values = collect_signal_to_pnl_values(&daily_events[i], &params);
                let mut signal_sorted = values.signal.clone();
                signal_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                info!("Signals len {}", signal_sorted.len());
                day_titles.push(day_title);
                daily_values.push(values);
                daily_signal_sorted.push(signal_sorted);
            }

            Self {
                day_titles,
                values: daily_values,
                signal_sorted: daily_signal_sorted,
            }
        }
    }

    impl StatisticsProvider for Runner {
        fn variants(&self) -> Vec<String> {
            PERCENTILES.iter().flat_map(|p| vec![format!("{p}% total PnL"), format!("{p}% win rate")]).collect()
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let mut result = Vec::new();
            for (day, (values, signal_sorted)) in self.values.iter().zip(self.signal_sorted.iter()).enumerate() {
                let percentile = PERCENTILES[variant as usize / 2];
                let offset_from_top = (signal_sorted.len() as f64 * percentile / 100.0).round() as usize;
                let threshold = signal_sorted[signal_sorted.len() - 1 - offset_from_top];
                let mut pnl_values = Vec::new();
                for (i, signal) in values.signal.iter().enumerate() {
                    if *signal >= threshold {
                        pnl_values.push(values.pnl[i] - values.commission[i]);
                    }
                }
                let date = self.day_titles[day].clone();
                let v = match variant % 2 {
                    0 => format!("{:.3}", pnl_values.iter().sum::<f64>()),
                    _ => format!("{:.2}%", positive_percentile(&pnl_values)),
                };
                result.push((date, v));
            }
            result
        }
    }

    let mut runner = Runner::new(daily_events, dir);
    collect_to_table(&mut runner)
}
/*
pub fn signal_to_future_pnl_advances(train_events: &[MarketEvent], test_events: &[MarketEvent]) -> Vec<String> {
    const PERCENTILES: [f64; 15] = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

    struct Runner {
        values: [SignalPnL; 2],
        signal_sorted: [Vec<f64>; 2],
    }

    impl Runner {
        pub fn new(train_events: &[MarketEvent], test_events: &[MarketEvent], params: &SignalParams) -> Self {
            let train_values = collect_signal_to_pnl_values(train_events, &params);
            let mut train_signal_sorted = train_values.signal.clone();
            train_signal_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let test_values = collect_signal_to_pnl_values(test_events, &params);
            let mut test_signal_sorted = test_values.signal.clone();
            test_signal_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            Self {
                values: [train_values, test_values],
                signal_sorted: [train_signal_sorted, test_signal_sorted],
            }
        }
    }

    impl StatisticsProvider for Runner {
        fn variants(&self) -> Vec<String> {
            vec!["N".to_string(), "mean PnL".to_string(), "median PnL".to_string(),
                 "std PnL".to_string(), "win rate".to_string(), "Signal threshold".to_string(),]
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            PERCENTILES.iter().map(|percentile| {
                let mut threshold = [0.0, 0.0];
                let mut pnl_values = [Vec::new(), Vec::new()];
                for data_set in 0..2 {
                    let offset_from_top = (self.signal_sorted[data_set].len() as f64 * percentile / 100.0).round() as usize;
                    threshold[data_set] = self.signal_sorted[data_set][self.signal_sorted[data_set].len() - 1 - offset_from_top];

                    for (i, signal) in self.values[data_set].signal.iter().enumerate() {
                        if *signal >= threshold[data_set] {
                            pnl_values[data_set].push(self.values[data_set].pnl[i] - self.values[data_set].commission[i]);
                        }
                    }
                }

                let percentile_str = format!("{}%", percentile);
                let v = match variant {
                    0 => format!("{} / {}", pnl_values[0].len(), pnl_values[1].len()),
                    1 => format!("{:.3} / {:.3}", mean(&pnl_values[0]), mean(&pnl_values[1])),
                    2 => format!("{:.3} / {:.3}", median(&pnl_values[0]), median(&pnl_values[1])),
                    3 => format!("{:.3} / {:.3}", standard_deviation(&pnl_values[0]).unwrap_or_default(), standard_deviation(&pnl_values[1]).unwrap_or_default()),
                    4 => format!("{:.3} / {:.3}", positive_percentile(&pnl_values[0]), positive_percentile(&pnl_values[1])),
                    _ => format!("{:.4} / {:.4}", threshold[0], threshold[1]),
                };
                (percentile_str, v)
            }).collect()
        }
    }
    
    let mut result = Vec::new();

    let params = SignalParams {hold_ms: DEFAULT_HOLD_MS, up: None, down: signal_trading_09_07().down};
    let mut runner = Runner::new(train_events, test_events, &params);
    result.push(collect_to_table(&mut runner));
    
    //let params = SignalParams {hold_ms: DEFAULT_HOLD_MS, up: signal_spearman_09_07().up, down: None};
    //let mut runner = Runner::new(train_events, test_events, &params);
    //result.push(collect_to_table(&mut runner));
    
    let params = SignalParams {hold_ms: DEFAULT_HOLD_MS, up: None, down: signal_huber_09_07().down};
    let mut runner = Runner::new(train_events, test_events, &params);
    result.push(collect_to_table(&mut runner));
    
    result
}

pub fn signal_to_future_pnl_relation(events: &[MarketEvent]) -> String {
    struct Runner<'a> {
        pub events: &'a [MarketEvent],
    }

    impl StatisticsProvider for Runner<'_> {
        fn variants(&self) -> Vec<String> {
            vec!["simulation UP".to_string(), "simulation DOWN".to_string(),
                 "Spearman UP".to_string(), "Spearman DOWN".to_string(),
                 "Huber UP".to_string(), "Huber DOWN".to_string()]
        }

        fn run(&mut self, variant: u8) -> Vec<(String, String)> {
            let params = match variant {
                0 => SignalParams { hold_ms: DEFAULT_HOLD_MS, up: signal_trading_09_07().up, down: None },
                1 => SignalParams { hold_ms: DEFAULT_HOLD_MS, up: None, down: signal_trading_09_07().down },
                2 => SignalParams { hold_ms: DEFAULT_HOLD_MS, up: signal_spearman_09_07().up, down: None },
                3 => SignalParams { hold_ms: DEFAULT_HOLD_MS, up: None, down: signal_spearman_09_07().down },
                4 => SignalParams { hold_ms: DEFAULT_HOLD_MS, up: signal_huber_09_07().up, down: None },
                5 => SignalParams { hold_ms: DEFAULT_HOLD_MS, up: None, down: signal_huber_09_07().down },
                _ => signal_unknown_09_03(),
            };
            let values = collect_signal_to_pnl_values(self.events, &params);
            if values.signal.len() < 100 {
                return (0..20).map(|i| (i.to_string(), "-".to_string())).collect();
            }

            let mut s = values.signal.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let min = s[s.len() / 10];
            let max = s[s.len() * 9 / 10];

            (0..20).map(|i| (i, min + (max - min) * i as f64 / 20.0 .. min + (max - min) * (i + 1) as f64 / 20.0))
                .map(|(index, range)| {
                    let mut signal_pnl = Vec::new();
                    let (mut win, mut loss) = (0, 0);
                    for (i, signal) in values.signal.iter().enumerate() {
                        if range.contains(signal) {
                            signal_pnl.push(values.pnl[i]);
                            if values.pnl[i] > 0.0 {
                                win += 1;
                            } else {
                                loss += 1;
                            }
                        }
                    }
                    signal_pnl.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mean = signal_pnl.iter().sum::<f64>() / signal_pnl.len() as f64;
                    let p25 = signal_pnl[signal_pnl.len() / 4];
                    let p75 = signal_pnl[signal_pnl.len() * 3 / 4];
                    let win_loss = 100.0 * win as f64 / (win + loss) as f64;
                    (index.to_string(), format!("{:.3} ({:.3} - {:.3}) {:.2}% {}", mean, p25, p75, win_loss, win + loss))
                }).collect()
        }
    }

    let mut runner = Runner { events };
    collect_to_table(&mut runner)
}
 */