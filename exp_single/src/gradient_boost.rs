use crate::orderbook_values::OrderBookValues;
use gbdt::config::Config;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use log::info;
use plotters::prelude::LogScalable;

pub struct GradientBoost {
    gbdt: GBDT,
}

impl GradientBoost {
    pub fn new() -> Self {
        let mut cfg = Config::new();
        cfg.set_feature_size(12);
        cfg.set_max_depth(3);
        cfg.set_iterations(100);
        cfg.set_shrinkage(0.05);
        cfg.set_min_leaf_size(200);
        cfg.set_loss("SquaredError");
        cfg.set_data_sample_ratio(0.8);
        cfg.set_feature_sample_ratio(1.0);
        cfg.set_debug(false);

        Self { gbdt: GBDT::new(&cfg) }
    }

    pub fn train(&mut self, data: &[OrderBookValues], horizons: &[f64]) {
        let mut train_data = data.iter().zip(horizons.iter())
            .map(|(values, horizon)| {
                Data::new_training_data(values_to_features(values), 1.0, *horizon as f32, None)
            }).collect::<Vec<_>>();

        info!("Start training...");
        self.gbdt.fit(&mut train_data);
        info!("Train completed");
    }

    pub fn test(&self, data: &[OrderBookValues]) -> Vec<f64> {
        let test_data = data.iter().map(|values| {
            Data::new_training_data(values_to_features(values), 1.0, 0.0, None)
        }).collect::<Vec<_>>();
        let predict_vec = self.gbdt.predict(&test_data);
        predict_vec.into_iter().map(|p| p.as_f64()).collect()
    }
}

fn values_to_features(values: &OrderBookValues) -> Vec<f32> {
    vec![
        values.mid_derivative as f32,
        (values.mid - values.trend) as f32,
        values.imbalances[0] as f32,
        values.imbalances[1] as f32,
        values.imbalances[2] as f32,
        values.imbalances[3] as f32,
        values.imbalances[4] as f32,
        values.imbalances_derivative[0] as f32,
        values.imbalances_derivative[1] as f32,
        values.imbalances_derivative[2] as f32,
        values.imbalances_derivative[3] as f32,
        values.imbalances_derivative[4] as f32,
    ]
}