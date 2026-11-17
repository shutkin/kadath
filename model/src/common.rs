use chrono::{DateTime, Utc};

pub type CommonError = Box<dyn std::error::Error + Send + Sync>;
pub type EmptyResult = Result<(), CommonError>;

#[derive(Debug, Copy, Clone, Default)]
pub struct TimeDiapason {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
}

impl TimeDiapason {
    pub fn new(from: DateTime<Utc>, to: DateTime<Utc>) -> TimeDiapason {
        TimeDiapason { from, to }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PredictionDebugInfo {
    pub is_filtered_by_trend: bool,
    pub linear_raw: f64,
    pub gbm_positive_raw: f64,
    pub gbm_negative_raw: f64,
    pub linear_ranked: f64,
    pub gbm_ranked: f64,
}

impl Default for PredictionDebugInfo {
    fn default() -> Self {
        Self {
            is_filtered_by_trend: false,
            linear_raw: 0.0,
            gbm_negative_raw: 0.0,
            gbm_positive_raw: 0.0,
            linear_ranked: -1.0,
            gbm_ranked: -1.0,
        }
    }
}
