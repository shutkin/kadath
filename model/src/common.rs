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

pub fn export_features_to_csv<D: std::fmt::Display>(data: &[Vec<D>], horizons: &[f32], filename: &str) {
    let size = data[0].len();
    let mut csv = Vec::new();

    let header = format!("{},horizon", (0..size).map(|i| format!("f{i}")).collect::<Vec<_>>().join(","));
    csv.push(header);

    for (features, horizon) in data.iter().zip(horizons.iter()) {
        if features.len() != size {
            println!("Feature length mismatch: expected {}, got {}", size, features.len())
        }
        let row = format!("{},{horizon:.2}", features.iter().map(|f| format!("{f:.6}")).collect::<Vec<_>>().join(","));
        csv.push(row);
    }

    if let Err(err) = std::fs::write(filename, csv.join("\n")) {
        println!("Failed to export data to CSV: {}", err);
    }
}
