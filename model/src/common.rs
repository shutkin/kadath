use chrono::{DateTime, Utc};

pub type CommonError = Box<dyn std::error::Error + Send + Sync>;
pub type EmptyResult = Result<(), CommonError>;

#[derive(Debug, Copy, Clone)]
pub struct TimeDiapason {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
}

impl TimeDiapason {
    pub fn new(from: DateTime<Utc>, to: DateTime<Utc>) -> TimeDiapason {
        TimeDiapason { from, to }
    }
}