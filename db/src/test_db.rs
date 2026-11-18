use chrono::{DateTime, Utc};
use model::common::CommonError;
use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres, Row};

pub struct TestDb {
    pool: Pool<Postgres>,
}

impl TestDb {
    pub async fn new(url: &str) -> Result<Self, CommonError> {
        let pool = PgPoolOptions::new().max_connections(2).connect(url).await?;
        Ok(Self { pool })
    }

    pub async fn start_session(&self, instrument_id: i32) -> Result<i32, CommonError> {
        let id_row = sqlx::query("INSERT INTO test_session (started, instrument_id) VALUES ($1, $2) RETURNING id")
            .bind(Utc::now())
            .bind(instrument_id)
            .fetch_one(&self.pool)
            .await?;
        Ok(id_row.get(0))
    }

    pub async fn insert_signals(&self, session_id: i32, time: DateTime<Utc>, s0: f32, s1: f32, s2: f32) -> Result<(), CommonError> {
        sqlx::query("INSERT INTO test_signals (session_id, event_time, signal1, signal2, signal3) VALUES ($1, $2, $3, $4, $5)")
            .bind(session_id)
            .bind(time)
            .bind(s0)
            .bind(s1)
            .bind(s2)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn batch_insert_signals(&self, session_id: i32, time_signals: &[(DateTime<Utc>, f32, f32, f32)]) -> Result<(), CommonError> {
        for chunk in time_signals.chunks(2048) {
            let times = chunk.iter().map(|&(t, _, _, _)| t).collect::<Vec<_>>();
            let s0 = chunk.iter().map(|&(_, s, _, _)| s).collect::<Vec<f32>>();
            let s2 = chunk.iter().map(|&(_, _, s, _)| s).collect::<Vec<f32>>();
            let s1 = chunk.iter().map(|&(_, _, _, s)| s).collect::<Vec<f32>>();
            sqlx::query("INSERT INTO test_signals (session_id, event_time, signal1, signal2, signal3) SELECT $1, * FROM UNNEST($2, $3, $4, $5)")
                .bind(session_id)
                .bind(times)
                .bind(s0)
                .bind(s1)
                .bind(s2)
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }
}