use chrono::{DateTime, Datelike, TimeDelta, Utc};
use log::{info, LevelFilter};
use simplelog::SimpleLogger;
use db::test_db::TestDb;
use model::common::{EmptyResult, PredictionDebugInfo};
use model::math_util::rolling_normalize_binary;
use crate::gradient_boost::{optimize_light_gbm, PredictionType, GradientBoost, DataVec};

const LIGHT_GBM_CONFIG: &str = r#"{"bagging_fraction":0.7327036369763196,"bagging_freq":3,"boosting_type":"gbdt","feature_fraction":0.5971240990903507,"lambda_l1":0.2160740371052931,"lambda_l2":5.65296063258647,"learning_rate":0.012229584614859448,"max_depth":-1,"metric":"binary_logloss","min_child_samples":609,"n_jobs":-1,"num_iterations":2000,"num_leaves":7,"objective":"binary","random_state":42,"verbosity":-1}"#;
const LIGHT_GBM_CONFIG_REGRESS: &str = r#"{"objective": "regression", "metric": "l2", "boosting": "gbdt", "max_depth": 3, "num_leaves": 6, "learning_rate": 0.03, "num_iterations": 200, "min_data_in_leaf": 500, "feature_fraction": 1.0, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l1": 1.0, "lambda_l2": 5.0, "min_gain_to_split": 0.01, "max_bin": 63, "verbosity": -1}"#;
const CONFIG_GBM: bool = false;

pub async fn test_session_experiment() -> EmptyResult {
    dotenv::dotenv().ok();
    SimpleLogger::init(LevelFilter::Info, simplelog::Config::default()).ok();
    let db_url = std::env::var("DB_URL").expect("DB_URL is not set");
    let test_db = TestDb::new(&db_url).await?;

    let signals = test_db.query_session_signals(2).await?
        .into_iter()
        .filter(|(time, _, _, _, _)| time.date_naive().day() == 11)
        .collect::<Vec<_>>();
    info!("Got {} signals", signals.len());

    let window_size = 25000;

    let linear = signals.iter().map(|(_, s, _, _, _)| *s as f64).collect::<Vec<_>>();
    info!("Rank linear signals...");
    let linear_ranked = rolling_normalize_binary(&linear, false, window_size);

    let gbm_positive = signals.iter().map(|(_, _, s, _, _)| *s as f64).collect::<Vec<_>>();
    info!("Rank GBM positive signals...");
    let gbm_positive_ranked = rolling_normalize_binary(&gbm_positive, false, window_size);
    let gbm_negative = signals.iter().map(|(_, _, _, s, _)| *s as f64).collect::<Vec<_>>();
    info!("Rank GBM negative signals...");
    let gbm_negative_ranked = rolling_normalize_binary(&gbm_negative, true, window_size);

    info!("Combine GBM signals...");
    let gbm_ranked = gbm_positive_ranked.iter()
        .zip(gbm_negative_ranked.iter())
        .map(|(positive, negative)| positive * (0.75 + 0.25 * negative))
        .collect::<Vec<_>>();

    let prediction = PredictionType::ClassificationPositive(4.0);
    if CONFIG_GBM {
        info!("Configure GBM");
        let test_start = *signals.iter().map(|(t, _, _, _, _)| t).max().unwrap() - TimeDelta::hours(2);
        let mut train_features = Vec::with_capacity(linear_ranked.len());
        let mut train_horizons = Vec::with_capacity(linear_ranked.len());
        let mut test_features = Vec::new();
        let mut test_horizons = Vec::new();
        for i in 0..linear_ranked.len() {
            let time = signals[i + window_size - 1].0;
            if time < test_start {
                train_features.push(vec![linear_ranked[i] as f32, signals[i + window_size - 1].2, signals[i + window_size - 1].3]);
                train_horizons.push(signals[i + window_size - 1].4);
            } else {
                test_features.push(vec![linear_ranked[i] as f32, signals[i + window_size - 1].2, signals[i + window_size - 1].3]);
                test_horizons.push(signals[i + window_size - 1].4);
            }
        }
        let cfg = optimize_light_gbm(
            prediction,
            &train_features,
            &train_horizons,
            &test_features,
            &test_horizons,
            40
        )?;
        info!("Optimized config: {cfg}");
    }

    info!("Train GBM...");
    let train_end = *signals.iter().skip(window_size).map(|(t, _, _, _, _)| t).min().unwrap() + TimeDelta::hours(4);
    let mut train_features = Vec::with_capacity(linear_ranked.len());
    let mut train_horizons = Vec::with_capacity(linear_ranked.len());
    let mut test_features = Vec::new();
    for i in 0..linear_ranked.len() {
        let time = signals[i + window_size - 1].0;
        if time > train_end {
            test_features.push(vec![linear_ranked[i] as f32, signals[i + window_size - 1].2, signals[i + window_size - 1].3]);
        } else {
            train_features.push(vec![linear_ranked[i] as f32, signals[i + window_size - 1].2, signals[i + window_size - 1].3]);
            train_horizons.push(signals[i + window_size - 1].4);
        }
    }
    let config = serde_json::from_str(LIGHT_GBM_CONFIG_REGRESS)?;
    let gbm = GradientBoost::train(prediction, &config, DataVec::Owned(train_features), &train_horizons)?;
    info!("Calculate final signal...");
    let finals = gbm.test(&test_features)?;

    info!("Save...");
    let mut debug_info = Vec::with_capacity(linear_ranked.len());
    let mut final_index = 0;
    for i in 0..linear_ranked.len() {
        let time = signals[i + window_size - 1].0;
        if time > train_end {
            let horizon = signals[i + window_size - 1].4 as f64;
            let info = PredictionDebugInfo {
                linear_raw: signals[i + window_size - 1].1 as f64,
                gbm_positive_raw: signals[i + window_size - 1].2 as f64,
                gbm_negative_raw: signals[i + window_size - 1].3 as f64,
                linear_ranked: linear_ranked[i],
                gbm_ranked: gbm_ranked[i],
                is_filtered_by_trend: true,
            };
            let final_signal = finals[final_index];
            final_index += 1;
            debug_info.push((time, info, final_signal, horizon));
        }
    }

    save_predictions_csv(&debug_info, "session_predictions_data.csv")?;

    Ok(())
}


fn save_predictions_csv(data: &[(DateTime<Utc>, PredictionDebugInfo, f64, f64)], filename: &str) -> EmptyResult {
    let mut csv = Vec::with_capacity(data.len() + 1);
    csv.push("time,linear,gbm positive,gbm negative,linear ranked,final,horizon".to_string());

    for (time, info, final_signal, horizon) in data {
        csv.push(format!("{time},{:.6},{:.6},{:.6},{:.6},{:.6},{horizon:.1}",
                         info.linear_raw, info.gbm_positive_raw,
                         info.gbm_negative_raw, info.linear_ranked,
                         final_signal));
    }

    std::fs::write(filename, csv.join("\n"))?;

    Ok(())
}
