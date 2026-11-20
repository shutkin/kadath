use serde::Deserialize;
use serde_json::Value;
use model::common::CommonError;

#[derive(Deserialize)]
pub struct PredefinedConfigs {
    pub positive: Value,
    pub negative: Value,
}

impl PredefinedConfigs {
    pub fn read(path: &str) -> Result<Vec<Self>, CommonError> {
        let result = serde_json::from_reader(std::fs::File::open(path)?)?;
        Ok(result)
    }
}