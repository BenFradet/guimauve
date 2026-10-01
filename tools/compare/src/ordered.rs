use std::{any::type_name_of_val, collections::HashMap};

use anyhow::{bail, Result, Context};
use serde::Deserialize;
use serde_json::Value;

use crate::{config::Ranking, json_pointer::JsonPointer, persistence::Persistence, side::Side};

// TODO: rename
#[derive(Deserialize)]
pub struct Ordered {
    pub path: JsonPointer,
    pub key: String,
    pub score: String,
    pub persistence: Persistence,
    #[serde(default)]
    pub top_k: Vec<usize>,
}

impl Ordered {
    fn extract(&self, value: &Value, side: Side) -> Result<Ranking> {
        let path = self.path.as_str();
        let items = value.as_array().with_context(|| {
            format!(
                "{side} {path} is a {}, expected an array",
                type_name_of_val(value)
            )
        })?;

        let mut keys = Vec::with_capacity(items.len());
        let mut scores = HashMap::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let key = item
                .get(&self.key)
                .and_then(|v| v.as_str())
                .with_context(|| format!("{side} {path}/{i}: no string field {:?}", self.key))?;
            let score = item
                .get(&self.score)
                .and_then(Value::as_f64)
                .with_context(|| format!("{side} {path}/{i}: no f64 field {:?}", self.score))?;

            if scores.insert(key.to_string(), score).is_some() {
                bail!("{side} {path} contains {} = {key:?} twice", self.key);
            }
            keys.push(key.to_string());
        }
        Ok(Ranking { keys, scores })
    }
}
