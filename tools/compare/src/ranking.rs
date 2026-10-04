use std::{any::type_name_of_val, collections::HashMap};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

use crate::{diff::Comparison, json_pointer::JsonPointer, persistence::Persistence, side::Side};

pub struct Scored {
    pub keys: Vec<String>,
    pub scores: HashMap<String, f64>,
}

#[derive(Deserialize)]
pub struct Ranking {
    pub path: JsonPointer,
    pub key: String,
    pub score: String,
    pub persistence: Persistence,
    #[serde(default)]
    pub top_k: Vec<usize>,
}

impl Ranking {
    pub fn compare(
        &self,
        baseline: &Value,
        candidate: &Value,
        tolerance: f64,
    ) -> Result<Comparison> {
        let path = self.path.as_str();
        let pick = |side: Side, body: &Value| -> Result<Scored> {
            let at = body
                .pointer(path)
                .with_context(|| format!("{side} has nothing at {path}"))?;
            self.extract(at, side)
        };

        let baseline = pick(Side::Baseline, baseline)?;
        let candidate = pick(Side::Candidate, candidate)?;

        let only_in_baseline = Self::difference(&baseline, &candidate);
        let only_in_candidate = Self::difference(&candidate, &baseline);

        todo!();
    }

    fn extract(&self, value: &Value, side: Side) -> Result<Scored> {
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
        Ok(Scored { keys, scores })
    }

    fn difference(from: &Scored, against: &Scored) -> Vec<String> {
        let mut keys: Vec<String> = from
            .keys
            .iter()
            .filter(|k| !against.scores.contains_key(*k))
            .cloned()
            .collect();
        keys.sort();
        keys
    }
}
