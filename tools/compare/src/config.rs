use std::{any::type_name_of_val, collections::HashMap, fmt::Display, fs, path::Path};

use anyhow::{Context, Error, Result, bail};
use serde::Deserialize;
use serde_json::Value;

use crate::json_pointer::JsonPointer;

#[derive(Deserialize)]
pub struct Service {
    pub url: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(try_from = "f64")]
pub struct Persistence(f64);

impl Persistence {
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Persistence {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self> {
        if value > 0. && value < 1. {
            Ok(Self(value))
        } else {
            bail!("persistence must be ]0; 1[, got {value}")
        }
    }
}

enum Qwe {
    Baseline,
    Candidate,
}

impl Display for Qwe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Qwe::Baseline => "baseline",
            Qwe::Candidate => "candidate",
        };
        write!(f, "{str}")
    }
}

pub struct Ranking {
    keys: Vec<String>,
    scores: HashMap<String, f64>,
}

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
    fn extract(&self, value: &Value, side: Qwe) -> Result<Ranking> {
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

#[derive(Deserialize)]
pub struct Config {
    pub baseline: Service,
    pub candidate: Service,
    #[serde(default)]
    pub ignore: Vec<JsonPointer>,
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    #[serde(default)]
    pub ordered: Vec<Ordered>,
}

fn default_tolerance() -> f64 {
    1e-5
}

pub fn load(path: &Path) -> Result<Config> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading config at {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing config at {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_parsing() {
        let inf: Result<Persistence> = (-0.25).try_into();
        assert!(inf.is_err());
        let sup: Result<Persistence> = (1.25).try_into();
        assert!(sup.is_err());
        let ok: Result<Persistence> = (0.25).try_into();
        assert!(ok.is_ok());
    }
}
