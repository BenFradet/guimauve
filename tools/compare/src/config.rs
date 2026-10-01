use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::{json_pointer::JsonPointer, ordered::Ordered};

#[derive(Deserialize)]
pub struct Service {
    pub url: String,
}

pub struct Ranking {
    pub keys: Vec<String>,
    pub scores: HashMap<String, f64>,
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

impl Config {
    fn parse(&self, mut tree: Value) -> Parsed {
        for pointer in &self.ignore {
            if let Some(slot) = tree.pointer_mut(pointer.as_str()) {
                *slot = Value::Null;
            }
        }
        let mut ordered = Vec::with_capacity(self.ordered.len());
        for spec in &self.ordered {
            ordered.push(tree.pointer_mut(spec.path.as_str()).map(Value::take));
        }
        Parsed { tree, ordered }
    }
}

pub struct Parsed {
    tree: Value,
    ordered: Vec<Option<Value>>,
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
    use crate::persistence::Persistence;

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
