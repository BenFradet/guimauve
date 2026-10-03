use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::{diff::Diff, ordered::Ordered};

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
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    #[serde(default)]
    pub ordered: Vec<Ordered>,
}

impl Config {
    pub fn compare(&self, baseline: &Value, candidate: &Value) -> Result<Diff> {
        let mut comparisons = Vec::with_capacity(self.ordered.len());
        for spec in &self.ordered {
            comparisons.push(spec.compare(baseline, candidate, self.tolerance)?);
        }
        Ok(Diff(comparisons))
    }
}

fn default_tolerance() -> f64 {
    1e-5
}

pub fn load(path: &Path) -> Result<Config> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading config at {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing config at {}", path.display()))
}
