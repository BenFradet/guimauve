use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::{diff::Diff, ranking::Ranking};

#[derive(Deserialize)]
pub struct Service {
    pub url: String,
}

#[derive(Deserialize)]
pub struct Config {
    pub baseline: Service,
    pub candidate: Service,
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    #[serde(default)]
    pub rankings: Vec<Ranking>,
}

impl Config {
    pub fn compare(&self, baseline: &Value, candidate: &Value) -> Result<Diff> {
        let mut comparisons = Vec::with_capacity(self.rankings.len());
        for spec in &self.rankings {
            comparisons.push(spec.compare(baseline, candidate, self.tolerance)?);
        }
        Ok(Diff(comparisons))
    }

    pub fn load(path: &Path) -> Result<Config> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading config at {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing config at {}", path.display()))
    }
}

fn default_tolerance() -> f64 {
    1e-5
}
