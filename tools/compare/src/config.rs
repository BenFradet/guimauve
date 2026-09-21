use std::{fs, path::Path};

use anyhow::{Context, Error, Result, bail};
use serde::Deserialize;

use crate::json_pointer::JsonPointer;

#[derive(Deserialize)]
pub struct Service {
    pub url: String,
}

#[derive(Deserialize)]
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

#[derive(Deserialize)]
pub struct Ordered {
    pub path: JsonPointer,
    pub key: String,
    pub score: String,
    pub persistence: Persistence,
    #[serde(default)]
    pub top_k: Vec<usize>,
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
