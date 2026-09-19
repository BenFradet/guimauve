use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::json_pointer::JsonPointer;

#[derive(Deserialize)]
pub struct Service {
    pub url: String,
}

#[derive(Deserialize)]
pub struct Ordered {
    pub path: JsonPointer,
    pub key: String,
    pub score: String,
    // c.f. https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899/
    pub rank_biased_overlap: Vec<usize>,
}

#[derive(Deserialize)]
pub struct Config {
    pub baseline: Service,
    pub candidate: Service,
    pub ignore: Vec<JsonPointer>,
    pub tolerance: f64,
    pub ordered: Vec<Ordered>,
}

const DEFAULT_TOLERANCE: f64 = 1e-5;

pub fn load(path: &Path) -> Result<Config> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading config at {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing config at {}", path.display()))
}
