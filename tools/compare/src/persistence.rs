use anyhow::{Error, Result, bail};
use serde::Deserialize;

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
