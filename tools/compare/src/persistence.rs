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
