use anyhow::{Error, Result, bail};
use serde::Deserialize;
use std::str::FromStr;

#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(try_from = "String")]
pub struct JsonPointer(String);

impl JsonPointer {
    fn as_str(&self) -> &str {
        &self.0
    }

    fn validate(s: &str) -> Result<()> {
        if s.starts_with('/') {
            Ok(())
        } else {
            bail!("JSON pointer must start with '/', got {s:?}")
        }
    }
}

impl FromStr for JsonPointer {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::validate(s)?;
        Ok(Self(s.to_owned()))
    }
}

impl TryFrom<String> for JsonPointer {
    type Error = Error;

    fn try_from(s: String) -> Result<Self> {
        Self::validate(&s)?;
        Ok(Self(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsing() {
        let correct: Result<JsonPointer> = "/going/down".parse();
        assert_eq!(JsonPointer("/going/down".to_string()), correct.unwrap());
        let incorrect: Result<JsonPointer> = "going/down".parse();
        assert_eq!(
            "JSON pointer must start with '/', got \"going/down\"",
            format!("{}", incorrect.unwrap_err())
        );
    }
}
