#[derive(Debug, Eq, PartialEq)]
pub struct JsonPointer(String);

impl JsonPointer {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for JsonPointer {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.starts_with('/') {
            Ok(Self(value))
        } else {
            Err(format!("JSON pointer must start with '/', got {value:?}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_from() {
        let correct: Result<JsonPointer, String> = "/going/down".to_string().try_into();
        assert_eq!(Ok(JsonPointer("/going/down".to_string())), correct);
        let incorrect: Result<JsonPointer, String> = "going/down".to_string().try_into();
        assert_eq!(Err("JSON pointer must start with '/', got \"going/down\"".to_string()), incorrect);
    }
}
