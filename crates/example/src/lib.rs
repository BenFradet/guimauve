use std::{collections::HashMap, fs};

use anyhow::Result;

pub mod model {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

type IdMapEntry = (i64, i64);
type IdMap = HashMap<String, IdMapEntry>;

fn parse_map(location: &str) -> Result<IdMap> {
    let content = fs::read_to_string(location)?;
    let id_map: IdMap = serde_json::from_str(&content)?;
    Ok(id_map)
}

