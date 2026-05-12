use std::{collections::HashMap, fs};

use anyhow::Result;
use plugin::model_plugin::ModelPlugin;

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

pub struct OnnxExample;
impl ModelPlugin for OnnxExample {
    type Error = bool;
    type Request = bool;
    type Response = bool;
    type ModelInput = bool;
    type ModelOutput = bool;

    fn pre(&self, req: Self::Request) -> Result<Self::ModelInput, Self::Error> {
        todo!()
    }

    fn infer(&self, input: Self::ModelInput) -> Result<Self::ModelOutput, Self::Error> {
        todo!()
    }

    fn post(&self, output: Self::ModelOutput) -> Result<Self::Response, Self::Error> {
        todo!()
    }
}

