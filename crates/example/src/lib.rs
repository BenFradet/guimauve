use std::{collections::HashMap, fs};

use anyhow::Result;
use burn::backend::{Flex, flex::FlexDevice};
use burn::tensor::{Float, Int, Tensor};
use model::Model;
use plugin::model_plugin::ModelPlugin;
use serde::{Deserialize, Serialize};

pub mod model {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

#[derive(Deserialize)]
struct OnnxExampleRequest {
    seed: String,
    fs: Vec<(String, i64, i64)>,
    cs: Vec<String>,
}

#[derive(Serialize)]
struct OnnxExampleResponse {
    ss: Vec<(String, f32)>,
}

struct OnnxExampleModelInput {
    ss: Tensor<Flex, 3, Int>,
    ass: Tensor<Flex, 2, Int>,
    gs: Tensor<Flex, 2, Int>,
    sts: Tensor<Flex, 3, Int>,
    sas: Tensor<Flex, 2, Int>,
    sgs: Tensor<Flex, 2, Int>,
    fts: Tensor<Flex, 2, Int>,
    cts: Tensor<Flex, 2, Int>,
}

struct OnnxExampleModelOutput {
    pub es: Tensor<Flex, 2, Float>,
}

type IdMapEntry = (Vec<i64>, i64, i64);
type IdMap = HashMap<String, IdMapEntry>;

struct OnnxExamplePlugin {
    model: model::Model<Flex>,
    device: FlexDevice,
    map1: IdMap,
    map2: IdMap,
}

impl OnnxExamplePlugin {
    fn new(map1_path: &str, map2_path: &str) -> Result<Self> {
        let map1 = parse_map(map1_path)?;
        let map2 = parse_map(map2_path)?;

        let device = FlexDevice;
        let model: Model<Flex> = Model::default();

        Ok(Self {
            model,
            device,
            map1,
            map2,
        })
    }
}

impl ModelPlugin for OnnxExamplePlugin {
    type Error = anyhow::Error;
    type Request = OnnxExampleRequest;
    type Response = OnnxExampleResponse;
    type ModelInput = OnnxExampleModelInput;
    type ModelOutput = OnnxExampleModelOutput;

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

fn parse_map(path: &str) -> Result<IdMap> {
    let content = fs::read_to_string(path)?;
    let id_map: IdMap = serde_json::from_str(&content)?;
    Ok(id_map)
}
