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
struct TranslationRequest {
    en_sentence: String,
}

#[derive(Serialize)]
struct TranslationResponse {
    pt_sentence: String,
}

struct TranslationModelInput {
    source: Tensor<Flex, 2, Int>,
    target: Tensor<Flex, 2, Int>,
}

struct TranslationModelOutput {
    logits: Tensor<Flex, 3, Float>,
}

struct TranslationPlugin {
    model: model::Model<Flex>,
    device: FlexDevice,
}

impl TranslationPlugin {
    fn new() -> Result<Self> {
        let device = FlexDevice;
        let model: Model<Flex> = Model::default();

        Ok(Self {
            model,
            device,
        })
    }
}

impl ModelPlugin for TranslationPlugin {
    type Error = anyhow::Error;
    type Request = TranslationRequest;
    type Response = TranslationResponse;
    type ModelInput = TranslationModelInput;
    type ModelOutput = TranslationModelOutput;

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
