use std::path::Path;

use anyhow::{Error, Result};
use burn::backend::{Flex, flex::FlexDevice};
use burn::tensor::{Float, Int, Tensor};
use model::Model;
use plugin::model_plugin::ModelPlugin;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

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
    en_tokenizer: Tokenizer,
    pt_tokenizer: Tokenizer,
}

impl TranslationPlugin {
    fn new(en_tokenizer_path: &Path, pt_tokenizer_path: &Path) -> Result<Self> {
        let device = FlexDevice;
        let model: Model<Flex> = Model::default();

        let en_tokenizer = Tokenizer::from_file(en_tokenizer_path).map_err(Error::msg)?;
        let pt_tokenizer = Tokenizer::from_file(pt_tokenizer_path).map_err(Error::msg)?;

        Ok(Self {
            model,
            device,
            en_tokenizer,
            pt_tokenizer,
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
