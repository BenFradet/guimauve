use std::path::Path;

use anyhow::{Error, Result};
use burn::backend::{Flex, flex::FlexDevice};
use burn::tensor::{Int, Tensor, TensorData};
use model::Model;
use plugin::model_plugin::ModelPlugin;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

pub mod model {
    include!(concat!(env!("OUT_DIR"), "/model/transformer.rs"));
}

#[derive(Deserialize)]
struct TranslationRequest {
    en_sentence: String,
}

#[derive(Serialize)]
struct TranslationResponse {
    pt_sentence: String,
}

// model is expecting [batch, seq_len], hence 2d
struct TranslationModelInput {
    source_token_ids: Tensor<Flex, 2, Int>,
}

struct TranslationModelOutput {
    predicted_token_ids: Tensor<Flex, 1, Int>,
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
        let tokenized = self
            .en_tokenizer
            .encode(req.en_sentence, false)
            .map_err(Error::msg)?;
        let ids = tokenized.get_ids();
        let input_tensor = Tensor::<Flex, 1, Int>::from_data(TensorData::from(ids), &self.device)
            // model is expecting [batch, seq_len], here [1, seq_len]
            .unsqueeze::<2>();
        Ok(TranslationModelInput {
            source_token_ids: input_tensor,
        })
    }

    fn infer(&self, input: Self::ModelInput) -> Result<Self::ModelOutput, Self::Error> {
        todo!()
    }

    fn post(&self, output: Self::ModelOutput) -> Result<Self::Response, Self::Error> {
        todo!()
    }
}
