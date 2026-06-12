use std::path::Path;

use anyhow::{Context, Error, Result};
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
pub struct TranslationRequest {
    en_sentence: String,
}

#[derive(Serialize)]
pub struct TranslationResponse {
    pt_sentence: String,
}

// model is expecting [batch, seq_len], hence 2d
pub struct TranslationModelInput {
    source_token_ids: Tensor<Flex, 2, Int>,
}

pub struct TranslationModelOutput {
    predicted_token_ids: Tensor<Flex, 1, Int>,
}

pub struct TranslationPlugin {
    model: model::Model<Flex>,
    device: FlexDevice,
    en_tokenizer: Tokenizer,
    pt_tokenizer: Tokenizer,
}

impl TranslationPlugin {
    pub fn new(en_tokenizer_path: &Path, pt_tokenizer_path: &Path) -> Result<Self> {
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

    fn encode_src(&self, en_sentence: &str) -> Result<Tensor<Flex, 2, Int>> {
        self.encode(&self.en_tokenizer, en_sentence)
    }

    fn encode(&self, tokenizer: &Tokenizer, sentence: &str) -> Result<Tensor<Flex, 2, Int>> {
        let tokenized = tokenizer.encode(sentence, false).map_err(Error::msg)?;
        let ids = tokenized.get_ids();
        Ok(
            Tensor::<Flex, 1, Int>::from_data(TensorData::from(ids), &self.device)
                // model is expecting [batch, seq_len], here [1, seq_len]
                .unsqueeze::<2>(),
        )
    }
}

impl ModelPlugin for TranslationPlugin {
    type Error = anyhow::Error;
    type Request = TranslationRequest;
    type Response = TranslationResponse;
    type ModelInput = TranslationModelInput;
    type ModelOutput = TranslationModelOutput;

    fn pre(&self, req: Self::Request) -> Result<Self::ModelInput, Self::Error> {
        let input_tensor = self.encode_src(&req.en_sentence)?;
        Ok(TranslationModelInput {
            source_token_ids: input_tensor,
        })
    }

    // auto regressive loop
    // c.f. https://huggingface.co/blog/atharv6f/autoregressive-loop
    fn infer(&self, input: Self::ModelInput) -> Result<Self::ModelOutput, Self::Error> {
        let start_id = self
            .pt_tokenizer
            .token_to_id("[START]")
            .context("couldn't convert [START]")?;
        let end_id = self
            .pt_tokenizer
            .token_to_id("[END]")
            .context("couldn't convert [END]")?;

        let vocab_size = self.pt_tokenizer.get_vocab_size(false);
        let max_seq_len = self
            .pt_tokenizer
            .get_truncation()
            .map(|t| t.max_length)
            .unwrap_or(128);

        let mut last_id = start_id;
        let mut ids = vec![start_id];

        while ids.len() < max_seq_len && last_id != end_id {
            let mut padded = vec![0; 127];
            padded[..ids.len()].copy_from_slice(&ids);
            let target =
                Tensor::<Flex, 1, Int>::from_data(TensorData::from(&padded[..]), &self.device)
                    .unsqueeze::<2>();
            let target_len = ids.len();
            // [1, target_len, vocab_size]
            let logits = self.model.forward(input.source_token_ids.clone(), target);
            // logits for the last token [1, 1, vocab_size]
            let last_logits = logits.slice([0..1, (target_len - 1)..target_len, 0..vocab_size]);
            let next_token_id = last_logits.argmax(2).into_scalar() as u32;
            last_id = next_token_id;
            ids.push(next_token_id);
        }

        let target = Tensor::<Flex, 1, Int>::from_data(TensorData::from(&ids[..]), &self.device);
        Ok(TranslationModelOutput {
            predicted_token_ids: target,
        })
    }

    fn post(&self, output: Self::ModelOutput) -> Result<Self::Response, Self::Error> {
        let data = output.predicted_token_ids.to_data().into_vec()?;
        let decoded = self.pt_tokenizer.decode(&data, true).map_err(Error::msg)?;
        Ok(TranslationResponse {
            pt_sentence: decoded,
        })
    }
}
