//! `guimauve`, French for marshmallow, is an inference server built on top of:
//! - [burn]
//! - [axum]
//! 
//! # Features
//! 
//! - seamless [burn] integration to import models
//! - production-ready inference serving thanks to [axum]
//! - concurrency control: semaphore-based admission control to prevent oversubscription
//! - ease of use:
//!   - bring your ONNX or [burn] model
//!   - implement a trait
//!   - get an inference server with backpressure and resource management
//! 
//! # Performance
//! 
//! Internal performance testing has proven `guimauve` to be a lot less resource-hungry and behave
//! better under load compared to a `FastAPI` + `PyTorch` set up.
//!
//! # Next up
//! 
//! - GPU inference
//! - batch inference
//! 
//! # Quickstart
//! 
//! Add `guimauve` as a dependency:
//! 
//! ```bash
//! cargo add guimauve
//! ```
//! 
//! ```rust,no_run
//! // Implement the [`ModelPlugin`] trait
//! // lib.rs
//! use burn::backend::flex;
//! use burn::tensor::{Int, Tensor};
//! use guimauve::model_plugin::ModelPlugin;
//! 
//! struct MyPlugin;
//! 
//! impl ModelPlugin for MyPlugin {
//!     type Request = serde_json::Value;
//!     type Response = serde_json::Value;
//!     type ModelInput = Tensor<Flex, 2, Int>;
//!     type ModelOutput = Tensor<Flex, 2, Int>;
//!     type Error = anyhow::Error;
//! 
//!     fn pre(&self, req: Self::Request) -> Result<Self::ModelInput, Self::Error> {
//!         // parse and prepare model input
//!         todo!()
//!     }
//! 
//!     fn infer(&self, input: Self::ModelInput) -> Result<Self::ModelOutput, Self::Error> {
//!         // run inference
//!         todo!()
//!     }
//! 
//!     fn post(&self, output: Self::ModelOutput) -> Result<Self::Response, Self::Error> {
//!         // format response
//!         todo!()
//!     }
//! }
//!
//! // Define your entrypoint
//! // main.rs
//! fn main() -> anyhow::Result<()> {
//!     let plugin = MyPlugin;
//! 
//!     guimauve::server::Server::builder(plugin)
//!         .address("0.0.0.0:3000")
//!         .build()?
//!         .serve()
//! }
//! ```
//! 
//! Two endpoints are available:
//! 
//! ```bash
//! curl http://0.0.0.0:3000/v1/health
//! # inference
//! curl -X POST http://0.0.0.0:3000/v1/infer \
//!     -H 'Content-Type: application/json' \
//!     -d '{"en_sentence": "why are people from Lisboa eating snails?"}'
//! ```
//! There is a dedicated [examples] folder for more.
//!
//! [`ModelPlugin`]: crate::model_plugin::ModelPlugin
//! [examples]: https://github.com/BenFradet/guimauve/tree/main/examples
//! [burn]: https://github.com/tracel-ai/burn
//! [axum]: https://github.com/tokio-rs/axum

#![warn(clippy::pedantic)]
pub mod model_plugin;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "store")]
pub mod store;
