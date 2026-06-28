use std::path::Path;

use anyhow::Result;
use onnx_example::TranslationPlugin;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("{}=trace", env!("CARGO_CRATE_NAME")).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let plugin = TranslationPlugin::new(
        Path::new("/artifacts/en_tokenizer.json"),
        Path::new("/artifacts/pt_tokenizer.json"),
    )?;

    plugin::server::serve(plugin, "0.0.0.0:3000").await
}
