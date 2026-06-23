use std::path::Path;

use anyhow::Result;
use example::TranslationPlugin;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let plugin = TranslationPlugin::new(
        Path::new("/artifacts/en_tokenizer.json"),
        Path::new("/artifacts/pt_tokenizer.json"),
    )?;

    plugin::server::serve(plugin, "0.0.0.0:3000").await
}
