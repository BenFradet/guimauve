use std::path::Path;

use anyhow::Result;
use onnx_example::TranslationPlugin;
use tokio::runtime::Builder;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<()> {
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

    let cpus = std::thread::available_parallelism()?.get();

    plugin::server::set_max_concurrency(cpus.saturating_sub(1).max(1));

    Builder::new_multi_thread()
        .worker_threads(1)
        // to use in conjunction with spawn_blocking
        .max_blocking_threads(cpus)
        .enable_all()
        .build()?
        .block_on(async { plugin::server::serve(plugin, "0.0.0.0:3000").await })
}
