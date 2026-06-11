use std::{path::Path, sync::Arc};

use anyhow::Result;
use axum::{
    Router,
    extract::{self, State},
    routing::post,
};
use example::{TranslationPlugin, TranslationRequest, TranslationResponse};
use plugin::model_plugin::ModelPlugin;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let plugin = Arc::new(TranslationPlugin::new(
        Path::new("../../models/pt_to_en/en_tokenizer.json"),
        Path::new("../../models/pt_to_en/pt_tokenizer.json"),
    )?);

    let app = Router::new()
        .route("/infer", post(infer))
        .with_state(plugin);

    let listener = TcpListener::bind("0.0.0.0:3000").await?;
    tracing::debug!("listening on {}", listener.local_addr()?);
    let _ = axum::serve(listener, app).await;

    Ok(())
}

async fn infer(
    State(plugin): State<Arc<TranslationPlugin>>,
    extract::Json(payload): extract::Json<TranslationRequest>,
) -> extract::Json<TranslationResponse> {
    let input = plugin.pre(payload).unwrap();
    let output = plugin.infer(input).unwrap();
    let response = plugin.post(output).unwrap();
    extract::Json(response)
}
