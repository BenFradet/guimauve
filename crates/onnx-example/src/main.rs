use std::{path::Path, sync::Arc};

use anyhow::Result;
use axum::{
    Router,
    extract::{self, State},
    http::StatusCode,
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
    println!("listening on {}", listener.local_addr()?);
    let _ = axum::serve(listener, app).await;

    Ok(())
}

async fn infer(
    State(plugin): State<Arc<TranslationPlugin>>,
    extract::Json(payload): extract::Json<TranslationRequest>,
) -> Result<extract::Json<TranslationResponse>, (StatusCode, String)> {
    let input = plugin.pre(payload).map_err(internal_server_error)?;
    let output = plugin.infer(input).map_err(internal_server_error)?;
    let response = plugin.post(output).map_err(internal_server_error)?;
    Ok(extract::Json(response))
}

fn internal_server_error(e: anyhow::Error) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
