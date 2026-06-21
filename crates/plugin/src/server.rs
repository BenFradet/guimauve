use std::sync::Arc;

use anyhow::Result;
use axum::{
    Router,
    extract::{self, State},
    http::StatusCode,
    routing::post,
};
use tokio::net::TcpListener;

use crate::model_plugin::ModelPlugin;

pub async fn serve<P: ModelPlugin>(plugin: P, addr: &str) -> Result<()> {
    let plugin = Arc::new(plugin);
    let app = Router::new()
        .route("/infer", post(infer::<P>))
        .with_state(plugin);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn infer<P: ModelPlugin>(
    State(plugin): State<Arc<P>>,
    extract::Json(payload): extract::Json<P::Request>,
) -> Result<extract::Json<P::Response>, (StatusCode, String)> {
    let input = plugin.pre(payload).map_err(internal_server_error)?;
    let output = plugin.infer(input).map_err(internal_server_error)?;
    let response = plugin.post(output).map_err(internal_server_error)?;
    Ok(extract::Json(response))
}

fn internal_server_error<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
