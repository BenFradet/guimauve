use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{self, State},
    http::StatusCode,
};

use crate::model_plugin::ModelPlugin;

async fn infer<P>(
    State(plugin): State<Arc<P>>,
    extract::Json(payload): extract::Json<P::Request>,
) -> Result<extract::Json<P::Response>, (StatusCode, String)>
where
    P: ModelPlugin,
{
    let input = plugin.pre(payload).map_err(internal_server_error)?;
    let output = plugin.infer(input).map_err(internal_server_error)?;
    let response = plugin.post(output).map_err(internal_server_error)?;
    Ok(extract::Json(response))
}

fn internal_server_error<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
