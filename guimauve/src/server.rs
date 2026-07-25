use std::sync::{Arc, OnceLock};

use anyhow::Result;
use axum::{
    Router,
    extract::{self, State},
    http::StatusCode,
    routing::{get, post},
};
use tokio::{net::TcpListener, signal, sync::Semaphore, task};

use crate::model_plugin::ModelPlugin;

// c.f. https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html#limit-the-number-of-incoming-requests-being-handled-at-the-same-time
static SEMAPHORE: OnceLock<Semaphore> = OnceLock::new();

pub async fn serve<P: ModelPlugin>(plugin: P, addr: &str) -> Result<()> {
    let plugin = Arc::new(plugin);
    let app = Router::new()
        .route("/infer", post(infer::<P>))
        .route("/health", get(|| async { StatusCode::OK }))
        .with_state(plugin);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("listening on {}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

pub fn set_max_concurrency(n: usize) {
    SEMAPHORE.get_or_init(|| Semaphore::new(n));
}

async fn infer<P: ModelPlugin>(
    State(plugin): State<Arc<P>>,
    extract::Json(payload): extract::Json<P::Request>,
) -> Result<extract::Json<P::Response>, (StatusCode, String)> {
    let _permit = match SEMAPHORE.get() {
        Some(sem) => sem.acquire().await.ok(),
        None => None,
    };
    // this is heavily cpu bound, hence spawn_blocking
    let response = task::spawn_blocking(move || {
        let input = plugin.pre(payload)?;
        let output = plugin.infer(input)?;
        plugin.post(output)
    })
    .await
    // in case of join errors
    .map_err(internal_server_error)?
    // plugin errors
    .map_err(internal_server_error)?;

    Ok(extract::Json(response))
}

fn internal_server_error<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install ctrl+c handler");
    };

    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
