use std::sync::{Arc, OnceLock};

use anyhow::Result;
use axum::{
    Router,
    extract::{self, State},
    http::StatusCode,
    routing::{get, post},
};
use tokio::{net::TcpListener, signal, sync::Semaphore, task};
use tokio::runtime::Builder;

use crate::model_plugin::ModelPlugin;

pub struct Server<P: ModelPlugin> {
    plugin: P,
    address: String,
    worker_threads: usize,
    max_blocking_threads: usize,
    max_concurrency: usize,
}

impl<P: ModelPlugin> Server<P> {
    pub fn builder(plugin: P) -> ServerBuilder<P> {
        ServerBuilder::new(plugin)
    }

    pub fn serve(self) -> Result<()> {
        set_max_concurrency(self.max_concurrency);

        Builder::new_multi_thread()
            .worker_threads(self.worker_threads)
            // to use in conjunction with spawn_blocking
            .max_blocking_threads(self.max_blocking_threads)
            .enable_all()
            .build()?
            .block_on(async { serve(self.plugin, &self.address).await })
    }
}

pub struct ServerBuilder<P: ModelPlugin> {
    plugin: P,
    address: Option<String>,
    worker_threads: Option<usize>,
    max_blocking_threads: Option<usize>,
    max_concurrency: Option<usize>,
}

impl<P: ModelPlugin> ServerBuilder<P> {
    pub fn new(plugin: P) -> Self {
        Self {
            plugin,
            address: None,
            worker_threads: None,
            max_blocking_threads: None,
            max_concurrency: None,
        }
    }

    pub fn address(mut self, address: String) -> Self {
        self.address = Some(address);
        self
    }

    pub fn worker_threads(mut self, worker_threads: usize) -> Self {
        self.worker_threads = Some(worker_threads);
        self
    }

    pub fn max_blocking_threads(mut self, max_blocking_threads: usize) -> Self {
        self.max_blocking_threads = Some(max_blocking_threads);
        self
    }

    pub fn max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = Some(max_concurrency);
        self
    }

    pub fn build(self) -> Result<Server<P>> {
        let parallelism = std::thread::available_parallelism()?.get();
        Ok(Server {
            plugin: self.plugin,
            address: self.address.unwrap_or("0.0.0.0:8080".to_string()),
            worker_threads: self.worker_threads.unwrap_or(1),
            max_blocking_threads: self.max_blocking_threads.unwrap_or(parallelism),
            max_concurrency: self
                .max_concurrency
                .unwrap_or(parallelism.saturating_sub(1).max(1)),
        })
    }
}

// c.f. https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html#limit-the-number-of-incoming-requests-being-handled-at-the-same-time
static SEMAPHORE: OnceLock<Semaphore> = OnceLock::new();

fn set_max_concurrency(n: usize) {
    SEMAPHORE.get_or_init(|| Semaphore::new(n));
}

async fn serve<P: ModelPlugin>(plugin: P, addr: &str) -> Result<()> {
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
