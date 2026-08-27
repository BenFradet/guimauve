use std::{sync::Arc, time::Duration};

use anyhow::Result;
use axum::{
    Router,
    extract::{self, State},
    http::StatusCode,
    routing::{get, post},
};
use tokio::runtime::Builder;
use tokio::{net::TcpListener, signal, sync::Semaphore, task};

use crate::model_plugin::ModelPlugin;

pub struct Server<P: ModelPlugin> {
    plugin: P,
    address: String,
    endpoint_inference: String,
    endpoint_health: String,
    worker_threads: usize,
    max_blocking_threads: usize,
    max_concurrency: usize,
    max_queue_wait: Duration,
}

impl<P: ModelPlugin> Server<P> {
    pub fn builder(plugin: P) -> ServerBuilder<P> {
        ServerBuilder::new(plugin)
    }

    pub fn serve(self) -> Result<()> {
        tracing::info!(
            worker_threads = self.worker_threads,
            max_blocking_threads = self.max_blocking_threads,
            max_concurrency = self.max_concurrency,
            max_queue_wait = ?self.max_queue_wait,
            "runtime configuration",
        );

        Builder::new_multi_thread()
            .worker_threads(self.worker_threads)
            // to use in conjunction with spawn_blocking
            .max_blocking_threads(self.max_blocking_threads)
            .enable_all()
            .build()?
            .block_on(async {
                serve(
                    self.plugin,
                    &self.address,
                    &self.endpoint_inference,
                    &self.endpoint_health,
                    self.max_concurrency,
                    self.max_queue_wait,
                )
                .await
            })
    }
}

struct ServerState<P: ModelPlugin> {
    plugin: Arc<P>,
    semaphore: Arc<Semaphore>,
    max_queue_wait: Duration,
}

impl<P: ModelPlugin> ServerState<P> {
    fn new(plugin: P, max_concurrency: usize, max_queue_wait: Duration) -> Self {
        Self {
            plugin: Arc::new(plugin),
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
            max_queue_wait,
        }
    }
}

impl<P: ModelPlugin> Clone for ServerState<P> {
    fn clone(&self) -> Self {
        Self {
            plugin: Arc::clone(&self.plugin),
            semaphore: Arc::clone(&self.semaphore),
            max_queue_wait: self.max_queue_wait,
        }
    }
}

pub struct ServerBuilder<P: ModelPlugin> {
    plugin: P,
    address: Option<String>,
    endpoint_inference: Option<String>,
    endpoint_health: Option<String>,
    worker_threads: Option<usize>,
    max_blocking_threads: Option<usize>,
    max_concurrency: Option<usize>,
    max_queue_wait: Option<Duration>,
}

impl<P: ModelPlugin> ServerBuilder<P> {
    pub fn new(plugin: P) -> Self {
        Self {
            plugin,
            address: None,
            endpoint_inference: None,
            endpoint_health: None,
            worker_threads: None,
            max_blocking_threads: None,
            max_concurrency: None,
            max_queue_wait: None,
        }
    }

    pub fn address(mut self, address: impl Into<String>) -> Self {
        self.address = Some(address.into());
        self
    }

    pub fn endpoint_inference(mut self, endpoint_inference: impl Into<String>) -> Self {
        self.endpoint_inference = Some(endpoint_inference.into());
        self
    }

    pub fn endpoint_health(mut self, endpoint_health: impl Into<String>) -> Self {
        self.endpoint_health = Some(endpoint_health.into());
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

    /// Sets the server max queue wait: the time spent waiting for a permit from the semaphore.
    /// Sheds sustained overload while absorbing short bursts.
    /// Needs to be lower than client timeout.
    /// Defaults to 1 second.
    pub fn max_queue_wait(mut self, max_queue_wait: Duration) -> Self {
        self.max_queue_wait = Some(max_queue_wait);
        self
    }

    pub fn build(self) -> Result<Server<P>> {
        let parallelism = std::thread::available_parallelism()?.get();
        Ok(Server {
            plugin: self.plugin,
            address: self.address.unwrap_or("0.0.0.0:8080".to_string()),
            endpoint_inference: self.endpoint_inference.unwrap_or("/v1/infer".to_string()),
            endpoint_health: self.endpoint_health.unwrap_or("/v1/health".to_string()),
            worker_threads: self.worker_threads.unwrap_or(1),
            max_blocking_threads: self.max_blocking_threads.unwrap_or(parallelism),
            max_concurrency: self
                .max_concurrency
                .unwrap_or(parallelism.saturating_sub(1).max(1)),
            max_queue_wait: self.max_queue_wait.unwrap_or(Duration::from_secs(1)),
        })
    }
}

async fn serve<P: ModelPlugin>(
    plugin: P,
    addr: &str,
    endpoint_inference: &str,
    endpoint_health: &str,
    max_concurrency: usize,
    max_queue_wait: Duration,
) -> Result<()> {
    // c.f. https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html#limit-the-number-of-incoming-requests-being-handled-at-the-same-time
    let state = ServerState::new(plugin, max_concurrency, max_queue_wait);
    let app = Router::new()
        .route(endpoint_inference, post(infer::<P>))
        .route(endpoint_health, get(|| async { StatusCode::OK }))
        .with_state(state);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(
        address = listener.local_addr()?.to_string(),
        endpoint_inference = endpoint_inference,
        endpoint_health = endpoint_health,
        "listening on",
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn infer<P: ModelPlugin>(
    State(state): State<ServerState<P>>,
    extract::Json(payload): extract::Json<P::Request>,
) -> Result<extract::Json<P::Response>, (StatusCode, String)> {
    let permit = tokio::time::timeout(
        state.max_queue_wait,
        Arc::clone(&state.semaphore).acquire_owned(),
    )
    .await
    // waited too long for a permit
    .map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "timed out waiting for a permit".to_string(),
        )
    })?
    // semaphore closed
    .map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "server is shutting down".to_string(),
        )
    })?;
    let plugin = state.plugin;
    // this is heavily cpu bound, hence spawn_blocking
    let response = task::spawn_blocking(move || {
        // released when work ends
        let _permit = permit;
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
