#![cfg(feature = "server")]

use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use guimauve::server::Server;
use serde_json::from_str;
use tower::ServiceExt;

use crate::common::{
    json_post,
    test_plugin::{Res, TestPlugin},
};

mod common;

async fn send(router: Router, req: Request<Body>) -> (StatusCode, String) {
    let res = router.oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn router(delay: Duration) -> Router {
    Server::builder(TestPlugin::new(delay))
        .build()
        .unwrap()
        .router()
}

#[tokio::test]
async fn health_returns_ok() {
    let req = Request::builder()
        .uri("/v1/health")
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(router(Duration::ZERO), req).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn infer_round_trips() {
    let req = json_post("/v1/infer", &serde_json::json!({"n": 21}));
    let (status, body) = send(router(Duration::ZERO), req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(from_str::<Res>(&body).unwrap(), Res { n: 42 });
}

#[tokio::test]
async fn plugin_error_is_500() {
    let req = json_post("/v1/infer", &serde_json::json!({"n": -1}));
    let (status, body) = send(router(Duration::ZERO), req).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body, "negative input");
}

#[tokio::test]
async fn custom_endpoints_are_honoured() {
    let inference_endpoint = "/predict";
    let health_endpoint = "/healthcheck";

    let router = Server::builder(TestPlugin::new(Duration::ZERO))
        .endpoint_inference(inference_endpoint)
        .endpoint_health(health_endpoint)
        .build()
        .unwrap()
        .router();

    let (status, _) = send(
        router.clone(),
        json_post(inference_endpoint, &serde_json::json!({ "n": 1 })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let healthcheck = Request::builder()
        .uri(health_endpoint)
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(router.clone(), healthcheck).await;
    assert_eq!(status, StatusCode::OK);
}
