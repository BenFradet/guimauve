#![cfg(feature = "server")]

mod common;

use std::{sync::atomic::Ordering, time::Duration};

use axum::http::StatusCode;
use guimauve::server::Server;
use tower::ServiceExt;

use crate::common::{json_post, test_plugin::TestPlugin};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overload_is_shed_with_503() {
    // queue wait dictates shedding after 20ms
    let router = Server::builder(TestPlugin::new(Duration::from_millis(300)))
        .max_concurrency(1)
        .max_queue_wait(Duration::from_millis(20))
        .build()
        .unwrap()
        .router();

    let slow = tokio::spawn({
        let router = router.clone();
        async move {
            router
                .oneshot(json_post("/v1/infer", &serde_json::json!({ "n": 1 })))
                .await
                .unwrap()
                .status()
        }
    });

    // first request takes the only permit
    tokio::time::sleep(Duration::from_millis(50)).await;

    let shed = router
        .oneshot(json_post("/v1/infer", &serde_json::json!({ "n": 2 })))
        .await
        .unwrap();

    assert_eq!(shed.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(slow.await.unwrap(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_concurrency_is_respected() {
    let plugin = TestPlugin::new(Duration::from_millis(50));
    let peak = plugin.peak_in_flight();

    // queue wait dictates queue don't shed
    let router = Server::builder(plugin)
        .max_concurrency(2)
        .max_queue_wait(Duration::from_secs(5))
        .build()
        .unwrap()
        .router();

    let mut handles = Vec::new();
    for n in 0..8 {
        let router = router.clone();
        handles.push(tokio::spawn(async move {
            router
                .oneshot(json_post("/v1/infer", &serde_json::json!({ "n": n })))
                .await
                .unwrap()
                .status()
        }));
    }
    for h in handles {
        assert_eq!(h.await.unwrap(), StatusCode::OK);
    }

    assert_eq!(peak.load(Ordering::SeqCst), 2);
}
