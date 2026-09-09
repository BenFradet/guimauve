use axum::{
    body::Body,
    http::{Request, header::CONTENT_TYPE},
};
use serde_json::{Value, to_vec};

pub mod test_plugin;

pub fn json_post(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(to_vec(body).unwrap()))
        .unwrap()
}
