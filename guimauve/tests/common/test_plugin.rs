use std::{
    sync::{Arc, atomic::{AtomicUsize, Ordering}}, thread, time::Duration,
};

use guimauve::model_plugin::ModelPlugin;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Req {
    pub n: i64,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Res {
    pub n: i64,
}

pub struct TestPlugin {
    infer_delay: Duration,
    in_flight: Arc<AtomicUsize>,
    peak_in_flight: Arc<AtomicUsize>,
}

impl TestPlugin {
    pub fn new(infer_delay: Duration) -> Self {
        Self {
            infer_delay,
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak_in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn peak_in_flight(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.peak_in_flight)
    }
}

impl ModelPlugin for TestPlugin {
    type Error = String;
    type Request = Req;
    type Response = Res;
    type ModelInput = i64;
    type ModelOutput = i64;

    fn pre(&self, req: Self::Request) -> Result<i64, String> {
        if req.n < 0 {
            Err("negative input".to_string())
        } else {
            Ok(req.n)
        }
    }

    fn infer(&self, input: Self::ModelInput) -> Result<Self::ModelOutput, Self::Error> {
        let ordering = Ordering::SeqCst;
        let now = self.in_flight.fetch_add(1, ordering) + 1;
        self.peak_in_flight.fetch_max(now, ordering);
        thread::sleep(self.infer_delay);
        self.in_flight.fetch_sub(1, ordering);
        Ok(input * 2)
    }

    fn post(&self, output: Self::ModelOutput) -> Result<Self::Response, Self::Error> {
        Ok(Res { n: output })
    }
}
