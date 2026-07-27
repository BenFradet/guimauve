<h2 align="center">
  <img src="logo.png" alt="log" width="212" height="350"/>
  <br>
  guimauve
</h2>

`guimauve`, French for marshmallow, is an inference server built on top of:
- [burn](https://github.com/tracel-ai/burn)
- [axum](https://github.com/tokio-rs/axum)

## Quickstart

Add `guimauve` as a dependency:

```bash
cargo add guimauve
```

Implement the `ModelPlugin` trait:

```rust
// lib.rs
use burn::tensor::{Int, Tensor};
use guimauve::model_plugin::ModelPlugin;

struct MyPlugin;

impl ModelPlugin for MyPlugin {
    type Request = serde_json::Value;
    type Response = serde_json::Value;
    type ModelInput = Tensor<Flex, 2, Int>;
    type ModelOutput = Tensor<Flex, 2, Int>;
    type Error = anyhow::Error;

    fn pre(&self, req: Self::Request) -> Result<Self::ModelInput, Self::Error> {
        // parse and prepare model input
        todo!()
    }

    fn infer(&self, input: Self::ModelInput) -> Result<Self::ModelOutput, Self::Error> {
        // run inference
        todo!()
    }

    fn post(&self, output: Self::ModelOutput) -> Result<Self::Response, Self::Error> {
        // format response
        todo!()
    }
}
```

Define your entrypoint:

```rust
// main.rs
fn main() -> anyhow::Result<()> {
    let cpus = std::thread::available_parallelism()?.get();
    guimauve::server::set_max_concurrency(cpus.saturating_sub(1).max(1));

    let plugin = MyPlugin;

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(cpus)
        .build()?
        .block_on(async { guimauve::server::serve(plugin, "0.0.0.0:3000").await })
}
```

Two endpoints are available:

```bash
curl http://0.0.0.0:3000/health
# inference
curl -X POST http://0.0.0.0:3000/infer \
    -H 'Content-Type: application/json' \
    -d '{"en_sentence": "why are people from Lisboa eating snails?"}'
```

You can check the tutorial below for a full-fledged example.

## Tutorial

This is a small tutorial to run an inference server on an English to Portuguese translation model.

Download the datasets from `https://web.archive.org/web/20240301220426if_/http://www.phontron.com/data/qi18naacl-dataset.tar.gz`
and extract them. We only care about the `pt_to_en` folder.

### Train tokenizers

From the `python-onnx-example` folder, run:

```bash
uv run \
    -m train.train_tokenizer \
    -i datasets/pt_to_en/pt.dev \
        datasets/pt_to_en/pt.test \
        datasets/pt_to_en/pt.train \
        datasets/pt_to_en/pt.train.r0.125 \
        datasets/pt_to_en/pt.train.r0.25 \
        datasets/pt_to_en/pt.train.r0.5 \
    -o pt_tokenizer.json \
    --max-seq-len 128
```

This will create a `pt_tokenizer.json` file.

Same thing for the English data which will create a `en_tokenizer.json` file.

### Train the transformer model

From the `python-onnx-example` folder, run:

```bash
uv run \
    -m train.train_transformer \
    --en-tokenizer models/pt_to_en/en_tokenizer.json \
    --en-train datasets/pt_to_en/en.train \
    --en-val datasets/pt_to_en/en.dev \
    --en-test datasets/pt_to_en/en.test \
    --pt-tokenizer models/pt_to_en/pt_tokenizer.json \
    --pt-train datasets/pt_to_en/pt.train \
    --pt-val datasets/pt_to_en/pt.dev \
    --pt-test datasets/pt_to_en/pt.test \
    --output-dir models/pt_to_en \
    --debug
```

This might take a while depending on your computer specs.
This will output a `transformer.onnx` file.

### Building the inference server docker image

Modify ./crates/onnx-example/build.rs and point the path to where your `transformer.onnx` file is
located.

Build the Docker image, `ARTIFACTS_DIR` should contain the tokenizer files built previously. 

```bash
docker build \
    --build-arg CRATE_NAME=onnx-example \
    --build-arg ARTIFACTS_DIR=models/pt_to_en \
    -t onnx-example .
```

### Run the Docker container

```bash
docker run -p 3000:3000 --name onnx-example onnx-example
```

Check it's working:

```bash
curl -X POST localhost:3000/infer \
    -H 'Content-Type: application/json' \
    -d '{"en_sentence": "why are people from Lisboa eating snails?"}'
```

### Docker compose set up

There is also a Docker compose set up with:

- [cadvisor](https://github.com/google/cadvisor) for resource usage monitoring
- [prometheus](https://github.com/prometheus/prometheus) for the time series database set up to scrape cadvisor
- [grafana](https://github.com/grafana/grafana) to display the data in prometheus in a dashboard

```
CRATE_NAME=onnx-example \
  ARTIFACTS_DIR=models/pt_to_en \
  docker compose up --force-recreate --remove-orphans --detach --build guimauve
```

You can check the Grafana dashboard at `http://localhost:8083/`.

## dbx-to-onnx

If you have a model already saved locally or in Databricks using MLflow, this repository offers
a small utility to convert it to ONNX:

```bash
uv run dbx-to-onnx \
    -m models/pt_to_en \
    -i "source:int64:1,128" \
    -i "target:int64:1,127" \
    -o models/pt_to_en
```

This will create a `model.onnx` file in the specified output directory.

There is more information in the dedicated [README](./dbx-to-onnx/README.md).
