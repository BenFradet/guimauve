mod config;
mod diff;
mod json_pointer;
mod persistence;
mod ranking;
mod side;

use anyhow::{Context, Result, bail};
use clap::Parser;
use reqwest::Client;
use serde_json::Value;
use std::{fs::File, io::{BufRead, BufReader}, path::PathBuf, process::ExitCode};

use crate::config::Config;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    requests: PathBuf,
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let client = Client::new();

    let file = File::open(&args.requests)
        .with_context(|| format!("opening {}", args.requests.display()))?;

    let mut total = 0;
    let mut failed = 0;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let no = index + 1;
        let line = line.with_context(|| format!("reading line {no}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let body: Value =
            serde_json::from_str(&line).with_context(|| format!("line {no} is not JSON"))?;
        total += 1;

        let baseline = post(&client, &config.baseline.url, &body)
            .await
            .with_context(|| format!("line {no}: baseline"))?;
        let candidate = post(&client, &config.candidate.url, &body)
            .await
            .with_context(|| format!("line {no}: candidate"))?;

        let diff = config
            .compare(&baseline, &candidate)
            .with_context(|| format!("line {no}"))?;
        println!("line {no}");
        print!("{diff}");
        if !diff.passed() {
            failed += 1;
        }
    }

    println!("{}/{total} requests passed", total - failed);
    if failed == 0 {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

async fn post(client: &Client, url: &str, body: &Value) -> Result<Value> {
    let response = client
        .post(url)
        .json(body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("reading the response of {url}"))?;
    if !status.is_success() {
        bail!("{url} returned {status}: {text}");
    }
    serde_json::from_str(&text).with_context(|| format!("{url} return a non-json body: {text}"))
}
