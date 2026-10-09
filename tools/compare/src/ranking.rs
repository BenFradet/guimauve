use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    diff::{Comparison, kendall_tau_a, overlap_at, rank_biased_overlap_ext},
    json_pointer::JsonPointer,
    persistence::Persistence,
    side::Side,
};

struct Scored {
    pub keys: Vec<String>,
    pub scores: HashMap<String, f64>,
}

#[derive(Deserialize)]
pub struct Ranking {
    pub path: JsonPointer,
    pub key: String,
    pub score: String,
    pub persistence: Persistence,
    #[serde(default)]
    pub top_k: Vec<usize>,
}

impl Ranking {
    pub fn compare(
        &self,
        baseline: &Value,
        candidate: &Value,
        tolerance: f64,
    ) -> Result<Comparison> {
        let path = self.path.as_str();
        let pick = |side: Side, body: &Value| -> Result<Scored> {
            let at = body
                .pointer(path)
                .with_context(|| format!("{side} has nothing at {path}"))?;
            self.extract(at, side)
        };

        let baseline = pick(Side::Baseline, baseline)?;
        let candidate = pick(Side::Candidate, candidate)?;

        let only_in_baseline = Self::difference(&baseline, &candidate);
        let only_in_candidate = Self::difference(&candidate, &baseline);

        let ranks: HashMap<&str, usize> = baseline
            .keys
            .iter()
            .filter(|k| candidate.scores.contains_key(*k))
            .enumerate()
            .map(|(rank, k)| (k.as_str(), rank))
            .collect();

        let permutation: Vec<usize> = candidate
            .keys
            .iter()
            .filter_map(|k| ranks.get(k.as_str()).copied())
            .collect();
        let (kendall_tau_a, discordant) = kendall_tau_a(&permutation);

        let rbo_ext = rank_biased_overlap_ext(&baseline.keys, &candidate.keys, self.persistence);

        let overlap_at_k: Vec<(usize, Option<f64>)> = self
            .top_k
            .iter()
            .map(|&k| (k, overlap_at(&baseline.keys, &candidate.keys, k)))
            .collect();

        let mut worst_score = None;
        let mut scores_over_tolerance = 0;
        for key in &baseline.keys {
            let Some(&b) = baseline.scores.get(key) else {
                continue;
            };
            let Some(&c) = candidate.scores.get(key) else {
                continue;
            };
            let delta = (b - c).abs();
            if delta > tolerance {
                scores_over_tolerance += 1;
            }
            if worst_score
                .as_ref()
                .is_none_or(|(_, wb, wc): &(String, f64, f64)| delta > (wb - wc).abs())
            {
                worst_score = Some((key.clone(), b, c));
            }
        }

        Ok(Comparison {
            pointer: self.path.clone(),
            baseline_len: baseline.keys.len(),
            candidate_len: candidate.keys.len(),
            only_in_baseline,
            only_in_candidate,
            kendall_tau_a,
            discordant,
            rbo_ext,
            overlap_at_k,
            worst_score,
            scores_over_tolerance,
        })
    }

    fn extract(&self, value: &Value, side: Side) -> Result<Scored> {
        let path = self.path.as_str();
        let items = value.as_array().with_context(|| {
            format!(
                "{side} {path} is a {}, expected an array",
                Self::kind(value),
            )
        })?;

        let mut keys = Vec::with_capacity(items.len());
        let mut scores = HashMap::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let key = item
                .get(&self.key)
                .and_then(|v| v.as_str())
                .with_context(|| format!("{side} {path}/{i}: no string field {:?}", self.key))?;
            let score = item
                .get(&self.score)
                .and_then(Value::as_f64)
                .with_context(|| format!("{side} {path}/{i}: no f64 field {:?}", self.score))?;

            if scores.insert(key.to_string(), score).is_some() {
                bail!("{side} {path} contains {} = {key:?} twice", self.key);
            }
            keys.push(key.to_string());
        }
        Ok(Scored { keys, scores })
    }

    fn difference(from: &Scored, against: &Scored) -> Vec<String> {
        let mut keys: Vec<String> = from
            .keys
            .iter()
            .filter(|k| !against.scores.contains_key(*k))
            .cloned()
            .collect();
        keys.sort();
        keys
    }

    fn kind(value: &Value) -> &'static str {
        match value {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
}
