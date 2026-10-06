use std::{collections::HashSet, hash::Hash};

use crate::{json_pointer::JsonPointer, persistence::Persistence};

pub struct Comparison {
    pub pointer: JsonPointer,
    pub baseline_len: usize,
    pub candidate_len: usize,
    pub only_in_baseline: Vec<String>,
    pub only_in_candidate: Vec<String>,
    pub kendall_tau_a: Option<f64>,
    // count of discordant pairs over the intersection
    pub discordant: usize,
    pub rbo_ext: f64,
    // (k, overlap) for each configured top_k
    pub overlap_at_k: Vec<(usize, Option<f64>)>,
    // (key, baseline, candidate) for the largest score delta
    pub worst_score: Option<(String, f64, f64)>,
    // 1e-5 is the default
    pub scores_over_tolerance: usize,
}

impl Comparison {
    // TODO: add kendall and rbo thresholds?
    pub fn passed(&self) -> bool {
        self.only_in_baseline.is_empty()
            && self.only_in_candidate.is_empty()
            && self.scores_over_tolerance == 0
    }
}

pub struct Diff(pub Vec<Comparison>);

impl Diff {
    pub fn passed(&self) -> bool {
        self.0.iter().all(Comparison::passed)
    }
}

// c.f. https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899/
/// c.f. https://github.com/changyaochen/rbo/blob/master/rbo/rbo.py
pub fn rank_biased_overlap_ext<T: Hash + Eq>(a: &[T], b: &[T], p: Persistence) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.;
    }
    if a.is_empty() || b.is_empty() {
        return 0.;
    }

    let p = p.as_f64();

    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let (l, s) = (long.len(), short.len());

    let mut short_seen: HashSet<&T> = HashSet::with_capacity(s);
    short_seen.insert(&short[0]);
    let mut long_seen: HashSet<&T> = HashSet::with_capacity(l);
    long_seen.insert(&long[0]);

    let mut x = usize::from(short[0] == long[0]);
    let mut x_at_s = x;
    let mut rbo = (1. - p) * x as f64;
    // extrapolate term
    let mut ext_term = p * x as f64;
    let mut disjoint = 0.;
    let mut p_d = 1.; // p^d

    for d in 1..l {
        p_d *= p;
        let depth = (d + 1) as f64;
        if d < s {
            short_seen.insert(&short[d]);
            long_seen.insert(&long[d]);

            if short[d] == long[d] {
                x += 1;
            } else {
                x += usize::from(long_seen.contains(&short[d]));
                x += usize::from(short_seen.contains(&long[d]));
            }
            x_at_s = x;

            let agreement = x as f64 / depth;
            rbo += (1. - p) * p_d * agreement;
            ext_term = agreement * p_d * p;
        } else {
            long_seen.insert(&long[d]);
            x += usize::from(short_seen.contains(&long[d]));

            let agreement = x as f64 / depth;
            rbo += (1. - p) * p_d * agreement;

            let x_s = x_at_s as f64;
            let sf = s as f64;
            disjoint += (1. - p) * p_d * x_s * (depth - sf) / depth / sf;
            ext_term = ((x as f64 - x_s) / depth + x_s / sf) * p_d * p;
        }
    }

    (rbo + disjoint + ext_term).clamp(0., 1.)
}

// returns tau and number of discordant
// c.f. https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
pub fn kendall_tau_a<T: Ord>(seq: &[T]) -> (Option<f64>, usize) {
    let n = seq.len();
    let mut discordant = 0;
    for (i, a) in seq.iter().enumerate() {
        for b in &seq[i + 1..] {
            if a > b {
                discordant += 1;
            }
        }
    }
    let pairs = if n < 2 { 0 } else { n * (n - 1) / 2 };
    let tau = if pairs == 0 {
        None
    } else {
        let pf = pairs as f64;
        Some((pf - 2. * discordant as f64) / pf)
    };
    (tau, discordant)
}

// fraction of the top-k prefixes that are contained in both sequences
pub fn overlap_at<T: Hash + Eq>(a: &[T], b: &[T], k: usize) -> Option<f64> {
    let depth = k.min(a.len().max(b.len()));
    if depth == 0 {
        return None;
    }

    let head: HashSet<&T> = a.iter().take(k).collect();
    let hits = b.iter().take(k).filter(|x| head.contains(*x)).count();
    Some(hits as f64 / depth as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rbo_test_helper<T: Hash + Eq>(fst: &[T], snd: &[T], expected: f64) {
        let p: Persistence = 0.95.try_into().unwrap();
        let res = rank_biased_overlap_ext(fst, snd, p);
        let rounded = (res * 1000.).round() / 1000.;
        assert_eq!(rounded, expected);
    }

    fn kendall_test_helper<T: Ord>(seq: &[T], discordant_exp: usize, tau_exp: Option<f64>) {
        let (tau, discordant) = kendall_tau_a(seq);
        let tau_rounded = tau.map(|t| (t * 1000.).round() / 1000.);
        assert_eq!(tau_exp, tau_rounded);
        assert_eq!(discordant_exp, discordant);
    }

    #[test]
    fn kendall_tests() {
        kendall_test_helper(&[0, 1, 2, 3], 0, Some(1.));
        kendall_test_helper(&[3, 2, 1, 0], 6, Some(-1.));
        kendall_test_helper(&[1, 0, 2, 3], 1, Some(0.667));
        kendall_test_helper(&[0, 1, 3, 2], 1, Some(0.667));
        kendall_test_helper(&[0], 0, None);
        kendall_test_helper::<usize>(&[], 0, None);
        kendall_test_helper(&[0, 1], 0, Some(1.));
        kendall_test_helper(&[1, 0], 1, Some(-1.));
        kendall_test_helper(&[0, 2, 5], 0, Some(1.));
        kendall_test_helper(&[5, 2, 0], 3, Some(-1.));
        kendall_test_helper(&[1, 0, 3, 2, 4], 2, Some(0.6));
    }

    #[test]
    fn why_two_metrics() {
        // tau is position-blind, RBO weighs the head more heavily through persistence
        let p: Persistence = 0.95.try_into().unwrap();
        assert_eq!(
            kendall_tau_a(&[1, 0, 2, 3]).1,
            kendall_tau_a(&[0, 1, 3, 2]).1
        );
        assert!(
            rank_biased_overlap_ext(&["a", "b", "c", "d"], &["b", "a", "c", "d"], p)
                < rank_biased_overlap_ext(&["a", "b", "c", "d"], &["a", "b", "d", "c"], p)
        );
    }

    #[test]
    fn rbo_tests() {
        rbo_test_helper(&["a", "b", "c", "d", "e"], &["f", "g", "h", "i", "j"], 0.);
        rbo_test_helper(
            &["a", "b", "c", "d", "e", "f", "g"],
            &["z", "c", "a", "v", "w", "x", "y"],
            0.291,
        );
        rbo_test_helper(&["a", "b", "c", "d", "e"], &["b", "a", "c", "d", "e"], 0.95);
        rbo_test_helper(
            &["a", "b", "c", "d", "e"],
            &["a", "b", "c", "e", "d"],
            0.989,
        );

        rbo_test_helper(&["a"], &["b", "a", "c"], 0.325);
        rbo_test_helper(&["a", "b", "c", "d", "e"], &["a"], 1.);

        rbo_test_helper(&["a"], &["a"], 1.);
        rbo_test_helper(&["a"], &["b"], 0.);
        rbo_test_helper(&[""], &[""], 1.);
        rbo_test_helper(&["a"], &[""], 0.);
        rbo_test_helper(&[""], &["a"], 0.);
        rbo_test_helper::<usize>(&[], &[], 1.);
        rbo_test_helper(&[], &["a"], 0.);
        rbo_test_helper(&["a"], &[], 0.);
    }

    #[test]
    fn overlap_tests() {
        assert_eq!(Some(1.), overlap_at(&["a", "b", "c"], &["b", "a", "z"], 2));
        assert_eq!(
            Some(2. / 3.),
            overlap_at(&["a", "b", "c"], &["b", "a", "z"], 3)
        );
        assert_eq!(Some(0.), overlap_at(&["a"], &["b"], 1));
        assert_eq!(Some(0.25), overlap_at(&["a", "b", "c", "d"], &["a"], 4));
        assert_eq!(None, overlap_at::<usize>(&[], &[], 10));
        assert_eq!(Some(0.), overlap_at(&[], &["a"], 1));
    }
}
