use std::{collections::HashSet, hash::Hash};

use crate::config::Persistence;

// c.f. https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899/
/// c.f. https://github.com/changyaochen/rbo/blob/master/rbo/rbo.py
fn rank_biased_overlap_ext<T: Hash + Eq>(a: &[T], b: &[T], p: Persistence) -> f64 {
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
