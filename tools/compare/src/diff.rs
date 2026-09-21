use std::collections::HashSet;

use crate::config::Persistence;

// c.f. https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899/
/// c.f. https://github.com/changyaochen/rbo/blob/master/rbo/rbo.py
fn rank_biased_overlap_ext(a: &[String], b: &[String], p: Persistence) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.;
    }
    if a.is_empty() || b.is_empty() {
        return 0.;
    }
    
    let p = p.as_f64();

    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let (l, s) = (long.len(), short.len());

    let mut x = vec![0; l];
    let mut a = vec![0.; l];
    let mut rbo = vec![0.; l];

    let mut short_seen: HashSet<&str> = HashSet::with_capacity(s);
    short_seen.insert(&short[0]);
    let mut long_seen: HashSet<&str> = HashSet::with_capacity(l);
    long_seen.insert(&long[0]);

    x[0] = if short[0] == long[0] { 1 } else { 0 };
    a[0] = x[0] as f64;
    rbo[0] = (1. - p) * a[0];

    let mut disjoint = 0.;
    // extrapolate term
    let mut ext_term = p * a[0];

    for d in 1..l {
        if d < s {
            short_seen.insert(&short[d]);
            long_seen.insert(&long[d]);

            let mut overlap_inc = 0;

            if short[d] == long[d] {
                overlap_inc += 1;
            } else {
                if long_seen.contains(short[d].as_str()) {
                    overlap_inc += 1;
                }
                if short_seen.contains(long[d].as_str()) {
                    overlap_inc += 1;
                }
            }

            x[d] = x[d - 1] + overlap_inc;
            a[d] = 2. * x[d] as f64 / (short_seen.len() as f64 + long_seen.len() as f64);
            rbo[d] = rbo[d - 1] + (1. - p) * p.powf(d as f64) * a[d];

            ext_term = a[d] * p.powf(d as f64 + 1.);
        } else {
            long_seen.insert(&long[d]);

            let overlap_inc = if short_seen.contains(long[d].as_str()) {
                1
            } else {
                0
            };

            x[d] = x[d - 1] + overlap_inc;
            a[d] = x[d] as f64 / (d as f64 + 1.);
            rbo[d] = rbo[d - 1] + (1. - p) * p.powf(d as f64) * a[d];

            // last common overlap
            let x_short = x[s - 1] as f64;

            disjoint += (1. - p)
                * p.powf(d as f64)
                * (x_short as f64 * (d as f64 + 1. - s as f64) / (d as f64 + 1.) / s as f64);
            ext_term = ((x[d] as f64 - x_short) / (d as f64 + 1.) + x[s - 1] as f64 / s as f64)
                * p.powf(d as f64 + 1.);
        }
    }

    (rbo[rbo.len() - 1] + disjoint + ext_term).clamp(0., 1.)
}
