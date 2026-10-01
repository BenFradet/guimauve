pub enum Side {
    Baseline,
    Candidate,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Side::Baseline => "baseline",
            Side::Candidate => "candidate",
        };
        write!(f, "{str}")
    }
}
