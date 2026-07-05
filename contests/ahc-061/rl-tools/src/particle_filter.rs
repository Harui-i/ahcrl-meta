use tools::{Input, State};

use crate::official_compat::get_candidates;

const WA_MIN: f64 = 0.3;
const WA_MAX: f64 = 1.0;
const EPS_MIN: f64 = 0.1;
const EPS_MAX: f64 = 0.5;
const LIU_WEST_A: f64 = 0.98;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub wa: f64,
    pub wb: f64,
    pub wc: f64,
    pub wd: f64,
    pub eps: f64,
}

#[derive(Clone, Debug)]
pub struct MoveSummary {
    pub candidates: Vec<(usize, usize)>,
    pub observed_idx: usize,
}

impl MoveSummary {
    pub fn new(
        input: &Input,
        state: &State,
        player: usize,
        observed: (usize, usize),
    ) -> Option<Self> {
        let candidates = get_candidates(input, state, player);
        let observed_idx = candidates.iter().position(|&xy| xy == observed)?;
        Some(Self {
            candidates,
            observed_idx,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ParticleFilterSmc {
    particles: Vec<Particle>,
    weights: Vec<f64>,
    rng: SplitMix64,
}

impl ParticleFilterSmc {
    pub fn new(num_particles: usize, seed: u64) -> Self {
        let n = num_particles.max(1);
        let mut rng = SplitMix64::new(seed);
        let mut particles = Vec::with_capacity(n);
        for _ in 0..n {
            particles.push(Particle {
                wa: rng.uniform(WA_MIN, WA_MAX),
                wb: rng.uniform(WA_MIN, WA_MAX),
                wc: rng.uniform(WA_MIN, WA_MAX),
                wd: rng.uniform(WA_MIN, WA_MAX),
                eps: rng.uniform(EPS_MIN, EPS_MAX),
            });
        }
        Self {
            particles,
            weights: vec![1.0 / n as f64; n],
            rng,
        }
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn update(
        &mut self,
        input: &Input,
        state: &State,
        player: usize,
        observed: (usize, usize),
    ) {
        let Some(summary) = MoveSummary::new(input, state, player, observed) else {
            return;
        };
        self.update_with_summary(input, state, player, &summary);
    }

    pub fn update_with_candidates(
        &mut self,
        input: &Input,
        state: &State,
        player: usize,
        observed: (usize, usize),
        candidates: &[(usize, usize)],
    ) {
        let Some(observed_idx) = candidates.iter().position(|&xy| xy == observed) else {
            return;
        };
        let summary = MoveSummary {
            candidates: candidates.to_vec(),
            observed_idx,
        };
        self.update_with_summary(input, state, player, &summary);
    }

    fn update_with_summary(
        &mut self,
        input: &Input,
        state: &State,
        player: usize,
        summary: &MoveSummary,
    ) {
        if summary.candidates.is_empty() {
            return;
        }

        let mut max_log = f64::NEG_INFINITY;
        let mut logs = Vec::with_capacity(self.particles.len());
        for (particle, &weight) in self.particles.iter().zip(self.weights.iter()) {
            let p = move_probability(input, state, player, &summary, particle).max(1e-300);
            let log_w = weight.max(1e-300).ln() + p.ln();
            max_log = max_log.max(log_w);
            logs.push(log_w);
        }

        let mut sum = 0.0;
        for (weight, &log_w) in self.weights.iter_mut().zip(logs.iter()) {
            *weight = (log_w - max_log).exp();
            sum += *weight;
        }
        if !sum.is_finite() || sum <= 0.0 {
            let uniform = 1.0 / self.weights.len() as f64;
            self.weights.fill(uniform);
            return;
        }
        for weight in &mut self.weights {
            *weight /= sum;
        }

        if self.effective_sample_size() < 0.5 * self.particles.len() as f64 {
            self.resample_liu_west();
        }
    }

    pub fn posterior_mean(&self) -> Particle {
        let mut mean = Particle {
            wa: 0.0,
            wb: 0.0,
            wc: 0.0,
            wd: 0.0,
            eps: 0.0,
        };
        for (p, &w) in self.particles.iter().zip(self.weights.iter()) {
            mean.wa += w * p.wa;
            mean.wb += w * p.wb;
            mean.wc += w * p.wc;
            mean.wd += w * p.wd;
            mean.eps += w * p.eps;
        }
        mean
    }

    pub fn predictive_distribution(&self, input: &Input, state: &State, player: usize) -> Vec<f32> {
        let candidates = get_candidates(input, state, player);
        self.predictive_distribution_for_candidates(input, state, player, &candidates)
    }

    pub fn predictive_distribution_for_candidates(
        &self,
        input: &Input,
        state: &State,
        player: usize,
        candidates: &[(usize, usize)],
    ) -> Vec<f32> {
        let mut dist = vec![0.0_f64; input.N * input.N];
        if candidates.is_empty() {
            return vec![0.0; input.N * input.N];
        }
        for (particle, &weight) in self.particles.iter().zip(self.weights.iter()) {
            add_policy_distribution(
                input,
                state,
                player,
                &candidates,
                particle,
                weight,
                &mut dist,
            );
        }
        dist.into_iter().map(|v| v as f32).collect()
    }

    pub fn predictive_distribution_board100_for_candidates(
        &self,
        input: &Input,
        state: &State,
        player: usize,
        candidates: &[(usize, usize)],
    ) -> [f32; 100] {
        let mut out = [0.0_f32; 100];
        if candidates.is_empty() {
            return out;
        }
        let mut dist = [0.0_f64; 100];
        for (particle, &weight) in self.particles.iter().zip(self.weights.iter()) {
            add_policy_distribution(
                input, state, player, candidates, particle, weight, &mut dist,
            );
        }
        for idx in 0..100 {
            out[idx] = dist[idx] as f32;
        }
        out
    }

    fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.weights.iter().map(|w| w * w).sum();
        if sum_sq <= 0.0 {
            0.0
        } else {
            1.0 / sum_sq
        }
    }

    fn resample_liu_west(&mut self) {
        let n = self.particles.len();
        let mean = self.posterior_mean();
        let std = self.posterior_std(mean);
        let mut cumulative = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &w in &self.weights {
            acc += w;
            cumulative.push(acc);
        }
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }

        let step = 1.0 / n as f64;
        let mut u = self.rng.next_f64() * step;
        let h = (1.0 - LIU_WEST_A * LIU_WEST_A).sqrt();
        let mut idx = 0;
        let mut next_particles = Vec::with_capacity(n);
        for _ in 0..n {
            while idx + 1 < n && cumulative[idx] < u {
                idx += 1;
            }
            let base = self.particles[idx];
            next_particles.push(Particle {
                wa: jitter_param(base.wa, mean.wa, std.wa, h, &mut self.rng, WA_MIN, WA_MAX),
                wb: jitter_param(base.wb, mean.wb, std.wb, h, &mut self.rng, WA_MIN, WA_MAX),
                wc: jitter_param(base.wc, mean.wc, std.wc, h, &mut self.rng, WA_MIN, WA_MAX),
                wd: jitter_param(base.wd, mean.wd, std.wd, h, &mut self.rng, WA_MIN, WA_MAX),
                eps: jitter_param(
                    base.eps,
                    mean.eps,
                    std.eps,
                    h,
                    &mut self.rng,
                    EPS_MIN,
                    EPS_MAX,
                ),
            });
            u += step;
        }
        self.particles = next_particles;
        self.weights.fill(1.0 / n as f64);
    }

    fn posterior_std(&self, mean: Particle) -> Particle {
        let mut var = Particle {
            wa: 0.0,
            wb: 0.0,
            wc: 0.0,
            wd: 0.0,
            eps: 0.0,
        };
        for (p, &w) in self.particles.iter().zip(self.weights.iter()) {
            var.wa += w * (p.wa - mean.wa).powi(2);
            var.wb += w * (p.wb - mean.wb).powi(2);
            var.wc += w * (p.wc - mean.wc).powi(2);
            var.wd += w * (p.wd - mean.wd).powi(2);
            var.eps += w * (p.eps - mean.eps).powi(2);
        }
        Particle {
            wa: var.wa.max(0.0).sqrt(),
            wb: var.wb.max(0.0).sqrt(),
            wc: var.wc.max(0.0).sqrt(),
            wd: var.wd.max(0.0).sqrt(),
            eps: var.eps.max(0.0).sqrt(),
        }
    }
}

fn move_probability(
    input: &Input,
    state: &State,
    player: usize,
    summary: &MoveSummary,
    particle: &Particle,
) -> f64 {
    let mut dist = vec![0.0_f64; input.N * input.N];
    add_policy_distribution(
        input,
        state,
        player,
        &summary.candidates,
        particle,
        1.0,
        &mut dist,
    );
    let (x, y) = summary.candidates[summary.observed_idx];
    dist[x * input.N + y]
}

fn add_policy_distribution(
    input: &Input,
    state: &State,
    player: usize,
    candidates: &[(usize, usize)],
    particle: &Particle,
    weight: f64,
    dist: &mut [f64],
) {
    if candidates.is_empty() {
        return;
    }
    let random_prob = particle.eps / candidates.len() as f64;
    for &(x, y) in candidates {
        dist[x * input.N + y] += weight * random_prob;
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut scores = Vec::with_capacity(candidates.len());
    for &(x, y) in candidates {
        let score = particle_score(input, state, player, x, y, particle);
        best_score = best_score.max(score);
        scores.push(score);
    }
    let tolerance = 1e-9 * best_score.abs().max(1.0);
    let best_count = scores
        .iter()
        .filter(|&&score| score >= best_score - tolerance)
        .count()
        .max(1);
    let greedy_prob = (1.0 - particle.eps) / best_count as f64;
    for (&(x, y), &score) in candidates.iter().zip(scores.iter()) {
        if score >= best_score - tolerance {
            dist[x * input.N + y] += weight * greedy_prob;
        }
    }
}

fn particle_score(
    input: &Input,
    state: &State,
    player: usize,
    x: usize,
    y: usize,
    particle: &Particle,
) -> f64 {
    let owner = state.owner[x][y];
    let level = state.level[x][y];
    let value = input.V[x][y] as f64;
    if owner == -1 {
        value * particle.wa
    } else if owner == player as i32 {
        if level < input.U {
            value * particle.wb
        } else {
            0.0
        }
    } else if level == 1 {
        value * particle.wc
    } else {
        value * particle.wd
    }
}

fn jitter_param(
    value: f64,
    mean: f64,
    std: f64,
    h: f64,
    rng: &mut SplitMix64,
    low: f64,
    high: f64,
) -> f64 {
    let center = LIU_WEST_A * value + (1.0 - LIU_WEST_A) * mean;
    (center + h * std * rng.standard_normal()).clamp(low, high)
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare_normal = Some(radius * theta.sin());
        radius * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tools::gen;

    #[test]
    fn prior_particles_are_in_bounds_and_normalized() {
        let pf = ParticleFilterSmc::new(16, 1);
        assert_eq!(pf.len(), 16);
        assert!((pf.weights().iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for p in pf.particles() {
            assert!((WA_MIN..=WA_MAX).contains(&p.wa));
            assert!((WA_MIN..=WA_MAX).contains(&p.wb));
            assert!((WA_MIN..=WA_MAX).contains(&p.wc));
            assert!((WA_MIN..=WA_MAX).contains(&p.wd));
            assert!((EPS_MIN..=EPS_MAX).contains(&p.eps));
        }
    }

    #[test]
    fn move_summary_finds_observed_candidate() {
        let input = gen(0, Some(4), Some(3));
        let state = State::new(&input);
        let candidates = get_candidates(&input, &state, 1);
        let summary = MoveSummary::new(&input, &state, 1, candidates[0]).unwrap();
        assert_eq!(summary.candidates[summary.observed_idx], candidates[0]);
    }

    #[test]
    fn update_keeps_weights_normalized_and_predictive_finite() {
        let input = gen(0, Some(4), Some(3));
        let state = State::new(&input);
        let observed = get_candidates(&input, &state, 1)[0];
        let mut pf = ParticleFilterSmc::new(16, 2);
        pf.update(&input, &state, 1, observed);
        assert!((pf.weights().iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(pf.weights().iter().all(|w| w.is_finite() && *w >= 0.0));
        let dist = pf.predictive_distribution(&input, &state, 1);
        assert!(dist.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!((dist.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
