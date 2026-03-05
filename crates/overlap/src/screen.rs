use crate::basis::Shell;

/// Which precision tier a shell pair lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionTier {
    /// Both shells tight, or well-separated: f64 is fine.
    F64,
    /// Near-field diffuse-diffuse or diffuse-tight: use f128.
    F128,
    /// Integral negligible, skip entirely.
    Skip,
}

/// Tunable screening thresholds.
#[derive(Clone, Debug)]
pub struct ScreenParams {
    /// Exponent below which a primitive is "diffuse" (in bohr^-2).
    /// aug-cc-pVDZ diffuse exponents are typically 0.02-0.05.
    pub diffuse_threshold: f64,

    /// Exponent ratio alpha_max/alpha_min above which the pair is ill-conditioned.
    /// Ratios > ~1000 cause cancellation in the overlap sum.
    pub ratio_threshold: f64,

    /// Distance (bohr) beyond which the Schwarz-like bound guarantees
    /// the overlap is below machine epsilon. For a pair with geometric
    /// mean exponent gamma, overlap ~ exp(-gamma * R^2 / (alpha_i + alpha_j)).
    /// We skip when exp(-gamma_min * R^2 / 2) < eps_skip.
    pub eps_skip: f64,

    /// Distance beyond which even diffuse pairs can safely use f64.
    /// At large R the overlap is small and cancellation doesn't matter.
    pub safe_distance: f64,
}

impl Default for ScreenParams {
    fn default() -> Self {
        Self {
            diffuse_threshold: 0.1,
            ratio_threshold: 500.0,
            eps_skip: 1e-14,
            safe_distance: 12.0,  // bohr (~6.3 angstrom)
        }
    }
}

/// Decide the precision tier for a shell pair.
///
/// The logic:
/// 1. Compute R^2 between shell centers.
/// 2. Use the minimum exponents (most diffuse primitives) to estimate
///    the maximum possible overlap magnitude and cancellation risk.
/// 3. If the overlap bound is negligible -> Skip.
/// 4. If both shells are tight (alpha_min > threshold) -> F64.
/// 5. If at least one is diffuse AND they're near-field -> F128.
/// 6. If diffuse but far apart (overlap is small, cancellation harmless) -> F64.
pub fn screen_shell_pair(a: &Shell, b: &Shell, params: &ScreenParams) -> PrecisionTier {
    let r2 = dist2(&a.center, &b.center);

    let alpha_min_a = a.alpha_min();
    let alpha_min_b = b.alpha_min();
    let alpha_sum = alpha_min_a + alpha_min_b;

    // Schwarz-like bound: |S_pq| <= (pi / (alpha_i + alpha_j))^{3/2}
    //                               * exp(-alpha_i * alpha_j / (alpha_i + alpha_j) * R^2)
    let gamma = alpha_min_a * alpha_min_b / alpha_sum;
    let exponent = gamma * r2;

    // Skip if the overlap bound is negligible
    if exponent > -params.eps_skip.ln() {
        return PrecisionTier::Skip;
    }

    let a_diffuse = alpha_min_a < params.diffuse_threshold;
    let b_diffuse = alpha_min_b < params.diffuse_threshold;

    // Both tight -> f64
    if !a_diffuse && !b_diffuse {
        return PrecisionTier::F64;
    }

    // At least one diffuse. Check exponent ratio for cancellation risk.
    let alpha_max_a = a.alpha_max();
    let alpha_max_b = b.alpha_max();
    let ratio = (alpha_max_a / alpha_min_b).max(alpha_max_b / alpha_min_a);

    // Large ratio + near-field = catastrophic cancellation in the contraction sum
    let r = r2.sqrt();
    if ratio > params.ratio_threshold && r < params.safe_distance {
        return PrecisionTier::F128;
    }

    // Diffuse-diffuse near-field: the overlap itself is O(1) but the
    // primitives nearly cancel, so the contraction sum loses digits.
    if a_diffuse && b_diffuse && r < params.safe_distance {
        return PrecisionTier::F128;
    }

    // Far apart or moderate ratio: overlap is small, f64 is fine
    PrecisionTier::F64
}

#[inline]
fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Primitive, ShellKind};

    fn tight_s(center: [f64; 3]) -> Shell {
        Shell {
            center,
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 100.0, coeff: 0.5 },
                Primitive { alpha: 10.0, coeff: 0.5 },
            ],
        }
    }

    fn diffuse_s(center: [f64; 3]) -> Shell {
        Shell {
            center,
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 50.0, coeff: 0.3 },
                Primitive { alpha: 0.04, coeff: 0.7 },
            ],
        }
    }

    #[test]
    fn tight_tight_is_f64() {
        let a = tight_s([0.0; 3]);
        let b = tight_s([1.0, 0.0, 0.0]);
        assert_eq!(screen_shell_pair(&a, &b, &ScreenParams::default()), PrecisionTier::F64);
    }

    #[test]
    fn diffuse_diffuse_near_is_f128() {
        let a = diffuse_s([0.0; 3]);
        let b = diffuse_s([2.0, 0.0, 0.0]);
        assert_eq!(screen_shell_pair(&a, &b, &ScreenParams::default()), PrecisionTier::F128);
    }

    #[test]
    fn diffuse_diffuse_far_is_f64() {
        let a = diffuse_s([0.0; 3]);
        let b = diffuse_s([20.0, 0.0, 0.0]);
        let tier = screen_shell_pair(&a, &b, &ScreenParams::default());
        assert!(tier == PrecisionTier::F64 || tier == PrecisionTier::Skip);
    }

    #[test]
    fn well_separated_tight_skips() {
        let a = tight_s([0.0; 3]);
        let b = tight_s([50.0, 0.0, 0.0]);
        assert_eq!(screen_shell_pair(&a, &b, &ScreenParams::default()), PrecisionTier::Skip);
    }
}
