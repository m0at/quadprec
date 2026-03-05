use rayon::prelude::*;
use crate::basis::BasisSet;
use crate::screen::{PrecisionTier, ScreenParams, screen_shell_pair};
use crate::integral::overlap_shell_pair;

/// Result of mixed-precision overlap assembly.
pub struct OverlapResult {
    /// The n_ao x n_ao overlap matrix in row-major order.
    pub matrix: Vec<f64>,
    /// Number of AOs.
    pub n_ao: usize,
    /// How many shell pairs used each tier.
    pub tier_counts: TierCounts,
}

#[derive(Default, Debug)]
pub struct TierCounts {
    pub f64_pairs: usize,
    pub f128_pairs: usize,
    pub skipped_pairs: usize,
}

/// Assemble the overlap matrix S using mixed precision.
///
/// Strategy:
/// 1. Generate all unique shell pairs (i, j) with i >= j (lower triangle).
/// 2. Screen each pair to decide precision tier.
/// 3. Compute shell-pair blocks in parallel via Rayon work-stealing.
/// 4. Scatter results into the full matrix (symmetric: S_ij = S_ji).
///
/// The matrix is stored as f64 regardless of compute precision.
/// For pairs computed in f128, the double-double result is downcast
/// to f64 after the contraction sum -- the precision benefit is in
/// avoiding cancellation during summation, not in the final storage.
pub fn assemble_overlap_mixed(basis: &BasisSet, params: &ScreenParams) -> OverlapResult {
    let n_ao = basis.n_ao();
    let offsets = basis.shell_offsets();
    let n_shells = basis.shells.len();

    // Generate shell pair indices (upper triangle including diagonal)
    let pairs: Vec<(usize, usize)> = (0..n_shells)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .collect();

    // Screen all pairs and compute in parallel
    let results: Vec<(usize, usize, PrecisionTier, Vec<f64>)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let si = &basis.shells[i];
            let sj = &basis.shells[j];
            let tier = screen_shell_pair(si, sj, params);
            let block = overlap_shell_pair(si, sj, tier);
            (i, j, tier, block)
        })
        .collect();

    // Scatter into full matrix
    let mut matrix = vec![0.0_f64; n_ao * n_ao];
    let mut counts = TierCounts::default();

    for (i, j, tier, block) in &results {
        let oi = offsets[*i];
        let oj = offsets[*j];
        let ni = basis.shells[*i].size();
        let nj = basis.shells[*j].size();

        match tier {
            PrecisionTier::F64 => counts.f64_pairs += 1,
            PrecisionTier::F128 => counts.f128_pairs += 1,
            PrecisionTier::Skip => { counts.skipped_pairs += 1; continue; }
        }

        for ai in 0..ni {
            for aj in 0..nj {
                let val = block[ai * nj + aj];
                matrix[(oi + ai) * n_ao + (oj + aj)] = val;
                if *i != *j {
                    matrix[(oj + aj) * n_ao + (oi + ai)] = val;
                }
            }
        }
    }

    OverlapResult { matrix, n_ao, tier_counts: counts }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Primitive, Shell, ShellKind};

    fn h2_basis() -> BasisSet {
        // Minimal STO-3G-like basis for H2: 3 primitives per atom, s-type only.
        let norm = |alpha: f64| (2.0 * alpha / std::f64::consts::PI).powf(0.75);
        let h1 = Shell {
            center: [0.0, 0.0, 0.0],
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 3.42525, coeff: 0.15433 * norm(3.42525) },
                Primitive { alpha: 0.62391, coeff: 0.53533 * norm(0.62391) },
                Primitive { alpha: 0.16886, coeff: 0.44463 * norm(0.16886) },
            ],
        };
        let h2 = Shell {
            center: [1.4, 0.0, 0.0], // ~0.74 angstrom in bohr
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 3.42525, coeff: 0.15433 * norm(3.42525) },
                Primitive { alpha: 0.62391, coeff: 0.53533 * norm(0.62391) },
                Primitive { alpha: 0.16886, coeff: 0.44463 * norm(0.16886) },
            ],
        };
        BasisSet { shells: vec![h1, h2] }
    }

    #[test]
    fn h2_overlap_symmetric() {
        let basis = h2_basis();
        let result = assemble_overlap_mixed(&basis, &ScreenParams::default());
        assert_eq!(result.n_ao, 2);
        let s = &result.matrix;
        // Symmetric
        assert!((s[0 * 2 + 1] - s[1 * 2 + 0]).abs() < 1e-15);
        // Diagonal close to 1 (normalized basis)
        assert!((s[0] - 1.0).abs() < 0.02, "S_11 = {}", s[0]);
        assert!((s[3] - 1.0).abs() < 0.02, "S_22 = {}", s[3]);
        // Off-diagonal: known STO-3G H2 overlap ~ 0.66
        assert!(s[1] > 0.5 && s[1] < 0.8,
            "S_12 = {} (expected ~0.66)", s[1]);
    }

    #[test]
    fn all_tight_uses_f64() {
        let basis = h2_basis();
        let result = assemble_overlap_mixed(&basis, &ScreenParams::default());
        // STO-3G has no diffuse functions -> all pairs should be F64
        assert_eq!(result.tier_counts.f128_pairs, 0);
        assert!(result.tier_counts.f64_pairs > 0);
    }

    #[test]
    fn diffuse_triggers_f128() {
        let norm = |alpha: f64| (2.0 * alpha / std::f64::consts::PI).powf(0.75);
        let s1 = Shell {
            center: [0.0; 3],
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 50.0, coeff: 0.3 * norm(50.0) },
                Primitive { alpha: 0.04, coeff: 0.7 * norm(0.04) },
            ],
        };
        let s2 = Shell {
            center: [2.0, 0.0, 0.0],
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 50.0, coeff: 0.3 * norm(50.0) },
                Primitive { alpha: 0.04, coeff: 0.7 * norm(0.04) },
            ],
        };
        let basis = BasisSet { shells: vec![s1, s2] };
        let result = assemble_overlap_mixed(&basis, &ScreenParams::default());
        assert!(result.tier_counts.f128_pairs > 0,
            "Diffuse pairs should trigger f128");
    }
}
