use quad::{f128, cholesky, solve_cholesky, matvec, cond_estimate};

/// Result of solving Ax = b with automatic precision selection.
pub struct Solution {
    /// Solution vector in quad precision. Use this when kappa(A) > ~10^10.
    pub x: Vec<f128>,
    /// Solution rounded to f64 (convenience; loses precision for ill-conditioned systems).
    pub x_f64: Vec<f64>,
    /// Estimated condition number of A.
    pub kappa: f64,
    /// 2-norm of the residual b - Ax (computed in f128).
    pub residual_norm: f64,
    /// Which strategy was used.
    pub strategy: Strategy,
    /// Number of iterative refinement steps (0 if pure f128).
    pub refinement_iters: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Float64,
    MixedPrecision,
    Float128,
}

/// Solve A*x = b where A is symmetric positive definite.
///
/// Automatically selects the cheapest precision strategy that converges:
/// 1. Try f64 Cholesky. If it succeeds and the system is well-conditioned, done.
/// 2. If ill-conditioned, do iterative refinement with f128 corrections.
/// 3. If f64 Cholesky fails entirely, fall back to full f128 factorization.
pub fn solve_spd(a_f64: &[f64], b_f64: &[f64], n: usize) -> Result<Solution, SolveError> {
    assert_eq!(a_f64.len(), n * n);
    assert_eq!(b_f64.len(), n);

    let a128 = promote_matrix(a_f64);
    let b128: Vec<f128> = b_f64.iter().map(|&v| f128::from_f64(v)).collect();

    if let Some(l64) = cholesky_f64(a_f64, n) {
        let x64 = solve_cholesky_f64(&l64, b_f64, n);
        let kappa = cond_estimate_f64(&l64, n);

        // Well-conditioned: f64 is sufficient
        if kappa < 1e10 {
            let residual = residual_norm_f128(&a128, &x64, &b128, n);
            let x128: Vec<f128> = x64.iter().map(|&v| f128::from_f64(v)).collect();
            return Ok(Solution {
                x: x128,
                x_f64: x64,
                kappa,
                residual_norm: residual,
                strategy: Strategy::Float64,
                refinement_iters: 0,
            });
        }

        // Ill-conditioned but f64 Cholesky succeeded → iterative refinement
        return iterative_refinement(&a128, &b128, x64, n);
    }

    // f64 Cholesky failed → full f128
    solve_pure_f128(&a128, &b128, n)
}

/// Solve when you already have the matrix in f128 (e.g. from mixed-precision assembly).
pub fn solve_spd_f128(a: &[f128], b: &[f128], n: usize) -> Result<Solution, SolveError> {
    solve_pure_f128(a, b, n)
}

fn iterative_refinement(
    a128: &[f128],
    b128: &[f128],
    x_init: Vec<f64>,
    n: usize,
) -> Result<Solution, SolveError> {
    let mut l128 = a128.to_vec();
    cholesky(&mut l128, n).map_err(|e| SolveError::NotPositiveDefinite { pivot: e.pivot })?;
    let kappa = cond_estimate(&l128, n);

    // Accumulate in f128 — critical for kappa > 10^10
    let mut x128: Vec<f128> = x_init.iter().map(|&v| f128::from_f64(v)).collect();

    let bnorm = norm_f128(b128);
    let mut best_residual = f64::MAX;
    let mut iters = 0;

    for _ in 0..30 {
        let ax = matvec(a128, &x128, n);
        let r: Vec<f128> = b128.iter().zip(ax.iter()).map(|(bi, ai)| *bi - *ai).collect();
        let rnorm = norm_f128(&r);
        iters += 1;

        if rnorm < 1e-28 * bnorm {
            break;
        }
        if iters > 3 && rnorm > 0.5 * best_residual {
            break;
        }
        if rnorm < best_residual {
            best_residual = rnorm;
        }

        let dx = solve_cholesky(&l128, &r, n);
        for i in 0..n {
            x128[i] = x128[i] + dx[i];
        }
    }

    let x_f64: Vec<f64> = x128.iter().map(|v| v.to_f64()).collect();
    let residual = residual_norm_f128_from_f128(a128, &x128, b128, n);

    Ok(Solution {
        x: x128,
        x_f64,
        kappa,
        residual_norm: residual,
        strategy: Strategy::MixedPrecision,
        refinement_iters: iters,
    })
}

fn solve_pure_f128(a128: &[f128], b128: &[f128], n: usize) -> Result<Solution, SolveError> {
    let mut l = a128.to_vec();
    cholesky(&mut l, n).map_err(|e| SolveError::NotPositiveDefinite { pivot: e.pivot })?;
    let kappa = cond_estimate(&l, n);
    let x128 = solve_cholesky(&l, b128, n);

    let x_f64: Vec<f64> = x128.iter().map(|v| v.to_f64()).collect();
    let residual = residual_norm_f128_from_f128(a128, &x128, b128, n);

    Ok(Solution {
        x: x128,
        x_f64,
        kappa,
        residual_norm: residual,
        strategy: Strategy::Float128,
        refinement_iters: 0,
    })
}

// ---- f64 helpers ----

fn cholesky_f64(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = a.to_vec();
    for j in 0..n {
        let mut sum = 0.0f64;
        for k in 0..j {
            sum += l[j * n + k] * l[j * n + k];
        }
        let diag = l[j * n + j] - sum;
        if diag <= 0.0 {
            return None;
        }
        let ljj = diag.sqrt();
        l[j * n + j] = ljj;
        let inv = 1.0 / ljj;
        for i in (j + 1)..n {
            let mut s = 0.0f64;
            for k in 0..j {
                s += l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = (l[i * n + j] - s) * inv;
        }
        for i in 0..j {
            l[i * n + j] = 0.0;
        }
    }
    Some(l)
}

fn solve_cholesky_f64(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = b.to_vec();
    for i in 0..n {
        for j in 0..i {
            y[i] -= l[i * n + j] * y[j];
        }
        y[i] /= l[i * n + i];
    }
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            y[i] -= l[j * n + i] * y[j];
        }
        y[i] /= l[i * n + i];
    }
    y
}

fn cond_estimate_f64(l: &[f64], n: usize) -> f64 {
    let mut dmax = 0.0f64;
    let mut dmin = f64::MAX;
    for i in 0..n {
        let d = l[i * n + i].abs();
        dmax = dmax.max(d);
        dmin = dmin.min(d);
    }
    (dmax / dmin).powi(2)
}

// ---- Precision bridge ----

fn promote_matrix(a: &[f64]) -> Vec<f128> {
    a.iter().map(|&v| f128::from_f64(v)).collect()
}

fn residual_norm_f128(a: &[f128], x_f64: &[f64], b: &[f128], n: usize) -> f64 {
    let x128: Vec<f128> = x_f64.iter().map(|&v| f128::from_f64(v)).collect();
    residual_norm_f128_from_f128(a, &x128, b, n)
}

fn residual_norm_f128_from_f128(a: &[f128], x: &[f128], b: &[f128], n: usize) -> f64 {
    let ax = matvec(a, x, n);
    let r: Vec<f128> = b.iter().zip(ax.iter()).map(|(bi, ai)| *bi - *ai).collect();
    norm_f128(&r)
}

fn norm_f128(x: &[f128]) -> f64 {
    let mut s = f128::ZERO;
    for &v in x {
        s = s + v * v;
    }
    s.sqrt().to_f64()
}

#[derive(Debug)]
pub enum SolveError {
    NotPositiveDefinite { pivot: usize },
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::NotPositiveDefinite { pivot } =>
                write!(f, "Matrix is not positive definite (failed at pivot {pivot})")
        }
    }
}

impl std::error::Error for SolveError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_well_conditioned() {
        let n = 10;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 1.0 / (i + j + 1) as f64;
            }
            a[i * n + i] += 1.0;
        }
        let b = vec![1.0; n];
        let sol = solve_spd(&a, &b, n).unwrap();
        assert_eq!(sol.strategy, Strategy::Float64);
        assert!(sol.residual_norm < 1e-12);
    }

    #[test]
    fn solve_hilbert_13() {
        let n = 13;
        let a = hilbert_matrix(n);
        let b = vec![1.0; n];
        let sol = solve_spd(&a, &b, n).unwrap();
        assert_eq!(sol.strategy, Strategy::Float128);
        assert!(sol.kappa > 1e12);
    }

    #[test]
    fn solve_hilbert_12_mixed() {
        // kappa ~ 10^13. f64 Cholesky works but solution quality is limited
        // by input precision. Solver should detect ill-conditioning and refine.
        let n = 12;
        let a = hilbert_matrix(n);
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i * n + j];
            }
        }
        let sol = solve_spd(&a, &b, n).unwrap();
        assert!(sol.strategy == Strategy::MixedPrecision,
            "Expected MixedPrecision, got {:?}", sol.strategy);
        // Residual should be near f128 machine zero (solver works perfectly)
        assert!(sol.residual_norm < 1e-25,
            "residual too large: {:.2e}", sol.residual_norm);
    }

    #[test]
    fn solve_hilbert_12_f128_input() {
        // Same system, but compute b = A*ones in f128 to eliminate input rounding.
        // NOW the solution should be accurate to ~kappa * eps_128 ~ 10^13 * 10^-32 = 10^-19.
        let n = 12;
        let a_f64 = hilbert_matrix(n);
        let a128: Vec<f128> = a_f64.iter().map(|&v| f128::from_f64(v)).collect();
        // b = A * ones computed in f128 (exact for this input)
        let ones = vec![f128::ONE; n];
        let mut b128 = vec![f128::ZERO; n];
        for i in 0..n {
            for j in 0..n {
                b128[i] = b128[i] + a128[i * n + j] * ones[j];
            }
        }
        let sol = solve_spd_f128(&a128, &b128, n).unwrap();
        let err: f64 = sol.x.iter()
            .map(|xi| (*xi - f128::ONE).to_f64().powi(2))
            .sum::<f64>().sqrt();
        assert!(err < 1e-15,
            "f128 solution error with f128 input: {err:e}");
    }

    fn hilbert_matrix(n: usize) -> Vec<f64> {
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 1.0 / (i + j + 1) as f64;
            }
        }
        a
    }
}
