use quad::f128;

/// Orthogonal basis from canonical orthogonalization of overlap matrix S.
pub struct OrthoBasis {
    /// Transformation matrix X (n_orig x n_orth), column-major conceptually but stored row-major.
    pub x_matrix: Vec<f128>,
    pub n_orig: usize,
    pub n_orth: usize,
}

/// Result of solving the generalized eigenvalue problem F C = S C eps.
pub struct GenEigenResult {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f128>,
    /// Eigenvectors as column-major-in-row-major: C[i * n_basis + k] = component i of eigenvector k.
    pub eigenvectors: Vec<f128>,
    /// Number of basis functions after removing linear dependencies.
    pub n_basis: usize,
}

// ---- Minimal inline utilities ----

/// C = A^T * B, A is m x n row-major, B is m x p row-major, result is n x p.
fn gemm_atb(a: &[f128], b: &[f128], m: usize, n: usize, p: usize) -> Vec<f128> {
    let mut c = vec![f128::ZERO; n * p];
    for k in 0..m {
        for i in 0..n {
            let aki = a[k * n + i];
            if aki.hi == 0.0 && aki.lo == 0.0 {
                continue;
            }
            for j in 0..p {
                c[i * p + j] = c[i * p + j] + aki * b[k * p + j];
            }
        }
    }
    c
}

/// C = A * B where A is m x n, B is n x p, result is m x p. All row-major.
fn gemm_rect(a: &[f128], b: &[f128], m: usize, n: usize, p: usize) -> Vec<f128> {
    let mut c = vec![f128::ZERO; m * p];
    for i in 0..m {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik.hi == 0.0 && aik.lo == 0.0 {
                continue;
            }
            for j in 0..p {
                c[i * p + j] = c[i * p + j] + aik * b[k * p + j];
            }
        }
    }
    c
}

/// Two-sided Jacobi eigenvalue decomposition for symmetric matrix.
/// Input: a (n x n, row-major, symmetric). Destroyed on output.
/// Returns: (eigenvalues, eigenvectors as row-major n x n where column k is eigenvector k).
fn jacobi_eigen(a: &mut [f128], n: usize) -> (Vec<f128>, Vec<f128>) {
    // Initialize V = I
    let mut v = vec![f128::ZERO; n * n];
    for i in 0..n {
        v[i * n + i] = f128::ONE;
    }

    let max_iter = 100 * n * n;
    for _iter in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].to_f64().abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Convergence check
        if max_val < 1e-60 {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        // Compute rotation angle
        let diff = aqq - app;
        let (c, s) = if apq.to_f64().abs() < 1e-70 {
            (f128::ONE, f128::ZERO)
        } else {
            let tau = diff / (apq + apq);
            // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
            let tau_abs = tau.abs();
            let t = if tau.hi >= 0.0 {
                f128::ONE / (tau_abs + (f128::ONE + tau * tau).sqrt())
            } else {
                -f128::ONE / (tau_abs + (f128::ONE + tau * tau).sqrt())
            };
            let c = f128::ONE / (f128::ONE + t * t).sqrt();
            let s = t * c;
            (c, s)
        };

        // Apply rotation to A: rotate rows/cols p and q
        // Update a[p][j] and a[q][j] for all j
        for j in 0..n {
            if j == p || j == q {
                continue;
            }
            let ajp = a[j * n + p];
            let ajq = a[j * n + q];
            let new_jp = c * ajp - s * ajq;
            let new_jq = s * ajp + c * ajq;
            a[j * n + p] = new_jp;
            a[p * n + j] = new_jp;
            a[j * n + q] = new_jq;
            a[q * n + j] = new_jq;
        }

        let new_pp = c * c * app - f128::from_f64(2.0) * c * s * apq + s * s * aqq;
        let new_qq = s * s * app + f128::from_f64(2.0) * c * s * apq + c * c * aqq;
        a[p * n + p] = new_pp;
        a[q * n + q] = new_qq;
        a[p * n + q] = f128::ZERO;
        a[q * n + p] = f128::ZERO;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f128> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// Canonical orthogonalization of overlap matrix S.
///
/// Diagonalizes S, removes eigenvalues below `threshold`, and returns
/// X = U * diag(1/sqrt(lambda)) for surviving eigenvalues.
pub fn canonical_orthogonalize(s: &[f128], n: usize, threshold: f64) -> OrthoBasis {
    assert_eq!(s.len(), n * n);

    let mut s_work = s.to_vec();
    let (evals, evecs) = jacobi_eigen(&mut s_work, n);

    // Collect indices of eigenvalues above threshold, sorted ascending
    let mut surviving: Vec<(usize, f128)> = evals
        .iter()
        .enumerate()
        .filter(|(_, e)| e.to_f64() > threshold)
        .map(|(i, e)| (i, *e))
        .collect();
    surviving.sort_by(|a, b| a.1.to_f64().partial_cmp(&b.1.to_f64()).unwrap());

    let m = surviving.len();

    // X[i, k] = sum_j evecs[i, j_k] * (1/sqrt(lambda_k))
    // where j_k is the k-th surviving eigenvector index
    let mut x = vec![f128::ZERO; n * m];
    for (k, &(idx, eval)) in surviving.iter().enumerate() {
        let inv_sqrt = eval.sqrt().recip();
        for i in 0..n {
            x[i * m + k] = evecs[i * n + idx] * inv_sqrt;
        }
    }

    OrthoBasis {
        x_matrix: x,
        n_orig: n,
        n_orth: m,
    }
}

/// Solve the generalized eigenvalue problem F C = S C eps (Roothaan-Hall).
///
/// 1. Canonical orthogonalization: X from S
/// 2. Transform: F' = X^T F X
/// 3. Diagonalize F' -> C', eps
/// 4. Back-transform: C = X C'
pub fn solve_gen_eigen(f: &[f128], s: &[f128], n: usize, threshold: f64) -> GenEigenResult {
    assert_eq!(f.len(), n * n);
    assert_eq!(s.len(), n * n);

    let basis = canonical_orthogonalize(s, n, threshold);
    let m = basis.n_orth;
    let x = &basis.x_matrix;

    // F' = X^T F X  (m x m)
    // First: tmp = F * X  (n x m)
    let tmp = gemm_rect(f, x, n, n, m);
    // F' = X^T * tmp  (m x m)
    let mut f_prime = gemm_atb(x, &tmp, n, m, m);

    // Symmetrize F' (numerical noise)
    for i in 0..m {
        for j in (i + 1)..m {
            let avg = (f_prime[i * m + j] + f_prime[j * m + i]) * f128::from_f64(0.5);
            f_prime[i * m + j] = avg;
            f_prime[j * m + i] = avg;
        }
    }

    // Diagonalize F'
    let (evals, evecs_prime) = jacobi_eigen(&mut f_prime, m);

    // Sort eigenvalues ascending
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| evals[a].to_f64().partial_cmp(&evals[b].to_f64()).unwrap());

    let sorted_evals: Vec<f128> = order.iter().map(|&i| evals[i]).collect();

    // Reorder eigenvectors (columns)
    let mut sorted_evecs = vec![f128::ZERO; m * m];
    for (new_k, &old_k) in order.iter().enumerate() {
        for i in 0..m {
            sorted_evecs[i * m + new_k] = evecs_prime[i * m + old_k];
        }
    }

    // Back-transform: C = X * C'  (n x m)
    let c = gemm_rect(x, &sorted_evecs, n, m, m);

    GenEigenResult {
        eigenvalues: sorted_evals,
        eigenvectors: c,
        n_basis: m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f64) -> f128 {
        f128::from_f64(x)
    }

    #[test]
    fn orthogonalize_identity() {
        let n = 3;
        let mut s = vec![f128::ZERO; n * n];
        for i in 0..n {
            s[i * n + i] = f128::ONE;
        }

        let basis = canonical_orthogonalize(&s, n, 1e-8);
        assert_eq!(basis.n_orth, n);

        // X^T X should be identity (since S = I, X^T S X = X^T X)
        let xtx = gemm_atb(&basis.x_matrix, &basis.x_matrix, n, n, n);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (xtx[i * n + j].to_f64() - expected).abs();
                assert!(err < 1e-28, "X^T X [{i}][{j}] = {}, expected {expected}", xtx[i * n + j].to_f64());
            }
        }
    }

    #[test]
    fn orthogonalize_2x2_overlap() {
        let n = 2;
        // S = [[1.0, 0.5], [0.5, 1.0]]
        let s = vec![f(1.0), f(0.5), f(0.5), f(1.0)];

        let basis = canonical_orthogonalize(&s, n, 1e-8);
        assert_eq!(basis.n_orth, 2);

        // Verify X^T S X = I
        let x = &basis.x_matrix;
        let tmp = gemm_rect(&s, x, n, n, basis.n_orth);
        let xtsx = gemm_atb(x, &tmp, n, basis.n_orth, basis.n_orth);

        for i in 0..basis.n_orth {
            for j in 0..basis.n_orth {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (xtsx[i * basis.n_orth + j].to_f64() - expected).abs();
                assert!(err < 1e-26, "X^T S X [{i}][{j}] = {}, expected {expected}", xtsx[i * basis.n_orth + j].to_f64());
            }
        }
    }

    #[test]
    fn orthogonalize_removes_linear_dependency() {
        let n = 2;
        // S = [[1.0, 1.0], [1.0, 1.0]] — rank 1
        let s = vec![f(1.0), f(1.0), f(1.0), f(1.0)];
        let basis = canonical_orthogonalize(&s, n, 1e-8);
        assert_eq!(basis.n_orth, 1, "Should remove one linearly dependent basis function");
    }

    #[test]
    fn solve_gen_eigen_2x2() {
        let n = 2;
        // F = [[2.0, 1.0], [1.0, 3.0]]
        // S = [[1.0, 0.2], [0.2, 1.0]]
        let fmat = vec![f(2.0), f(1.0), f(1.0), f(3.0)];
        let s = vec![f(1.0), f(0.2), f(0.2), f(1.0)];

        let result = solve_gen_eigen(&fmat, &s, n, 1e-8);
        assert_eq!(result.n_basis, 2);

        // Verify F C = S C diag(eps)
        // For each eigenvector k: F * c_k = eps_k * S * c_k
        for k in 0..result.n_basis {
            let eps = result.eigenvalues[k];
            // Extract eigenvector column k
            let c_k: Vec<f128> = (0..n).map(|i| result.eigenvectors[i * result.n_basis + k]).collect();

            // F * c_k
            let mut fc = vec![f128::ZERO; n];
            for i in 0..n {
                for j in 0..n {
                    fc[i] = fc[i] + fmat[i * n + j] * c_k[j];
                }
            }

            // eps * S * c_k
            let mut esc = vec![f128::ZERO; n];
            for i in 0..n {
                for j in 0..n {
                    esc[i] = esc[i] + eps * s[i * n + j] * c_k[j];
                }
            }

            for i in 0..n {
                let err = (fc[i] - esc[i]).to_f64().abs();
                assert!(err < 1e-25, "F*c - eps*S*c [{i}] for eigenvector {k}: err = {err:e}");
            }
        }

        // Eigenvalues should be in ascending order
        assert!(result.eigenvalues[0].to_f64() < result.eigenvalues[1].to_f64());
    }

    #[test]
    fn solve_gen_eigen_identity_overlap() {
        // With S = I, generalized eigenvalue problem reduces to standard eigenvalue problem.
        let n = 3;
        // Symmetric F
        let fmat = vec![
            f(3.0), f(1.0), f(0.0),
            f(1.0), f(2.0), f(1.0),
            f(0.0), f(1.0), f(4.0),
        ];
        let mut s = vec![f128::ZERO; n * n];
        for i in 0..n {
            s[i * n + i] = f128::ONE;
        }

        let result = solve_gen_eigen(&fmat, &s, n, 1e-8);
        assert_eq!(result.n_basis, 3);

        // Verify eigenvalues satisfy det(F - eps*I) = 0 by checking F*c = eps*c
        for k in 0..result.n_basis {
            let eps = result.eigenvalues[k];
            let c_k: Vec<f128> = (0..n).map(|i| result.eigenvectors[i * result.n_basis + k]).collect();

            let mut fc = vec![f128::ZERO; n];
            for i in 0..n {
                for j in 0..n {
                    fc[i] = fc[i] + fmat[i * n + j] * c_k[j];
                }
            }

            for i in 0..n {
                let err = (fc[i] - eps * c_k[i]).to_f64().abs();
                assert!(err < 1e-25, "F*c - eps*c [{i}] for eigenvector {k}: err = {err:e}");
            }
        }

        // Ascending order
        for k in 1..result.n_basis {
            assert!(result.eigenvalues[k].to_f64() >= result.eigenvalues[k - 1].to_f64());
        }
    }
}
