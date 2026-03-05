use rayon::prelude::*;
use crate::f128;

const PAR_THRESHOLD: usize = 64;

/// Dot product of two f128 slices, parallelized via Rayon for large inputs.
pub fn dot(a: &[f128], b: &[f128]) -> f128 {
    const CHUNK: usize = 1024;
    if a.len() < CHUNK {
        dot_serial(a, b)
    } else {
        a.par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(ac, bc)| dot_serial(ac, bc))
            .reduce(|| f128::ZERO, |acc, x| acc + x)
    }
}

#[inline]
fn dot_serial(a: &[f128], b: &[f128]) -> f128 {
    let mut sum = f128::ZERO;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// y = A * x, A is row-major [rows x cols].
pub fn gemv(a: &[f128], rows: usize, cols: usize, x: &[f128], y: &mut [f128]) {
    assert_eq!(a.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);

    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let row = &a[i * cols..(i + 1) * cols];
        *yi = dot_serial(row, x);
    });
}

/// In-place Cholesky: A = L * L^T. Row-major, n x n.
/// Returns the lower-triangular factor in the lower triangle of `a`.
pub fn cholesky(a: &mut [f128], n: usize) -> Result<(), CholeskyError> {
    assert_eq!(a.len(), n * n);

    // Temp buffer for the j-th row of L, used to parallelize the off-diagonal update.
    let mut row_j = vec![f128::ZERO; n];

    for j in 0..n {
        let mut sum = f128::ZERO;
        for k in 0..j {
            let ljk = a[j * n + k];
            sum += ljk * ljk;
        }
        let diag = a[j * n + j] - sum;
        if diag.hi <= 0.0 {
            return Err(CholeskyError { pivot: j, value: diag.to_f64() });
        }
        let ljj = diag.sqrt();
        a[j * n + j] = ljj;
        let ljj_inv = ljj.recip();

        // Snapshot the j-th row so we can read it without aliasing `a`.
        row_j[..j].copy_from_slice(&a[j * n..j * n + j]);

        let remaining = n - j - 1;
        if remaining == 0 {
            continue;
        }

        // Slice off rows (j+1)..n
        let rest = &mut a[(j + 1) * n..];

        if remaining >= PAR_THRESHOLD && j >= 4 {
            // Parallel: each row is independent since we only read row_j (snapshot)
            // and write to our own row.
            let rj = &row_j[..j];
            rest.par_chunks_mut(n).take(remaining).for_each(|row| {
                let mut s = f128::ZERO;
                for k in 0..j {
                    s += row[k] * rj[k];
                }
                row[j] = (row[j] - s) * ljj_inv;
            });
        } else {
            for idx in 0..remaining {
                let row = &mut rest[idx * n..(idx + 1) * n];
                let mut s = f128::ZERO;
                for k in 0..j {
                    s += row[k] * row_j[k];
                }
                row[j] = (row[j] - s) * ljj_inv;
            }
        }
    }
    Ok(())
}

/// Forward solve: L * y = b (L is lower-triangular, row-major n x n).
pub fn forward_solve(l: &[f128], b: &[f128], n: usize) -> Vec<f128> {
    let mut y = b.to_vec();
    for i in 0..n {
        for j in 0..i {
            y[i] = y[i] - l[i * n + j] * y[j];
        }
        y[i] = y[i] / l[i * n + i];
    }
    y
}

/// Backward solve: L^T * x = y (L is lower-triangular, row-major n x n).
pub fn backward_solve(l: &[f128], y: &[f128], n: usize) -> Vec<f128> {
    let mut x = y.to_vec();
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] = x[i] - l[j * n + i] * x[j];
        }
        x[i] = x[i] / l[i * n + i];
    }
    x
}

/// Solve L L^T x = b. L must already be factored.
pub fn solve_cholesky(l: &[f128], b: &[f128], n: usize) -> Vec<f128> {
    let y = forward_solve(l, b, n);
    backward_solve(l, &y, n)
}

/// Matrix-vector product y = A * x (row-major, n x n).
pub fn matvec(a: &[f128], x: &[f128], n: usize) -> Vec<f128> {
    let mut y = vec![f128::ZERO; n];
    if n >= PAR_THRESHOLD {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut s = f128::ZERO;
            for j in 0..n {
                s = s + a[i * n + j] * x[j];
            }
            *yi = s;
        });
    } else {
        for i in 0..n {
            for j in 0..n {
                y[i] = y[i] + a[i * n + j] * x[j];
            }
        }
    }
    y
}

/// C = A * B, row-major. A is m×k, B is k×n, C is m×n.
pub fn gemm(a: &[f128], b: &[f128], c: &mut [f128], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut s = f128::ZERO;
            for p in 0..k {
                s += a_row[p] * b[p * n + j];
            }
            c_row[j] = s;
        }
    });
}

/// C = A^T * B, row-major. A is k×m (so A^T is m×k), B is k×n, C is m×n.
pub fn gemm_atb(a: &[f128], b: &[f128], c: &mut [f128], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), k * m);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        for j in 0..n {
            let mut s = f128::ZERO;
            for p in 0..k {
                s += a[p * m + i] * b[p * n + j];
            }
            c_row[j] = s;
        }
    });
}

/// Condition number estimate from Cholesky diagonal: kappa ~ (max/min)^2.
pub fn cond_estimate(l: &[f128], n: usize) -> f64 {
    let mut dmax = 0.0_f64;
    let mut dmin = f64::MAX;
    for i in 0..n {
        let d = l[i * n + i].to_f64().abs();
        dmax = dmax.max(d);
        dmin = dmin.min(d);
    }
    (dmax / dmin).powi(2)
}

#[derive(Debug)]
pub struct CholeskyError {
    pub pivot: usize,
    pub value: f64,
}

impl std::fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cholesky failed at pivot {} (value={:.4e})", self.pivot, self.value)
    }
}

impl std::error::Error for CholeskyError {}

// --- Blocked Cholesky ---

fn extract_block(a: &[f128], lda: usize, r0: usize, c0: usize, m: usize, k: usize, dst: &mut [f128]) {
    for i in 0..m {
        dst[i * k..(i + 1) * k].copy_from_slice(&a[(r0 + i) * lda + c0..(r0 + i) * lda + c0 + k]);
    }
}

fn insert_block(a: &mut [f128], lda: usize, r0: usize, c0: usize, m: usize, k: usize, src: &[f128]) {
    for i in 0..m {
        a[(r0 + i) * lda + c0..(r0 + i) * lda + c0 + k].copy_from_slice(&src[i * k..(i + 1) * k]);
    }
}

fn trsm_right_lt(l: &[f128], jb: usize, b: &mut [f128], m: usize) {
    for j in 0..jb {
        let ljj_inv = l[j * jb + j].recip();
        for i in 0..m {
            let mut s = b[i * jb + j];
            for k in 0..j {
                s -= b[i * jb + k] * l[j * jb + k];
            }
            b[i * jb + j] = s * ljj_inv;
        }
    }
}

/// Blocked Cholesky factorization: A = L * L^T. Row-major, n x n.
pub fn cholesky_blocked(a: &mut [f128], n: usize, nb: usize) -> Result<(), CholeskyError> {
    assert_eq!(a.len(), n * n);
    if n == 0 { return Ok(()); }
    let nb = if nb == 0 { 64 } else { nb };

    let mut diag_buf = vec![f128::ZERO; nb * nb];
    let mut panel_buf = vec![f128::ZERO; n * nb];

    let mut j_block = 0;
    while j_block < n {
        let jb = std::cmp::min(nb, n - j_block);

        extract_block(a, n, j_block, j_block, jb, jb, &mut diag_buf);
        cholesky(&mut diag_buf[..jb * jb], jb).map_err(|e| CholeskyError {
            pivot: e.pivot + j_block,
            value: e.value,
        })?;
        insert_block(a, n, j_block, j_block, jb, jb, &diag_buf);

        let remaining = n - j_block - jb;
        if remaining > 0 {
            extract_block(a, n, j_block + jb, j_block, remaining, jb, &mut panel_buf);
            trsm_right_lt(&diag_buf, jb, &mut panel_buf[..remaining * jb], remaining);
            insert_block(a, n, j_block + jb, j_block, remaining, jb, &panel_buf);

            let trail_start = j_block + jb;
            let trail_size = remaining;
            let panel = &panel_buf[..remaining * jb];
            let trail = &mut a[trail_start * n + trail_start..];

            if trail_size >= PAR_THRESHOLD {
                let n_stride = n;
                let jb_local = jb;
                trail.par_chunks_mut(n_stride).enumerate().take(trail_size).for_each(|(i, row)| {
                    for col_j in 0..=i {
                        let pi = &panel[i * jb_local..(i + 1) * jb_local];
                        let pj = &panel[col_j * jb_local..(col_j + 1) * jb_local];
                        let mut s = f128::ZERO;
                        for k in 0..jb_local { s += pi[k] * pj[k]; }
                        row[col_j] -= s;
                    }
                });
            } else {
                for i in 0..trail_size {
                    for col_j in 0..=i {
                        let mut s = f128::ZERO;
                        for k in 0..jb { s += panel_buf[i * jb + k] * panel_buf[col_j * jb + k]; }
                        trail[i * n + col_j] -= s;
                    }
                }
            }
        }
        j_block += jb;
    }
    Ok(())
}

// --- Jacobi Eigensolve ---

/// Classic cyclic Jacobi eigenvalue solver for symmetric matrices.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are row-major n x n.
pub fn jacobi_eigen(a: &[f128], n: usize, max_iter: usize) -> Result<(Vec<f128>, Vec<f128>), &'static str> {
    assert_eq!(a.len(), n * n);
    let mut s = a.to_vec();
    let mut v = vec![f128::ZERO; n * n];
    for i in 0..n { v[i * n + i] = f128::ONE; }
    let threshold = f128::from_f64(1e-30);

    for _iter in 0..max_iter {
        let mut off_norm = f128::ZERO;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = s[i * n + j];
                off_norm += val * val;
            }
        }
        off_norm = off_norm.sqrt();
        if off_norm < threshold {
            let eigenvalues: Vec<f128> = (0..n).map(|i| s[i * n + i]).collect();
            let mut eigvecs = vec![f128::ZERO; n * n];
            for i in 0..n {
                for j in 0..n { eigvecs[i * n + j] = v[j * n + i]; }
            }
            return Ok((eigenvalues, eigvecs));
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = s[p * n + q];
                if apq.abs() < f128::from_f64(1e-60) { continue; }
                let app = s[p * n + p];
                let aqq = s[q * n + q];
                let tau = (aqq - app) / (f128::from_f64(2.0) * apq);
                let tau_abs = tau.abs();
                let t_denom = tau_abs + (f128::ONE + tau * tau).sqrt();
                let t = if tau.hi >= 0.0 { f128::ONE / t_denom } else { -f128::ONE / t_denom };
                let c = f128::ONE / (f128::ONE + t * t).sqrt();
                let s_rot = t * c;

                s[p * n + p] = app - t * apq;
                s[q * n + q] = aqq + t * apq;
                s[p * n + q] = f128::ZERO;
                s[q * n + p] = f128::ZERO;

                for r in 0..n {
                    if r == p || r == q { continue; }
                    let srp = s[r * n + p];
                    let srq = s[r * n + q];
                    s[r * n + p] = c * srp - s_rot * srq;
                    s[p * n + r] = c * srp - s_rot * srq;
                    s[r * n + q] = s_rot * srp + c * srq;
                    s[q * n + r] = s_rot * srp + c * srq;
                }

                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s_rot * vrq;
                    v[r * n + q] = s_rot * vrp + c * vrq;
                }
            }
        }
    }
    Err("Jacobi eigenvalue solver did not converge within max iterations")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_basic() {
        let a: Vec<f128> = (0..100).map(|i| f128::from_f64(i as f64)).collect();
        let b: Vec<f128> = (0..100).map(|_| f128::from_f64(1.0)).collect();
        let result = dot(&a, &b);
        assert!((result.to_f64() - 4950.0).abs() < 1e-10);
    }

    #[test]
    fn cholesky_3x3() {
        let mut a = vec![
            f128::from_f64(4.0), f128::from_f64(2.0), f128::from_f64(0.0),
            f128::from_f64(2.0), f128::from_f64(5.0), f128::from_f64(1.0),
            f128::from_f64(0.0), f128::from_f64(1.0), f128::from_f64(6.0),
        ];
        let n = 3;
        cholesky(&mut a, n).expect("Should be PD");

        let orig = [
            [4.0, 2.0, 0.0],
            [2.0, 5.0, 1.0],
            [0.0, 1.0, 6.0],
        ];
        for i in 0..n {
            for j in 0..n {
                let mut sum = f128::ZERO;
                for k in 0..=std::cmp::min(i, j) {
                    let lik = if i >= k { a[i * n + k] } else { f128::ZERO };
                    let ljk = if j >= k { a[j * n + k] } else { f128::ZERO };
                    sum += lik * ljk;
                }
                let err = (sum.to_f64() - orig[i][j]).abs();
                assert!(err < 1e-28, "L*L^T[{i}][{j}] error: {err:e}");
            }
        }
    }

    #[test]
    fn cholesky_parallel_200x200() {
        // Random-ish SPD: A = D + mu*I where D_ij = 1/(|i-j|+1)
        let n = 200;
        let mut a = vec![f128::ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = 1.0 / ((i as f64 - j as f64).abs() + 1.0);
                a[i * n + j] = f128::from_f64(v);
            }
            a[i * n + i] = a[i * n + i] + f128::from_f64(2.0); // ensure PD
        }
        cholesky(&mut a, n).expect("Should be PD");
    }

    // --- GEMM tests ---

    #[test]
    fn gemm_identity() {
        let n = 3;
        let mut eye = vec![f128::ZERO; n * n];
        for i in 0..n { eye[i * n + i] = f128::ONE; }
        let a: Vec<f128> = (1..=9).map(|v| f128::from_f64(v as f64)).collect();
        let mut c = vec![f128::ZERO; n * n];
        gemm(&eye, &a, &mut c, n, n, n);
        for i in 0..n * n {
            let err = (c[i] - a[i]).abs().to_f64();
            assert!(err < 1e-28, "identity*A mismatch at {i}: {err:e}");
        }
    }

    #[test]
    fn gemm_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0].map(|v| f128::from_f64(v));
        let b = [5.0, 6.0, 7.0, 8.0].map(|v| f128::from_f64(v));
        let mut c = vec![f128::ZERO; 4];
        gemm(&a, &b, &mut c, 2, 2, 2);
        let expected = [19.0, 22.0, 43.0, 50.0];
        for i in 0..4 {
            let err = (c[i].to_f64() - expected[i]).abs();
            assert!(err < 1e-28, "2x2 gemm mismatch at {i}: {err:e}");
        }
    }

    #[test]
    fn gemm_atb_matches_transpose() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|v| f128::from_f64(v));
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0].map(|v| f128::from_f64(v));
        let at = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0].map(|v| f128::from_f64(v));
        let mut c1 = vec![f128::ZERO; 8];
        let mut c2 = vec![f128::ZERO; 8];
        gemm(&at, &b, &mut c1, 2, 4, 3);
        gemm_atb(&a, &b, &mut c2, 2, 4, 3);
        for i in 0..8 {
            let err = (c1[i] - c2[i]).abs().to_f64();
            assert!(err < 1e-28, "gemm_atb mismatch at {i}: {err:e}");
        }
    }

    // --- Blocked Cholesky tests ---

    fn make_spd(n: usize) -> Vec<f128> {
        let mut a = vec![f128::ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = f128::from_f64(1.0 / ((i as f64 - j as f64).abs() + 1.0));
            }
            a[i * n + i] = a[i * n + i] + f128::from_f64(n as f64);
        }
        a
    }

    #[test]
    fn blocked_matches_unblocked() {
        let n = 20;
        let a = make_spd(n);
        let mut a_elem = a.clone();
        let mut a_block = a;
        cholesky(&mut a_elem, n).unwrap();
        cholesky_blocked(&mut a_block, n, 8).unwrap();
        for i in 0..n {
            for j in 0..=i {
                let err = (a_elem[i * n + j] - a_block[i * n + j]).abs().to_f64();
                assert!(err < 1e-28, "blocked mismatch at [{i}][{j}]: {err:e}");
            }
        }
    }

    // --- Jacobi tests ---

    #[test]
    fn jacobi_2x2() {
        let a = vec![
            f128::from_f64(2.0), f128::from_f64(1.0),
            f128::from_f64(1.0), f128::from_f64(2.0),
        ];
        let (mut evals, evecs) = jacobi_eigen(&a, 2, 100).unwrap();
        evals.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
        assert!((evals[0].to_f64() - 1.0).abs() < 1e-28);
        assert!((evals[1].to_f64() - 3.0).abs() < 1e-28);
        let v0 = &evecs[0..2];
        let v1 = &evecs[2..4];
        let d = (v0[0] * v1[0] + v0[1] * v1[1]).to_f64().abs();
        assert!(d < 1e-28, "eigenvectors not orthogonal: dot = {d:e}");
    }

    #[test]
    fn jacobi_3x3() {
        let a = vec![
            f128::from_f64(4.0), f128::from_f64(1.0), f128::from_f64(0.0),
            f128::from_f64(1.0), f128::from_f64(3.0), f128::from_f64(1.0),
            f128::from_f64(0.0), f128::from_f64(1.0), f128::from_f64(2.0),
        ];
        let a_copy = a.clone();
        let (evals, evecs) = jacobi_eigen(&a, 3, 200).unwrap();
        for i in 0..3 {
            let vi: Vec<f128> = evecs[i * 3..(i + 1) * 3].to_vec();
            let av = matvec(&a_copy, &vi, 3);
            for j in 0..3 {
                let err = (av[j] - evals[i] * vi[j]).abs().to_f64();
                assert!(err < 1e-27, "A*v[{i}][{j}] - lambda*v error: {err:e}");
            }
        }
        let trace: f64 = evals.iter().map(|e| e.to_f64()).sum();
        assert!((trace - 9.0).abs() < 1e-25);
    }

    #[test]
    fn jacobi_identity() {
        let n = 4;
        let mut a = vec![f128::ZERO; n * n];
        for i in 0..n { a[i * n + i] = f128::ONE; }
        let (evals, _) = jacobi_eigen(&a, n, 10).unwrap();
        for (i, ev) in evals.iter().enumerate() {
            assert!((ev.to_f64() - 1.0).abs() < 1e-30, "eigenvalue[{i}] = {}", ev.to_f64());
        }
    }

    #[test]
    fn solve_roundtrip() {
        let n = 4;
        let mut a = vec![f128::ZERO; n * n];
        // Hilbert-like but shifted to be well-conditioned
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = f128::from_f64(1.0 / (i + j + 1) as f64);
            }
            a[i * n + i] = a[i * n + i] + f128::from_f64(1.0);
        }
        let a_copy = a.clone();
        cholesky(&mut a, n).unwrap();
        let b = vec![f128::ONE; n];
        let x = solve_cholesky(&a, &b, n);
        let ax = matvec(&a_copy, &x, n);
        for i in 0..n {
            let err = (ax[i] - b[i]).abs().to_f64();
            assert!(err < 1e-28, "residual[{i}] = {err:e}");
        }
    }
}
