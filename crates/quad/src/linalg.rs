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
