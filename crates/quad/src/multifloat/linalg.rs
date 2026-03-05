use rayon::prelude::*;
use super::core::MultiFloat;

const PAR_THRESHOLD: usize = 64;

/// Dot product of two MultiFloat<N> slices.
pub fn mf_dot<const N: usize>(a: &[MultiFloat<N>], b: &[MultiFloat<N>]) -> MultiFloat<N> {
    assert_eq!(a.len(), b.len());
    let mut sum = MultiFloat::<N>::ZERO;
    for i in 0..a.len() {
        sum = sum + a[i] * b[i];
    }
    sum
}

/// y = A * x, where A is n x m row-major.
pub fn mf_gemv<const N: usize>(
    a: &[MultiFloat<N>],
    x: &[MultiFloat<N>],
    y: &mut [MultiFloat<N>],
    n: usize,
    m: usize,
) {
    assert_eq!(a.len(), n * m);
    assert_eq!(x.len(), m);
    assert_eq!(y.len(), n);

    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let row = &a[i * m..(i + 1) * m];
        let mut s = MultiFloat::<N>::ZERO;
        for j in 0..m {
            s = s + row[j] * x[j];
        }
        *yi = s;
    });
}

/// C = A * B, row-major. A is m x k, B is k x n, C is m x n.
pub fn mf_gemm<const N: usize>(
    a: &[MultiFloat<N>],
    b: &[MultiFloat<N>],
    c: &mut [MultiFloat<N>],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut s = MultiFloat::<N>::ZERO;
            for p in 0..k {
                s = s + a_row[p] * b[p * n + j];
            }
            c_row[j] = s;
        }
    });
}

/// C = A^T * B, row-major. A is k x m (so A^T is m x k), B is k x n, C is m x n.
pub fn mf_gemm_atb<const N: usize>(
    a: &[MultiFloat<N>],
    b: &[MultiFloat<N>],
    c: &mut [MultiFloat<N>],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), k * m);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        for j in 0..n {
            let mut s = MultiFloat::<N>::ZERO;
            for p in 0..k {
                s = s + a[p * m + i] * b[p * n + j];
            }
            c_row[j] = s;
        }
    });
}

/// In-place lower-triangular Cholesky factorization: A = L * L^T.
/// Row-major, n x n. Returns error if matrix is not positive definite.
pub fn mf_cholesky<const N: usize>(
    a: &mut [MultiFloat<N>],
    n: usize,
) -> Result<(), &'static str> {
    assert_eq!(a.len(), n * n);

    let mut row_j = vec![MultiFloat::<N>::ZERO; n];

    for j in 0..n {
        let mut sum = MultiFloat::<N>::ZERO;
        for k in 0..j {
            let ljk = a[j * n + k];
            sum = sum + ljk * ljk;
        }
        let diag = a[j * n + j] - sum;
        if diag.limbs[0] <= 0.0 {
            return Err("Cholesky failed: matrix is not positive definite");
        }
        let ljj = diag.sqrt();
        a[j * n + j] = ljj;
        let ljj_inv = ljj.recip();

        row_j[..j].copy_from_slice(&a[j * n..j * n + j]);

        let remaining = n - j - 1;
        if remaining == 0 {
            continue;
        }

        let rest = &mut a[(j + 1) * n..];

        if remaining >= PAR_THRESHOLD && j >= 4 {
            let rj = &row_j[..j];
            rest.par_chunks_mut(n).take(remaining).for_each(|row| {
                let mut s = MultiFloat::<N>::ZERO;
                for k in 0..j {
                    s = s + row[k] * rj[k];
                }
                row[j] = (row[j] - s) * ljj_inv;
            });
        } else {
            for idx in 0..remaining {
                let row = &mut rest[idx * n..(idx + 1) * n];
                let mut s = MultiFloat::<N>::ZERO;
                for k in 0..j {
                    s = s + row[k] * row_j[k];
                }
                row[j] = (row[j] - s) * ljj_inv;
            }
        }
    }
    Ok(())
}

/// Solve L L^T x = b via forward then back substitution.
/// L must already be the Cholesky factor (lower-triangular, row-major n x n).
pub fn mf_cholesky_solve<const N: usize>(
    l: &[MultiFloat<N>],
    b: &[MultiFloat<N>],
    x: &mut [MultiFloat<N>],
    n: usize,
) {
    assert_eq!(l.len(), n * n);
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);

    // Forward solve: L * y = b
    let mut y = b.to_vec();
    for i in 0..n {
        for j in 0..i {
            y[i] = y[i] - l[i * n + j] * y[j];
        }
        y[i] = y[i] / l[i * n + i];
    }

    // Backward solve: L^T * x = y
    x.copy_from_slice(&y);
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] = x[i] - l[j * n + i] * x[j];
        }
        x[i] = x[i] / l[i * n + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type MF4 = MultiFloat<4>;

    fn mf(v: f64) -> MF4 {
        MF4::from_f64(v)
    }

    #[test]
    fn dot_basic() {
        let a: Vec<MF4> = (0..100).map(|i| mf(i as f64)).collect();
        let b: Vec<MF4> = (0..100).map(|_| mf(1.0)).collect();
        let result = mf_dot(&a, &b);
        assert!((result.to_f64() - 4950.0).abs() < 1e-10);
    }

    #[test]
    fn gemv_identity() {
        let n = 3;
        let mut eye = vec![MF4::ZERO; n * n];
        for i in 0..n {
            eye[i * n + i] = MF4::ONE;
        }
        let x: Vec<MF4> = vec![mf(1.0), mf(2.0), mf(3.0)];
        let mut y = vec![MF4::ZERO; n];
        mf_gemv(&eye, &x, &mut y, n, n);
        for i in 0..n {
            let err = (y[i].to_f64() - x[i].to_f64()).abs();
            assert!(err < 1e-28, "gemv identity mismatch at {i}: {err:e}");
        }
    }

    #[test]
    fn gemm_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0].map(|v| mf(v));
        let b = [5.0, 6.0, 7.0, 8.0].map(|v| mf(v));
        let mut c = vec![MF4::ZERO; 4];
        mf_gemm(&a, &b, &mut c, 2, 2, 2);
        let expected = [19.0, 22.0, 43.0, 50.0];
        for i in 0..4 {
            let err = (c[i].to_f64() - expected[i]).abs();
            assert!(err < 1e-28, "2x2 gemm mismatch at {i}: {err:e}");
        }
    }

    #[test]
    fn gemm_atb_matches_transpose() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|v| mf(v));
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0].map(|v| mf(v));
        let at = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0].map(|v| mf(v));
        let mut c1 = vec![MF4::ZERO; 8];
        let mut c2 = vec![MF4::ZERO; 8];
        mf_gemm(&at, &b, &mut c1, 2, 4, 3);
        mf_gemm_atb(&a, &b, &mut c2, 2, 4, 3);
        for i in 0..8 {
            let err = (c1[i].to_f64() - c2[i].to_f64()).abs();
            assert!(err < 1e-28, "gemm_atb mismatch at {i}: {err:e}");
        }
    }

    #[test]
    fn cholesky_3x3() {
        let mut a = vec![
            mf(4.0), mf(2.0), mf(0.0),
            mf(2.0), mf(5.0), mf(1.0),
            mf(0.0), mf(1.0), mf(6.0),
        ];
        let n = 3;
        mf_cholesky(&mut a, n).expect("Should be PD");

        let orig = [
            [4.0, 2.0, 0.0],
            [2.0, 5.0, 1.0],
            [0.0, 1.0, 6.0],
        ];
        for i in 0..n {
            for j in 0..n {
                let mut sum = MF4::ZERO;
                for k in 0..=std::cmp::min(i, j) {
                    let lik = if i >= k { a[i * n + k] } else { MF4::ZERO };
                    let ljk = if j >= k { a[j * n + k] } else { MF4::ZERO };
                    sum = sum + lik * ljk;
                }
                let err = (sum.to_f64() - orig[i][j]).abs();
                assert!(err < 1e-28, "L*L^T[{i}][{j}] error: {err:e}");
            }
        }
    }

    #[test]
    fn solve_roundtrip() {
        let n = 4;
        let mut a = vec![MF4::ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = mf(1.0 / (i + j + 1) as f64);
            }
            a[i * n + i] = a[i * n + i] + mf(1.0);
        }
        let a_copy = a.clone();
        mf_cholesky(&mut a, n).unwrap();
        let b = vec![MF4::ONE; n];
        let mut x = vec![MF4::ZERO; n];
        mf_cholesky_solve(&a, &b, &mut x, n);

        // Verify A * x == b
        let mut ax = vec![MF4::ZERO; n];
        mf_gemv(&a_copy, &x, &mut ax, n, n);
        for i in 0..n {
            let err = (ax[i].to_f64() - b[i].to_f64()).abs();
            assert!(err < 1e-28, "residual[{i}] = {err:e}");
        }
    }
}
