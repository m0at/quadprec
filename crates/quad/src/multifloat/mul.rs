use std::ops::{Mul, MulAssign};
use super::core::{MultiFloat, two_sum, two_prod, renormalize_from};

/// Multiply two MultiFloat<N> expansions.
///
/// Computes cross-products a[i]*b[j] for all pairs where i+j < N.
/// All products use error-free two_prod; error terms are accumulated into
/// the 2N work array as long as they fit.
///
/// All terms are accumulated into a 2N-sized working array via error-free two_sum,
/// then renormalized down to N limbs.
impl<const N: usize> Mul for MultiFloat<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Working buffer: indices 0..2N hold the convolution terms.
        // We can't use {2*N} as a const generic on stable, so we use a
        // heap-free approach with a fixed large buffer (max 16 for N=8).
        // For correctness at any N, we accumulate into a [f64; 16] and
        // then renormalize the first 2*N entries.
        debug_assert!(N <= 8, "MultiFloat<N> mul supports N <= 8");

        // We'll accumulate into `work[0..2*N]`. Indices beyond that stay zero.
        let mut work = [0.0f64; 16];
        let a = &self.limbs;
        let b = &rhs.limbs;

        // Inline helper: error-free accumulate val into work[idx],
        // cascading error into work[idx+1..].
        #[inline(always)]
        fn accumulate(work: &mut [f64; 16], idx: usize, val: f64, limit: usize) {
            let (s, mut e) = two_sum(work[idx], val);
            work[idx] = s;
            let mut k = idx + 1;
            while e != 0.0 && k < limit {
                let (s2, e2) = two_sum(work[k], e);
                work[k] = s2;
                e = e2;
                k += 1;
            }
        }

        let limit = 2 * N;
        for i in 0..N {
            for j in 0..N {
                if i + j >= N {
                    break;
                }

                let (p, e) = two_prod(a[i], b[j]);
                accumulate(&mut work, i + j, p, limit);
                if i + j + 1 < limit {
                    accumulate(&mut work, i + j + 1, e, limit);
                }
            }
        }

        // Renormalize the 2*N working terms down to N output limbs.
        // We dispatch on N to call renormalize_from with the right const generics.
        let limbs = match N {
            1 => {
                let buf: [f64; 2] = [work[0], work[1]];
                renormalize_from::<N, 2>(&buf)
            }
            2 => {
                let buf: [f64; 4] = [work[0], work[1], work[2], work[3]];
                renormalize_from::<N, 4>(&buf)
            }
            3 => {
                let buf: [f64; 6] = [work[0], work[1], work[2], work[3], work[4], work[5]];
                renormalize_from::<N, 6>(&buf)
            }
            4 => {
                let buf: [f64; 8] = [
                    work[0], work[1], work[2], work[3],
                    work[4], work[5], work[6], work[7],
                ];
                renormalize_from::<N, 8>(&buf)
            }
            5 => {
                let buf: [f64; 10] = [
                    work[0], work[1], work[2], work[3], work[4],
                    work[5], work[6], work[7], work[8], work[9],
                ];
                renormalize_from::<N, 10>(&buf)
            }
            6 => {
                let buf: [f64; 12] = [
                    work[0], work[1], work[2], work[3], work[4], work[5],
                    work[6], work[7], work[8], work[9], work[10], work[11],
                ];
                renormalize_from::<N, 12>(&buf)
            }
            7 => {
                let buf: [f64; 14] = [
                    work[0], work[1], work[2], work[3], work[4], work[5], work[6],
                    work[7], work[8], work[9], work[10], work[11], work[12], work[13],
                ];
                renormalize_from::<N, 14>(&buf)
            }
            8 => {
                let buf: [f64; 16] = work;
                renormalize_from::<N, 16>(&buf)
            }
            _ => unreachable!("MultiFloat<N> mul only supports N <= 8"),
        };

        Self { limbs }
    }
}

impl<const N: usize> MulAssign for MultiFloat<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Multiply MultiFloat<N> by a scalar f64.
///
/// Only the first cross-product (limbs[0] * rhs) uses two_prod;
/// lower limbs use plain multiply.
impl<const N: usize> Mul<f64> for MultiFloat<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        debug_assert!(N <= 8, "MultiFloat<N> mul supports N <= 8");

        let mut work = [0.0f64; 16];

        // All limbs: use two_prod for full error capture
        for i in 0..N {
            let (p, e) = two_prod(self.limbs[i], rhs);
            let (s, err) = two_sum(work[i], p);
            work[i] = s;
            if i + 1 < 2 * N {
                let (s2, err2) = two_sum(work[i + 1], e);
                work[i + 1] = s2;
                if i + 2 < 2 * N {
                    work[i + 2] += err2;
                }
                let (s3, err3) = two_sum(work[i + 1], err);
                work[i + 1] = s3;
                if i + 2 < 2 * N {
                    work[i + 2] += err3;
                }
            }
        }

        let limbs = match N {
            1 => {
                let buf: [f64; 2] = [work[0], work[1]];
                renormalize_from::<N, 2>(&buf)
            }
            2 => {
                let buf: [f64; 4] = [work[0], work[1], work[2], work[3]];
                renormalize_from::<N, 4>(&buf)
            }
            3 => {
                let buf: [f64; 6] = [work[0], work[1], work[2], work[3], work[4], work[5]];
                renormalize_from::<N, 6>(&buf)
            }
            4 => {
                let buf: [f64; 8] = [
                    work[0], work[1], work[2], work[3],
                    work[4], work[5], work[6], work[7],
                ];
                renormalize_from::<N, 8>(&buf)
            }
            5 => {
                let buf: [f64; 10] = [
                    work[0], work[1], work[2], work[3], work[4],
                    work[5], work[6], work[7], work[8], work[9],
                ];
                renormalize_from::<N, 10>(&buf)
            }
            6 => {
                let buf: [f64; 12] = [
                    work[0], work[1], work[2], work[3], work[4], work[5],
                    work[6], work[7], work[8], work[9], work[10], work[11],
                ];
                renormalize_from::<N, 12>(&buf)
            }
            7 => {
                let buf: [f64; 14] = [
                    work[0], work[1], work[2], work[3], work[4], work[5], work[6],
                    work[7], work[8], work[9], work[10], work[11], work[12], work[13],
                ];
                renormalize_from::<N, 14>(&buf)
            }
            8 => {
                let buf: [f64; 16] = work;
                renormalize_from::<N, 16>(&buf)
            }
            _ => unreachable!("MultiFloat<N> mul only supports N <= 8"),
        };

        Self { limbs }
    }
}

impl<const N: usize> Mul<MultiFloat<N>> for f64 {
    type Output = MultiFloat<N>;

    #[inline]
    fn mul(self, rhs: MultiFloat<N>) -> MultiFloat<N> {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::super::core::MultiFloat;

    type F256 = MultiFloat<4>;
    type F512 = MultiFloat<8>;

    #[test]
    fn mul_one_identity() {
        let a = F256::from_f64(std::f64::consts::PI);
        let result = a * F256::ONE;
        assert_eq!(result.to_f64(), std::f64::consts::PI);
    }

    #[test]
    fn mul_zero() {
        let a = F256::from_f64(42.0);
        let result = a * F256::ZERO;
        assert_eq!(result.to_f64(), 0.0);
    }

    #[test]
    fn mul_simple_integers() {
        let a = F256::from_f64(3.0);
        let b = F256::from_f64(7.0);
        let c = a * b;
        assert_eq!(c.to_f64(), 21.0);
    }

    #[test]
    fn mul_precision_dd_style() {
        // Similar to the f128 dd.rs test: (1 + eps/2)^2 should preserve the cross-term
        let eps = f64::EPSILON;
        let small = eps * 0.5;
        let a = F256::from_limbs([1.0, small, 0.0, 0.0]);
        let c = a * a;
        let expected = 2.0 * small + small * small;
        let actual = c.limbs[0] - 1.0;
        // The result is 1 + 2*small + small^2; the cross-term 2*small should be captured
        let sum: f64 = c.limbs.iter().sum::<f64>() - 1.0;
        assert!(
            (sum - expected).abs() < 1e-31,
            "Expected ~{expected:e}, got {sum:e}"
        );
    }

    #[test]
    fn mul_scalar_f64() {
        let a = F256::from_f64(std::f64::consts::PI);
        let c = a * 2.0;
        assert!((c.to_f64() - 2.0 * std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn mul_f64_commutative() {
        let a = F256::from_f64(2.5);
        let c1 = a * 3.0;
        let c2 = 3.0 * a;
        assert_eq!(c1.to_f64(), c2.to_f64());
    }

    #[test]
    fn mul_assign_works() {
        let mut a = F256::from_f64(5.0);
        a *= F256::from_f64(4.0);
        assert_eq!(a.to_f64(), 20.0);
    }

    #[test]
    fn mul_negative() {
        let a = F256::from_f64(-3.0);
        let b = F256::from_f64(7.0);
        let c = a * b;
        assert_eq!(c.to_f64(), -21.0);
    }

    #[test]
    fn mul_f512_basic() {
        let a = F512::from_f64(6.0);
        let b = F512::from_f64(7.0);
        let c = a * b;
        assert_eq!(c.to_f64(), 42.0);
    }

    #[test]
    fn mul_third_roundtrip() {
        // 1/3 * 3 should be very close to 1 at high precision
        // We approximate 1/3 by repeated refinement
        let three = F256::from_f64(3.0);
        let third = F256::from_limbs([
            3.333333333333333e-1,
            1.8503717077085942e-17,
            -5.164681284890277e-34,
            2.8687459674023424e-50,
        ]);
        let result = third * three;
        let err = (result.limbs.iter().sum::<f64>() - 1.0).abs();
        assert!(err < 1e-60, "1/3 * 3 error: {err:e}");
    }
}
