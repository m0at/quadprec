use std::ops::{Div, DivAssign, Rem};
use super::core::MultiFloat;

// These are available because MultiFloat<N> implements Add, Sub, Mul in sibling modules

impl<const N: usize> MultiFloat<N> {
    /// Compute the reciprocal 1/self via Newton iteration.
    ///
    /// Starting from an f64 seed (~53 bits), each Newton step roughly doubles
    /// the number of correct bits:
    ///   x_{k+1} = x_k + x_k * (1 - self * x_k)
    ///
    /// Steps needed: ceil(log2(N)) to cover N*53 bits of precision.
    pub fn recip(self) -> Self {
        if self.limbs[0] == 0.0 {
            let mut limbs = [0.0f64; N];
            limbs[0] = if self.limbs[0].is_sign_positive() {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            return Self { limbs };
        }
        if !self.limbs[0].is_finite() {
            return Self::from_f64(0.0);
        }

        let mut x = Self::from_f64(1.0 / self.limbs[0]);
        let one = Self::one();
        // ceil(log2(N)): number of Newton steps
        // ceil(log2(N)) + 1: the extra step compensates for rounding in each iteration
        // Each Newton step roughly doubles correct bits. Starting from ~53 bits,
        // we need ceil(log2(N)) steps to reach N*53 bits, plus 2 extra for rounding.
        let steps = if N <= 1 { 1 } else { (N as f64).log2().ceil() as usize + 2 };
        for _ in 0..steps {
            // x = x + x * (1 - self * x)
            let residual = one - self * x;
            x = x + x * residual;
        }
        x
    }
}

impl<const N: usize> Div for MultiFloat<N> {
    type Output = Self;

    /// Division via Newton reciprocal: a / b = a * (1/b).
    ///
    /// The reciprocal is computed with enough Newton iterations to achieve
    /// full N*53-bit precision.
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Handle special cases
        if rhs.limbs[0] == 0.0 {
            let sign = if self.limbs[0].is_sign_positive() == rhs.limbs[0].is_sign_positive() {
                1.0
            } else {
                -1.0
            };
            let mut limbs = [0.0f64; N];
            if self.limbs[0] == 0.0 {
                limbs[0] = f64::NAN;
            } else {
                limbs[0] = sign * f64::INFINITY;
            }
            return Self { limbs };
        }
        if self.limbs[0] == 0.0 {
            return Self::ZERO;
        }
        if !self.limbs[0].is_finite() || !rhs.limbs[0].is_finite() {
            return Self::from_f64(self.limbs[0] / rhs.limbs[0]);
        }

        self * rhs.recip()
    }
}

impl<const N: usize> DivAssign for MultiFloat<N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const N: usize> Rem for MultiFloat<N> {
    type Output = Self;

    /// Remainder: self - trunc(self / rhs) * rhs
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        self - q * rhs
    }
}

// --- Mixed MultiFloat<N> / f64 ops ---

impl<const N: usize> Div<f64> for MultiFloat<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        self / Self::from_f64(rhs)
    }
}

impl<const N: usize> DivAssign<f64> for MultiFloat<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / Self::from_f64(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type F256 = MultiFloat<4>;
    type F512 = MultiFloat<8>;

    #[test]
    fn recip_one() {
        let one = F256::ONE;
        let r = one.recip();
        let err = (r - one).limbs[0].abs();
        assert!(err < 1e-60, "recip(1) error: {err:e}");
    }

    #[test]
    fn div_one_third() {
        let one = F256::ONE;
        let three = F256::from_f64(3.0);
        let third = one / three;
        let back = third * three;
        let err = (back - one).limbs[0].abs();
        assert!(err < 1e-60, "1/3 * 3 error: {err:e}");
    }

    #[test]
    fn div_by_self_is_one() {
        let x = F256::from_f64(std::f64::consts::PI);
        let result = x / x;
        let err = (result - F256::ONE).limbs[0].abs();
        assert!(err < 1e-60, "pi/pi error: {err:e}");
    }

    #[test]
    fn div_f512_precision() {
        let one = F512::ONE;
        let seven = F512::from_f64(7.0);
        let seventh = one / seven;
        let back = seventh * seven;
        let err = (back - one).limbs[0].abs();
        assert!(err < 1e-100, "1/7 * 7 (f512) error: {err:e}");
    }

    #[test]
    fn rem_basic() {
        let ten = F256::from_f64(10.0);
        let three = F256::from_f64(3.0);
        let r = ten % three;
        let err = (r - F256::from_f64(1.0)).limbs[0].abs();
        assert!(err < 1e-60, "10 % 3 error: {err:e}");
    }

    #[test]
    fn div_by_f64() {
        let six = F256::from_f64(6.0);
        let result = six / 3.0;
        let err = (result - F256::from_f64(2.0)).limbs[0].abs();
        assert!(err < 1e-60, "6 / 3.0 error: {err:e}");
    }

    #[test]
    fn recip_zero() {
        let zero = F256::ZERO;
        let r = zero.recip();
        assert!(r.limbs[0].is_infinite(), "recip(0) should be infinite");
    }

    #[test]
    fn div_zero_by_zero() {
        let zero = F256::ZERO;
        let r = zero / zero;
        assert!(r.limbs[0].is_nan(), "0/0 should be NaN");
    }
}
