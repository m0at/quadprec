use std::fmt;
use std::iter::{Sum, Product};
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f128 {
    pub hi: f64,
    pub lo: f64,
}

// --- Primitives (Dekker/Knuth) ---

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

impl f128 {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    #[inline(always)]
    pub fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    #[inline(always)]
    pub fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    pub fn abs(self) -> Self {
        if self.hi.is_sign_negative() { -self } else { self }
    }

    pub fn sqrt(self) -> Self {
        if self.hi < 0.0 {
            return Self::new(f64::NAN, 0.0);
        }
        if self.hi == 0.0 {
            return Self::ZERO;
        }
        if !self.hi.is_finite() {
            return Self::new(self.hi, 0.0);
        }
        // Goldschmidt from f64 seed, then Newton correction for full precision.
        let x = Self::from_f64(1.0 / self.hi.sqrt());
        let ax = self * x;
        let hx = x * Self::from_f64(0.5);
        let r = ax + ax * (Self::from_f64(0.5) - ax * hx);
        // Second Newton step: r2 = 0.5 * (r + self/r) locks in all ~106 bits.
        (r + self / r) * Self::from_f64(0.5)
    }

    pub fn recip(self) -> Self {
        let x0 = Self::from_f64(1.0 / self.hi);
        x0 + x0 * (Self::ONE - self * x0)
    }

    #[inline(always)]
    fn renorm(hi: f64, lo: f64) -> Self {
        let (s, e) = quick_two_sum(hi, lo);
        Self { hi: s, lo: e }
    }

    pub fn is_nan(self) -> bool { self.hi.is_nan() }
    pub fn is_finite(self) -> bool { self.hi.is_finite() }
    pub fn is_zero(self) -> bool { self.hi == 0.0 }
}

// --- Arithmetic ---

impl Add for f128 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let (s1, e1) = two_sum(self.hi, rhs.hi);
        let (s2, e2) = two_sum(self.lo, rhs.lo);
        let e = e1 + s2;
        let (s, e) = quick_two_sum(s1, e);
        Self::renorm(s, e + e2)
    }
}

impl Sub for f128 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Neg for f128 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self { hi: -self.hi, lo: -self.lo }
    }
}

impl Mul for f128 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (p1, e1) = two_prod(self.hi, rhs.hi);
        let e = e1 + (self.hi * rhs.lo + self.lo * rhs.hi);
        Self::renorm(p1, e)
    }
}

impl Div for f128 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Direct division: more accurate than self * rhs.recip()
        let q1 = self.hi / rhs.hi;
        let r = self - Self::from_f64(q1) * rhs;
        let q2 = r.hi / rhs.hi;
        Self::renorm(q1, q2)
    }
}

impl AddAssign for f128 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl SubAssign for f128 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl MulAssign for f128 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl DivAssign for f128 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

// --- Mixed f128/f64 ops (avoids unnecessary promotion) ---

impl Add<f64> for f128 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f64) -> Self {
        let (s1, e1) = two_sum(self.hi, rhs);
        Self::renorm(s1, e1 + self.lo)
    }
}

impl Sub<f64> for f128 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f64) -> Self { self + (-rhs) }
}

impl Mul<f64> for f128 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        let (p1, e1) = two_prod(self.hi, rhs);
        Self::renorm(p1, e1 + self.lo * rhs)
    }
}

impl Div<f64> for f128 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        let q1 = self.hi / rhs;
        let r = self - Self::from_f64(q1) * Self::from_f64(rhs);
        Self::renorm(q1, r.hi / rhs)
    }
}

// --- Conversions ---

impl From<f64> for f128 {
    #[inline(always)]
    fn from(x: f64) -> Self { Self::from_f64(x) }
}

impl From<f32> for f128 {
    #[inline(always)]
    fn from(x: f32) -> Self { Self::from_f64(x as f64) }
}

impl From<i32> for f128 {
    #[inline(always)]
    fn from(x: i32) -> Self { Self::from_f64(x as f64) }
}

impl From<u32> for f128 {
    #[inline(always)]
    fn from(x: u32) -> Self { Self::from_f64(x as f64) }
}

impl From<i64> for f128 {
    fn from(x: i64) -> Self {
        // Split into two 32-bit halves to avoid overflow on large values.
        let hi_part = (x >> 32) as f64 * (1u64 << 32) as f64;
        let lo_part = (x as u64 & 0xFFFFFFFF) as f64;
        let (s, e) = two_sum(hi_part, lo_part);
        Self { hi: s, lo: e }
    }
}

impl From<u64> for f128 {
    fn from(x: u64) -> Self {
        let hi_part = (x >> 32) as f64 * (1u64 << 32) as f64;
        let lo_part = (x & 0xFFFFFFFF) as f64;
        let (s, e) = two_sum(hi_part, lo_part);
        Self { hi: s, lo: e }
    }
}

// --- Iterator traits ---

impl Sum for f128 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a> Sum<&'a f128> for f128 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + *b)
    }
}

impl Product for f128 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<'a> Product<&'a f128> for f128 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * *b)
    }
}

// --- Display ---

impl fmt::Debug for f128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f128({:e} + {:e})", self.hi, self.lo)
    }
}

impl fmt::Display for f128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.32e}", self.hi + self.lo)
    }
}

impl PartialEq for f128 {
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd for f128 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.hi.partial_cmp(&other.hi) {
            Some(std::cmp::Ordering::Equal) => self.lo.partial_cmp(&other.lo),
            ord => ord,
        }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_exact() {
        let a = f128::from_f64(1.0);
        let b = f128::from_f64(1e-20);
        let c = a + b;
        let d = c - a;
        assert!((d.to_f64() - 1e-20).abs() < 1e-35,
            "Expected ~1e-20, got {}", d.to_f64());
    }

    #[test]
    fn mul_precision() {
        let eps = f64::EPSILON;
        let small = eps * 0.5;
        let a = f128::new(1.0, small);
        let c = a * a;
        let expected = 2.0 * small + small * small;
        let actual = (c - f128::ONE).to_f64();
        assert!((actual - expected).abs() < 1e-31,
            "Expected ~{expected:e}, got {actual:e}");
    }

    #[test]
    fn sqrt_precision() {
        let two = f128::from_f64(2.0);
        let s = two.sqrt();
        let check = s * s;
        let err = (check - two).abs();
        assert!(err.to_f64() < 1e-30, "sqrt(2)^2 error: {:e}", err.to_f64());
    }

    #[test]
    fn sqrt_edge_cases() {
        assert!(f128::from_f64(-1.0).sqrt().is_nan());
        assert_eq!(f128::ZERO.sqrt().to_f64(), 0.0);
        assert!(f128::from_f64(f64::INFINITY).sqrt().to_f64().is_infinite());
        assert!(f128::from_f64(f64::NAN).sqrt().is_nan());
    }

    #[test]
    fn div_precision() {
        // 1/3 * 3 should be very close to 1
        let third = f128::ONE / f128::from_f64(3.0);
        let result = third * f128::from_f64(3.0);
        let err = (result - f128::ONE).abs().to_f64();
        assert!(err < 1e-31, "1/3 * 3 error: {err:e}");
    }

    #[test]
    fn catastrophic_cancellation() {
        let a = f128::from_f64(1e16);
        let b = f128::from_f64(1.0);
        let result = (a + b) - a;
        assert!((result.to_f64() - 1.0).abs() < 1e-15,
            "Catastrophic cancellation: got {} instead of 1.0", result.to_f64());
    }

    #[test]
    fn kahan_sum_scenario() {
        let n = 100_000;
        let one = f128::from_f64(1.0);
        let mut sum = f128::ZERO;
        for _ in 0..n {
            sum += one;
        }
        assert_eq!(sum.to_f64(), n as f64);
    }

    #[test]
    fn from_i64_large() {
        // Values > 2^53 should not lose precision
        let x: i64 = (1i64 << 53) + 1;
        let f = f128::from(x);
        let back = f.hi as i64 + f.lo as i64;
        assert_eq!(back, x, "i64 roundtrip failed for 2^53+1");
    }

    #[test]
    fn mixed_f64_ops() {
        let a = f128::from_f64(1.0);
        let b = a + 1e-20;
        let c = b - 1e-20;
        assert!((c.to_f64() - 1.0).abs() < 1e-31);

        let d = f128::from_f64(3.0) * 2.0;
        assert_eq!(d.to_f64(), 6.0);
    }

    #[test]
    fn abs_negative_zero() {
        let nz = f128::new(-0.0, 0.0);
        let result = nz.abs();
        assert!(!result.hi.is_sign_negative(), "abs(-0) should be +0");
    }

    #[test]
    fn sum_trait() {
        let vals: Vec<f128> = (1..=100).map(|i| f128::from_f64(i as f64)).collect();
        let total: f128 = vals.iter().sum();
        assert_eq!(total.to_f64(), 5050.0);
    }
}
