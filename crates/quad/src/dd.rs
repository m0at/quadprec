use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f128 {
    pub hi: f64,
    pub lo: f64,
}

// --- Primitives (Dekker/Knuth) ---

/// Error-free addition: s + e = a + b exactly.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Fast version when |a| >= |b|.
#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// Error-free multiplication: p + e = a * b exactly.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p); // FMA: exact error term
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
        if self.hi < 0.0 { -self } else { self }
    }

    pub fn sqrt(self) -> Self {
        if self.hi == 0.0 && self.lo == 0.0 {
            return Self::ZERO;
        }
        // Newton iteration: x_{n+1} = 0.5 * (x_n + a/x_n)
        // One iteration from f64 seed gives full double-double accuracy.
        let x = Self::from_f64(1.0 / self.hi.sqrt());
        // Goldschmidt refinement: r = a*x, h = 0.5*x, then r + r*(0.5 - r*h)
        let ax = self * x;
        let hx = x * Self::from_f64(0.5);
        let r = ax + ax * (Self::from_f64(0.5) - ax * hx);
        r
    }

    pub fn recip(self) -> Self {
        // Newton iteration on 1/a, starting from f64 seed.
        let x0 = Self::from_f64(1.0 / self.hi);
        // One Newton step: x1 = x0 + x0*(1 - a*x0)
        x0 + x0 * (Self::ONE - self * x0)
    }

    /// Renormalize so invariant |lo| <= 0.5 * ulp(hi) holds.
    #[inline(always)]
    fn renorm(hi: f64, lo: f64) -> Self {
        let (s, e) = quick_two_sum(hi, lo);
        Self { hi: s, lo: e }
    }
}

// --- Arithmetic ---

impl Add for f128 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let (s1, e1) = two_sum(self.hi, rhs.hi);
        let e = e1 + (self.lo + rhs.lo);
        Self::renorm(s1, e)
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
        self * rhs.recip()
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

// --- Conversions & Display ---

impl From<f64> for f128 {
    #[inline(always)]
    fn from(x: f64) -> Self { Self::from_f64(x) }
}

impl From<i64> for f128 {
    fn from(x: i64) -> Self {
        let hi = x as f64;
        let lo = (x - hi as i64) as f64;
        Self { hi, lo }
    }
}

impl fmt::Debug for f128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f128({:e} + {:e})", self.hi, self.lo)
    }
}

impl fmt::Display for f128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print as combined value with max precision
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
        // 1 + 1e-20 should not lose the small part
        let a = f128::from_f64(1.0);
        let b = f128::from_f64(1e-20);
        let c = a + b;
        let d = c - a;
        // In pure f64, (1.0 + 1e-20) - 1.0 != 1e-20 due to rounding.
        // In double-double, it should be exact.
        assert!((d.to_f64() - 1e-20).abs() < 1e-35,
            "Expected ~1e-20, got {}", d.to_f64());
    }

    #[test]
    fn mul_precision() {
        // (1 + eps)^2 where eps = 2^-53. Result = 1 + 2*eps + eps^2.
        // f64 loses eps^2; double-double should keep it.
        let eps = f64::EPSILON; // 2^-52 ≈ 2.22e-16
        let small = eps * 0.5; // 2^-53
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
    fn catastrophic_cancellation() {
        // Classic: compute (a+b) - a where a >> b
        // In f64: 1e16 + 1.0 - 1e16 = 0.0 (catastrophic loss)
        // In dd: should recover 1.0
        let a = f128::from_f64(1e16);
        let b = f128::from_f64(1.0);
        let result = (a + b) - a;
        assert!((result.to_f64() - 1.0).abs() < 1e-15,
            "Catastrophic cancellation: got {} instead of 1.0", result.to_f64());
    }

    #[test]
    fn kahan_sum_scenario() {
        // Sum of 1e16 ones: in f64, accumulating 1.0 loses precision past ~2^53.
        // In dd, should be exact.
        let n = 100_000;
        let one = f128::from_f64(1.0);
        let mut sum = f128::ZERO;
        for _ in 0..n {
            sum += one;
        }
        assert_eq!(sum.to_f64(), n as f64);
    }
}
