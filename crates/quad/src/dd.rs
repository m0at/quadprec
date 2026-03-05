use std::fmt;
use std::iter::{Sum, Product};
use std::ops::{Add, Sub, Mul, Div, Neg, Rem, AddAssign, SubAssign, MulAssign, DivAssign};
use std::str::FromStr;

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
    pub const LN2: Self = Self { hi: 0.6931471805599453_f64, lo: 2.3190468138462996e-17_f64 };

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
    pub(crate) fn renorm(hi: f64, lo: f64) -> Self {
        let (s, e) = quick_two_sum(hi, lo);
        Self { hi: s, lo: e }
    }

    pub fn is_nan(self) -> bool { self.hi.is_nan() }
    pub fn is_finite(self) -> bool { self.hi.is_finite() }
    pub fn is_zero(self) -> bool { self.hi == 0.0 }

    /// Truncate toward zero.
    pub fn trunc(self) -> Self {
        let t = self.hi.trunc();
        if t == self.hi {
            Self::new(t, self.lo.trunc())
        } else {
            Self::new(t, 0.0)
        }
    }

    /// Compute e^self via argument reduction and Taylor series.
    pub fn exp(self) -> Self {
        if self.hi == 0.0 && self.lo == 0.0 {
            return Self::ONE;
        }
        let k = (self.hi / Self::LN2.hi).round();
        let r = self - Self::from_f64(k) * Self::LN2;
        let mut term = Self::ONE;
        let mut sum = Self::ONE;
        for i in 1..=25 {
            term = term * r / Self::from_f64(i as f64);
            sum = sum + term;
            if term.abs().hi < 1e-35 { break; }
        }
        let scale = f64::from_bits(((1023_i64 + k as i64) as u64) << 52);
        sum * Self::from_f64(scale)
    }

    /// Natural logarithm via Newton's method.
    pub fn ln(self) -> Self {
        if self.hi <= 0.0 {
            return Self::new(f64::NAN, f64::NAN);
        }
        if self.hi == 1.0 && self.lo == 0.0 {
            return Self::ZERO;
        }
        let mut x = Self::from_f64(self.hi.ln());
        for _ in 0..3 {
            let e = (-x).exp();
            x = x + (self * e - Self::ONE);
        }
        x
    }

    /// self^exp = exp(exp * ln(self))
    pub fn pow(self, exp: Self) -> Self {
        (exp * self.ln()).exp()
    }
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

impl Rem for f128 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        self - q * rhs
    }
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

// --- Parsing ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseF128Error;

impl fmt::Display for ParseF128Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid f128 literal")
    }
}

impl std::error::Error for ParseF128Error {}

impl FromStr for f128 {
    type Err = ParseF128Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() { return Err(ParseF128Error); }

        let mut chars = s.as_bytes();
        let negative = match chars[0] {
            b'-' => { chars = &chars[1..]; true }
            b'+' => { chars = &chars[1..]; false }
            _ => false,
        };
        if chars.is_empty() { return Err(ParseF128Error); }

        // Split on 'e' or 'E'
        let (mantissa, exp_part) = {
            let mut split = None;
            for (i, &c) in chars.iter().enumerate() {
                if c == b'e' || c == b'E' { split = Some(i); break; }
            }
            match split {
                Some(i) => (&chars[..i], Some(&chars[i + 1..])),
                None => (chars, None),
            }
        };
        if mantissa.is_empty() { return Err(ParseF128Error); }

        let ten = Self::from_f64(10.0);
        let mut acc = Self::ZERO;
        let mut saw_dot = false;
        let mut any_digit = false;
        let mut pow10_neg = Self::ONE;

        for &c in mantissa {
            if c == b'.' {
                if saw_dot { return Err(ParseF128Error); }
                saw_dot = true;
                continue;
            }
            if c < b'0' || c > b'9' { return Err(ParseF128Error); }
            any_digit = true;
            let digit = Self::from_f64((c - b'0') as f64);
            if !saw_dot {
                acc = acc * ten + digit;
            } else {
                pow10_neg = pow10_neg / ten;
                acc = acc + digit * pow10_neg;
            }
        }
        if !any_digit { return Err(ParseF128Error); }

        if let Some(exp_bytes) = exp_part {
            if exp_bytes.is_empty() { return Err(ParseF128Error); }
            let exp_str = std::str::from_utf8(exp_bytes).map_err(|_| ParseF128Error)?;
            let exp: i32 = exp_str.parse().map_err(|_| ParseF128Error)?;
            if exp > 0 {
                for _ in 0..exp { acc = acc * ten; }
            } else if exp < 0 {
                for _ in 0..(-exp) { acc = acc / ten; }
            }
        }

        if negative { acc = -acc; }
        Ok(acc)
    }
}

// --- Serde ---

#[cfg(feature = "serde")]
mod serde_impl {
    use super::f128;
    use serde::{Serialize, Serializer, Deserialize, Deserializer};
    use serde::de::{self, Visitor, MapAccess, Unexpected};
    use std::fmt;

    impl Serialize for f128 {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if serializer.is_human_readable() {
                let s = format!("{:+.18e}+{:+.18e}", self.hi, self.lo);
                serializer.serialize_str(&s)
            } else {
                use serde::ser::SerializeStruct;
                let mut st = serializer.serialize_struct("f128", 2)?;
                st.serialize_field("hi", &self.hi)?;
                st.serialize_field("lo", &self.lo)?;
                st.end()
            }
        }
    }

    impl<'de> Deserialize<'de> for f128 {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct F128Visitor;

            impl<'de> Visitor<'de> for F128Visitor {
                type Value = f128;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("a string \"<hi>+<lo>\" or an object {hi, lo}")
                }

                fn visit_str<E: de::Error>(self, v: &str) -> Result<f128, E> {
                    parse_dd_string(v).map_err(|_| {
                        de::Error::invalid_value(Unexpected::Str(v), &self)
                    })
                }

                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<f128, A::Error> {
                    let mut hi: Option<f64> = None;
                    let mut lo: Option<f64> = None;
                    while let Some(key) = map.next_key::<&str>()? {
                        match key {
                            "hi" => hi = Some(map.next_value()?),
                            "lo" => lo = Some(map.next_value()?),
                            _ => { let _ = map.next_value::<serde::de::IgnoredAny>()?; }
                        }
                    }
                    let hi = hi.ok_or_else(|| de::Error::missing_field("hi"))?;
                    let lo = lo.ok_or_else(|| de::Error::missing_field("lo"))?;
                    Ok(f128::new(hi, lo))
                }
            }

            deserializer.deserialize_any(F128Visitor)
        }
    }

    fn parse_dd_string(s: &str) -> Result<f128, ()> {
        let bytes = s.as_bytes();
        for i in 1..bytes.len().saturating_sub(1) {
            if bytes[i] == b'+' && (bytes[i + 1] == b'+' || bytes[i + 1] == b'-') {
                if bytes[i - 1] == b'e' || bytes[i - 1] == b'E' { continue; }
                let hi_str = &s[..i];
                let lo_str = &s[i+1..];
                let hi: f64 = hi_str.parse().map_err(|_| ())?;
                let lo: f64 = lo_str.parse().map_err(|_| ())?;
                return Ok(f128::new(hi, lo));
            }
        }
        Err(())
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

    // --- exp/ln/pow tests ---

    #[test]
    fn exp_zero_is_one() {
        assert_eq!(f128::ZERO.exp(), f128::ONE);
    }

    #[test]
    fn exp_one_is_e() {
        let e = f128::ONE.exp();
        let e_hi = 2.718281828459045_f64;
        let err = (e - f128::new(e_hi, 0.0)).abs();
        assert!(err.hi < 1e-15, "exp(1) hi mismatch: {:e}", err.hi);
        let roundtrip = e * (-f128::ONE).exp();
        let rt_err = (roundtrip - f128::ONE).abs();
        assert!(rt_err.to_f64() < 1e-30, "exp(1)*exp(-1) error: {:e}", rt_err.to_f64());
    }

    #[test]
    fn ln_e_is_one() {
        let e = f128::ONE.exp();
        let result = e.ln();
        let err = (result - f128::ONE).abs();
        assert!(err.to_f64() < 1e-30, "ln(e) error: {:e}", err.to_f64());
    }

    #[test]
    fn ln_one_is_zero() {
        assert_eq!(f128::ONE.ln(), f128::ZERO);
    }

    #[test]
    fn exp_ln_roundtrip() {
        let x = f128::from_f64(3.14159265358979);
        let result = x.ln().exp();
        let err = (result - x).abs();
        assert!(err.to_f64() < 1e-29, "exp(ln(pi)) error: {:e}", err.to_f64());
    }

    #[test]
    fn pow_two_ten() {
        let two = f128::from_f64(2.0);
        let ten = f128::from_f64(10.0);
        let result = two.pow(ten);
        let err = (result - f128::from_f64(1024.0)).abs();
        assert!(err.to_f64() < 1e-25, "2^10 error: {:e}", err.to_f64());
    }

    // --- FromStr tests ---

    #[test]
    fn parse_one_roundtrip() {
        let v: f128 = "1.0".parse().unwrap();
        assert_eq!(v.hi, 1.0);
        assert_eq!(v.lo, 0.0);
    }

    #[test]
    fn parse_pi_full_precision() {
        let pi: f128 = "3.14159265358979323846264338327950288".parse().unwrap();
        let f64_pi = std::f64::consts::PI;
        let dd_val = pi.hi + pi.lo;
        let f64_err = (f64_pi - 3.14159265358979323846264338327950288_f64).abs();
        let dd_err = (dd_val - 3.14159265358979323846264338327950288_f64).abs();
        assert!(dd_err <= f64_err, "dd error {dd_err:e} > f64 error {f64_err:e}");
        assert!(pi.lo != 0.0, "lo should be nonzero for full-precision pi parse");
    }

    #[test]
    fn parse_scientific_notation() {
        let v: f128 = "1.23e-10".parse().unwrap();
        assert!((v.to_f64() - 1.23e-10_f64).abs() < 1e-25);
        let v2: f128 = "5.0E3".parse().unwrap();
        assert!((v2.to_f64() - 5000.0).abs() < 1e-10);
    }

    #[test]
    fn parse_negative() {
        let v: f128 = "-42.5".parse().unwrap();
        assert!((v.to_f64() - (-42.5)).abs() < 1e-30);
    }

    #[test]
    fn parse_invalid() {
        assert!("".parse::<f128>().is_err());
        assert!("abc".parse::<f128>().is_err());
        assert!("1.2.3".parse::<f128>().is_err());
        assert!("-".parse::<f128>().is_err());
        assert!("1.0e".parse::<f128>().is_err());
    }
}
