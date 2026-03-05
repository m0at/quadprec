use crate::f128;
use num_traits::{
    Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero,
};

impl Zero for f128 {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.hi == 0.0 && self.lo == 0.0
    }
}

impl One for f128 {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

impl ToPrimitive for f128 {
    fn to_i64(&self) -> Option<i64> {
        let v = self.hi + self.lo;
        if v.is_finite() && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
            Some(v as i64)
        } else {
            None
        }
    }
    fn to_u64(&self) -> Option<u64> {
        let v = self.hi + self.lo;
        if v.is_finite() && v >= 0.0 && v <= u64::MAX as f64 {
            Some(v as u64)
        } else {
            None
        }
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.hi + self.lo)
    }
    fn to_f32(&self) -> Option<f32> {
        Some((self.hi + self.lo) as f32)
    }
}

impl FromPrimitive for f128 {
    fn from_i64(n: i64) -> Option<Self> {
        Some(<f128 as From<i64>>::from(n))
    }
    fn from_u64(n: u64) -> Option<Self> {
        let hi = n as f64;
        let lo = (n.wrapping_sub(hi as u64)) as f64;
        Some(Self::new(hi, lo))
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(f128::from_f64(n))
    }
    fn from_f32(n: f32) -> Option<Self> {
        Some(f128::from_f64(n as f64))
    }
}

impl NumCast for f128 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(f128::from_f64)
    }
}

impl Num for f128 {
    type FromStrRadixErr = &'static str;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return Err("f128 only supports base-10 parsing");
        }
        let v: f64 = src.parse().map_err(|_| "failed to parse f128")?;
        Ok(f128::from_f64(v))
    }
}

impl Float for f128 {
    fn nan() -> Self {
        Self::new(f64::NAN, 0.0)
    }
    fn infinity() -> Self {
        Self::new(f64::INFINITY, 0.0)
    }
    fn neg_infinity() -> Self {
        Self::new(f64::NEG_INFINITY, 0.0)
    }
    fn neg_zero() -> Self {
        Self::new(-0.0, 0.0)
    }
    fn min_value() -> Self {
        Self::new(f64::MIN, 0.0)
    }
    fn min_positive_value() -> Self {
        Self::new(f64::MIN_POSITIVE, 0.0)
    }
    fn max_value() -> Self {
        Self::new(f64::MAX, 0.0)
    }

    fn is_nan(self) -> bool {
        self.hi.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.hi.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.hi.is_finite()
    }
    fn is_normal(self) -> bool {
        self.hi.is_normal()
    }
    fn is_sign_positive(self) -> bool {
        self.hi.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.hi.is_sign_negative()
    }

    fn classify(self) -> std::num::FpCategory {
        self.hi.classify()
    }

    fn floor(self) -> Self {
        let hi_floor = self.hi.floor();
        if hi_floor == self.hi {
            f128::renorm(hi_floor, self.lo.floor())
        } else {
            Self::new(hi_floor, 0.0)
        }
    }
    fn ceil(self) -> Self {
        let hi_ceil = self.hi.ceil();
        if hi_ceil == self.hi {
            f128::renorm(hi_ceil, self.lo.ceil())
        } else {
            Self::new(hi_ceil, 0.0)
        }
    }
    fn round(self) -> Self {
        let hi_round = self.hi.round();
        if hi_round == self.hi {
            f128::renorm(hi_round, self.lo.round())
        } else {
            let diff = hi_round - self.hi;
            if diff.abs() == 0.5 {
                if self.lo > 0.0 {
                    Self::new(self.hi.ceil(), 0.0)
                } else if self.lo < 0.0 {
                    Self::new(self.hi.floor(), 0.0)
                } else {
                    Self::new(hi_round, 0.0)
                }
            } else {
                Self::new(hi_round, 0.0)
            }
        }
    }
    fn trunc(self) -> Self {
        f128::trunc(self)
    }
    fn fract(self) -> Self {
        self - self.trunc()
    }

    fn abs(self) -> Self {
        f128::abs(self)
    }
    fn signum(self) -> Self {
        if self.is_nan() {
            Self::nan()
        } else if self.hi > 0.0 || (self.hi == 0.0 && self.lo > 0.0) {
            Self::ONE
        } else if self.hi < 0.0 || (self.hi == 0.0 && self.lo < 0.0) {
            -Self::ONE
        } else {
            Self::ZERO
        }
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    fn recip(self) -> Self {
        f128::recip(self)
    }
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::ONE;
        }
        let mut result = Self::ONE;
        let mut base = self;
        let mut exp = n.unsigned_abs();
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        if n < 0 { result.recip() } else { result }
    }
    fn powf(self, n: Self) -> Self {
        let v = (self.hi + self.lo).powf(n.hi + n.lo);
        Self::from_f64(v)
    }
    fn sqrt(self) -> Self {
        f128::sqrt(self)
    }

    fn exp(self) -> Self {
        Self::from_f64((self.hi + self.lo).exp())
    }
    fn exp2(self) -> Self {
        Self::from_f64((self.hi + self.lo).exp2())
    }
    fn ln(self) -> Self {
        Self::from_f64((self.hi + self.lo).ln())
    }
    fn log(self, base: Self) -> Self {
        Self::from_f64((self.hi + self.lo).log(base.hi + base.lo))
    }
    fn log2(self) -> Self {
        Self::from_f64((self.hi + self.lo).log2())
    }
    fn log10(self) -> Self {
        Self::from_f64((self.hi + self.lo).log10())
    }
    fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
    fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
    fn abs_sub(self, other: Self) -> Self {
        let d = self - other;
        if d > Self::ZERO { d } else { Self::ZERO }
    }
    fn cbrt(self) -> Self {
        Self::from_f64((self.hi + self.lo).cbrt())
    }
    fn hypot(self, other: Self) -> Self {
        (self * self + other * other).sqrt()
    }
    fn sin(self) -> Self {
        Self::from_f64((self.hi + self.lo).sin())
    }
    fn cos(self) -> Self {
        Self::from_f64((self.hi + self.lo).cos())
    }
    fn tan(self) -> Self {
        Self::from_f64((self.hi + self.lo).tan())
    }
    fn asin(self) -> Self {
        Self::from_f64((self.hi + self.lo).asin())
    }
    fn acos(self) -> Self {
        Self::from_f64((self.hi + self.lo).acos())
    }
    fn atan(self) -> Self {
        Self::from_f64((self.hi + self.lo).atan())
    }
    fn atan2(self, other: Self) -> Self {
        Self::from_f64((self.hi + self.lo).atan2(other.hi + other.lo))
    }
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = (self.hi + self.lo).sin_cos();
        (Self::from_f64(s), Self::from_f64(c))
    }
    fn exp_m1(self) -> Self {
        Self::from_f64((self.hi + self.lo).exp_m1())
    }
    fn ln_1p(self) -> Self {
        Self::from_f64((self.hi + self.lo).ln_1p())
    }
    fn sinh(self) -> Self {
        Self::from_f64((self.hi + self.lo).sinh())
    }
    fn cosh(self) -> Self {
        Self::from_f64((self.hi + self.lo).cosh())
    }
    fn tanh(self) -> Self {
        Self::from_f64((self.hi + self.lo).tanh())
    }
    fn asinh(self) -> Self {
        Self::from_f64((self.hi + self.lo).asinh())
    }
    fn acosh(self) -> Self {
        Self::from_f64((self.hi + self.lo).acosh())
    }
    fn atanh(self) -> Self {
        Self::from_f64((self.hi + self.lo).atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.hi.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let exp = ((bits >> 52) & 0x7ff) as i16 - 1023 - 52;
        let mantissa = if exp == -1023 - 52 {
            (bits & 0x000f_ffff_ffff_ffff) << 1
        } else {
            (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
        };
        (mantissa, exp, sign)
    }

    fn epsilon() -> Self {
        // dd epsilon is roughly 2^-105
        Self::from_f64(f64::EPSILON * f64::EPSILON * 0.5)
    }

    fn to_degrees(self) -> Self {
        self * Self::from_f64(180.0) / Self::from_f64(std::f64::consts::PI)
    }
    fn to_radians(self) -> Self {
        self * Self::from_f64(std::f64::consts::PI) / Self::from_f64(180.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::f128;
    use num_traits::{Float, NumCast, One, Zero};

    #[test]
    fn zero_one() {
        let z: f128 = Zero::zero();
        assert!(z.is_zero());
        let o: f128 = One::one();
        assert_eq!(o.hi + o.lo, 1.0);
    }

    #[test]
    fn num_cast() {
        let v: f128 = NumCast::from(42i32).unwrap();
        assert_eq!(v.hi + v.lo, 42.0);
        let v2: f128 = NumCast::from(3.14f64).unwrap();
        assert!((v2.hi + v2.lo - 3.14).abs() < 1e-15);
    }

    #[test]
    fn float_special_values() {
        assert!(f128::nan().is_nan());
        assert!(f128::infinity().is_infinite());
        assert!(f128::neg_infinity().is_infinite());
        assert!(f128::infinity().is_sign_positive());
        assert!(f128::neg_infinity().is_sign_negative());
    }

    #[test]
    fn float_floor_ceil_round_trunc() {
        let v = f128::from_f64(2.7);
        assert_eq!(Float::floor(v).hi, 2.0);
        assert_eq!(Float::ceil(v).hi, 3.0);
        assert_eq!(Float::round(v).hi, 3.0);
        assert_eq!(Float::trunc(v).hi, 2.0);

        let neg = f128::from_f64(-2.3);
        assert_eq!(Float::floor(neg).hi, -3.0);
        assert_eq!(Float::ceil(neg).hi, -2.0);
        assert_eq!(Float::trunc(neg).hi, -2.0);
    }

    #[test]
    fn float_abs_signum() {
        let neg = f128::from_f64(-5.0);
        assert_eq!(Float::abs(neg).hi, 5.0);
        assert_eq!(Float::signum(neg).hi, -1.0);
        assert_eq!(Float::signum(f128::from_f64(3.0)).hi, 1.0);
    }

    #[test]
    fn float_min_max() {
        let a = f128::from_f64(1.0);
        let b = f128::from_f64(2.0);
        assert_eq!(Float::min(a, b).hi, 1.0);
        assert_eq!(Float::max(a, b).hi, 2.0);
    }

    #[test]
    fn float_recip_powi() {
        let v = f128::from_f64(4.0);
        let r = Float::recip(v);
        assert!((r.hi + r.lo - 0.25).abs() < 1e-30);
        let p = Float::powi(v, 3);
        assert!((p.hi + p.lo - 64.0).abs() < 1e-28);
        let pi = Float::powi(v, -1);
        assert!((pi.hi + pi.lo - 0.25).abs() < 1e-30);
    }

    #[test]
    fn float_fract() {
        let v = f128::from_f64(3.75);
        let fr = Float::fract(v);
        assert!((fr.hi + fr.lo - 0.75).abs() < 1e-15);
    }

    #[test]
    fn from_str_radix() {
        use num_traits::Num;
        let v = f128::from_str_radix("3.14", 10).unwrap();
        assert!((v.hi + v.lo - 3.14).abs() < 1e-15);
        assert!(f128::from_str_radix("ff", 16).is_err());
    }

    #[test]
    fn to_from_primitive() {
        use num_traits::{FromPrimitive, ToPrimitive};
        let v = f128::from_f64(42.5);
        assert_eq!(ToPrimitive::to_i64(&v), Some(42));
        assert_eq!(ToPrimitive::to_f64(&v), Some(42.5));

        let w = <f128 as FromPrimitive>::from_i64(100).unwrap();
        assert_eq!(w.hi + w.lo, 100.0);
    }

    #[test]
    fn float_is_finite_normal() {
        assert!(f128::from_f64(1.0).is_finite());
        assert!(Float::is_normal(f128::from_f64(1.0)));
        assert!(!f128::nan().is_finite());
        assert!(!f128::infinity().is_finite());
        assert!(!Float::is_normal(f128::from_f64(0.0)));
    }

    #[test]
    fn remainder() {
        let a = f128::from_f64(7.0);
        let b = f128::from_f64(3.0);
        let r = a % b;
        assert!((r.hi + r.lo - 1.0).abs() < 1e-30);
    }
}
