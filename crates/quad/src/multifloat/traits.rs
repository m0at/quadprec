use super::core::MultiFloat;

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::{self, Visitor, SeqAccess, Unexpected};
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::fmt;

    impl<const N: usize> Serialize for MultiFloat<N> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if serializer.is_human_readable() {
                let mut s = String::new();
                for (i, &limb) in self.limbs.iter().enumerate() {
                    if i > 0 {
                        s.push('+');
                    }
                    use std::fmt::Write;
                    write!(s, "{:+.14e}", limb).unwrap();
                }
                serializer.serialize_str(&s)
            } else {
                let mut st = serializer.serialize_struct("MultiFloat", 1)?;
                st.serialize_field("limbs", &self.limbs.as_slice())?;
                st.end()
            }
        }
    }

    impl<'de, const N: usize> Deserialize<'de> for MultiFloat<N> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct MfVisitor<const M: usize>;

            impl<'de, const M: usize> Visitor<'de> for MfVisitor<M> {
                type Value = MultiFloat<M>;

                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(
                        f,
                        "a string of {} limbs in scientific notation separated by '+', \
                         or a struct with a \"limbs\" array",
                        M
                    )
                }

                fn visit_str<E: de::Error>(self, v: &str) -> Result<MultiFloat<M>, E> {
                    parse_limbs_string::<M>(v).map_err(|_| {
                        de::Error::invalid_value(Unexpected::Str(v), &self)
                    })
                }

                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> Result<MultiFloat<M>, A::Error> {
                    let mut limbs = [0.0f64; M];
                    for i in 0..M {
                        limbs[i] = seq
                            .next_element()?
                            .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                    }
                    Ok(MultiFloat { limbs })
                }

                fn visit_map<A: de::MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> Result<MultiFloat<M>, A::Error> {
                    let mut limbs: Option<[f64; M]> = None;
                    while let Some(key) = map.next_key::<&str>()? {
                        match key {
                            "limbs" => {
                                let v: Vec<f64> = map.next_value()?;
                                if v.len() != M {
                                    return Err(de::Error::invalid_length(v.len(), &self));
                                }
                                let mut arr = [0.0f64; M];
                                arr.copy_from_slice(&v);
                                limbs = Some(arr);
                            }
                            _ => {
                                let _ = map.next_value::<de::IgnoredAny>()?;
                            }
                        }
                    }
                    let limbs = limbs.ok_or_else(|| de::Error::missing_field("limbs"))?;
                    Ok(MultiFloat { limbs })
                }
            }

            deserializer.deserialize_any(MfVisitor::<N>)
        }
    }

    /// Parse a human-readable serialization string back into limbs.
    ///
    /// Format: "+6.93147180559945e-1++2.31904681384630e-17+..."
    /// Each limb is in scientific notation with a leading sign; limbs are separated by '+'.
    /// Because each limb has a leading '+' or '-', the separator '+' is always followed
    /// by another '+' or '-', which lets us split unambiguously.
    fn parse_limbs_string<const N: usize>(s: &str) -> Result<MultiFloat<N>, ()> {
        let mut limbs = [0.0f64; N];
        let mut count = 0;
        let bytes = s.as_bytes();
        let mut start = 0;

        // Walk through the string finding limb boundaries.
        // A limb boundary is a '+' that is NOT part of an exponent (preceded by 'e'/'E')
        // AND is followed by '+' or '-' (the sign of the next limb).
        let mut i = 1;
        while i < bytes.len() {
            let is_boundary = bytes[i] == b'+'
                && i + 1 < bytes.len()
                && (bytes[i + 1] == b'+' || bytes[i + 1] == b'-')
                && bytes[i - 1] != b'e'
                && bytes[i - 1] != b'E';

            if is_boundary {
                if count >= N {
                    return Err(());
                }
                let limb_str = &s[start..i];
                limbs[count] = limb_str.parse().map_err(|_| ())?;
                count += 1;
                start = i + 1; // skip the separator '+'
                i = start + 1;
            } else {
                i += 1;
            }
        }

        // Last limb
        if count >= N {
            return Err(());
        }
        let limb_str = &s[start..];
        limbs[count] = limb_str.parse().map_err(|_| ())?;
        count += 1;

        if count != N {
            return Err(());
        }
        Ok(MultiFloat { limbs })
    }
}

// ---------------------------------------------------------------------------
// num-traits
// ---------------------------------------------------------------------------

#[cfg(feature = "num-traits")]
mod num_impl {
    use super::*;
    use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};

    impl<const N: usize> Zero for MultiFloat<N> {
        #[inline]
        fn zero() -> Self {
            Self::ZERO
        }
        #[inline]
        fn is_zero(&self) -> bool {
            self.limbs[0] == 0.0
        }
    }

    impl<const N: usize> One for MultiFloat<N> {
        #[inline]
        fn one() -> Self {
            Self::one()
        }
    }

    impl<const N: usize> ToPrimitive for MultiFloat<N> {
        fn to_i64(&self) -> Option<i64> {
            let v = self.to_f64();
            if v.is_finite() && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                Some(v as i64)
            } else {
                None
            }
        }
        fn to_u64(&self) -> Option<u64> {
            let v = self.to_f64();
            if v.is_finite() && v >= 0.0 && v <= u64::MAX as f64 {
                Some(v as u64)
            } else {
                None
            }
        }
        fn to_f64(&self) -> Option<f64> {
            Some(MultiFloat::to_f64(*self))
        }
        fn to_f32(&self) -> Option<f32> {
            Some(MultiFloat::to_f64(*self) as f32)
        }
    }

    impl<const N: usize> FromPrimitive for MultiFloat<N> {
        fn from_i64(n: i64) -> Option<Self> {
            Some(<MultiFloat<N> as From<i64>>::from(n))
        }
        fn from_u64(n: u64) -> Option<Self> {
            Some(<MultiFloat<N> as From<u64>>::from(n))
        }
        fn from_f64(n: f64) -> Option<Self> {
            Some(MultiFloat::from_f64(n))
        }
        fn from_f32(n: f32) -> Option<Self> {
            Some(MultiFloat::from_f64(n as f64))
        }
    }

    impl<const N: usize> NumCast for MultiFloat<N> {
        fn from<T: ToPrimitive>(n: T) -> Option<Self> {
            n.to_f64().map(MultiFloat::from_f64)
        }
    }

    impl<const N: usize> Num for MultiFloat<N> {
        type FromStrRadixErr = &'static str;

        fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
            if radix != 10 {
                return Err("MultiFloat only supports base-10 parsing");
            }
            src.parse::<MultiFloat<N>>()
                .map_err(|_| "failed to parse MultiFloat")
        }
    }

    impl<const N: usize> Float for MultiFloat<N> {
        fn nan() -> Self {
            Self::from_f64(f64::NAN)
        }
        fn infinity() -> Self {
            Self::from_f64(f64::INFINITY)
        }
        fn neg_infinity() -> Self {
            Self::from_f64(f64::NEG_INFINITY)
        }
        fn neg_zero() -> Self {
            Self::from_f64(-0.0)
        }
        fn min_value() -> Self {
            Self::from_f64(f64::MIN)
        }
        fn min_positive_value() -> Self {
            Self::from_f64(f64::MIN_POSITIVE)
        }
        fn max_value() -> Self {
            Self::from_f64(f64::MAX)
        }

        fn is_nan(self) -> bool {
            self.limbs[0].is_nan()
        }
        fn is_infinite(self) -> bool {
            self.limbs[0].is_infinite()
        }
        fn is_finite(self) -> bool {
            self.limbs[0].is_finite()
        }
        fn is_normal(self) -> bool {
            self.limbs[0].is_normal()
        }
        fn is_sign_positive(self) -> bool {
            self.limbs[0].is_sign_positive()
        }
        fn is_sign_negative(self) -> bool {
            self.limbs[0].is_sign_negative()
        }

        fn classify(self) -> std::num::FpCategory {
            self.limbs[0].classify()
        }

        fn floor(self) -> Self {
            let hi_floor = self.limbs[0].floor();
            if hi_floor == self.limbs[0] {
                // Leading limb is integer; floor the next limb and propagate.
                let mut result = self;
                for i in 1..N {
                    let t = result.limbs[i].floor();
                    result.limbs[i] = t;
                    if t != self.limbs[i] {
                        // All subsequent limbs become zero (fractional part removed).
                        for j in (i + 1)..N {
                            result.limbs[j] = 0.0;
                        }
                        break;
                    }
                }
                result.renorm();
                result
            } else {
                Self::from_f64(hi_floor)
            }
        }

        fn ceil(self) -> Self {
            let hi_ceil = self.limbs[0].ceil();
            if hi_ceil == self.limbs[0] {
                let mut result = self;
                for i in 1..N {
                    let t = result.limbs[i].ceil();
                    result.limbs[i] = t;
                    if t != self.limbs[i] {
                        for j in (i + 1)..N {
                            result.limbs[j] = 0.0;
                        }
                        break;
                    }
                }
                result.renorm();
                result
            } else {
                Self::from_f64(hi_ceil)
            }
        }

        fn round(self) -> Self {
            let hi_round = self.limbs[0].round();
            if hi_round == self.limbs[0] {
                let mut result = self;
                for i in 1..N {
                    let t = result.limbs[i].round();
                    result.limbs[i] = t;
                    if t != self.limbs[i] {
                        for j in (i + 1)..N {
                            result.limbs[j] = 0.0;
                        }
                        break;
                    }
                }
                result.renorm();
                result
            } else {
                let diff = hi_round - self.limbs[0];
                if diff.abs() == 0.5 && N >= 2 {
                    if self.limbs[1] > 0.0 {
                        Self::from_f64(self.limbs[0].ceil())
                    } else if self.limbs[1] < 0.0 {
                        Self::from_f64(self.limbs[0].floor())
                    } else {
                        Self::from_f64(hi_round)
                    }
                } else {
                    Self::from_f64(hi_round)
                }
            }
        }

        fn trunc(self) -> Self {
            MultiFloat::trunc(self)
        }

        fn fract(self) -> Self {
            self - self.trunc()
        }

        fn abs(self) -> Self {
            MultiFloat::abs(self)
        }

        fn signum(self) -> Self {
            if self.is_nan() {
                Self::nan()
            } else if self.limbs[0] > 0.0
                || (self.limbs[0] == 0.0 && N >= 2 && self.limbs[1] > 0.0)
            {
                Self::one()
            } else if self.limbs[0] < 0.0
                || (self.limbs[0] == 0.0 && N >= 2 && self.limbs[1] < 0.0)
            {
                -Self::one()
            } else {
                Self::ZERO
            }
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            self * a + b
        }

        fn recip(self) -> Self {
            MultiFloat::recip(self)
        }

        fn powi(self, n: i32) -> Self {
            if n == 0 {
                return Self::one();
            }
            let mut result = Self::one();
            let mut base = self;
            let mut exp = n.unsigned_abs();
            while exp > 0 {
                if exp & 1 == 1 {
                    result = result * base;
                }
                base = base * base;
                exp >>= 1;
            }
            if n < 0 {
                result.recip()
            } else {
                result
            }
        }

        fn powf(self, n: Self) -> Self {
            MultiFloat::pow(self, n)
        }

        fn sqrt(self) -> Self {
            MultiFloat::sqrt(self)
        }

        fn exp(self) -> Self {
            MultiFloat::exp(self)
        }

        fn exp2(self) -> Self {
            // 2^self = exp(self * ln2)
            (self * Self::ln2()).exp()
        }

        fn ln(self) -> Self {
            MultiFloat::ln(self)
        }

        fn log(self, base: Self) -> Self {
            self.ln() / base.ln()
        }

        fn log2(self) -> Self {
            self.ln() / Self::ln2()
        }

        fn log10(self) -> Self {
            self.ln() / Self::from_f64(10.0_f64.ln())
        }

        fn max(self, other: Self) -> Self {
            if self >= other {
                self
            } else {
                other
            }
        }

        fn min(self, other: Self) -> Self {
            if self <= other {
                self
            } else {
                other
            }
        }

        fn abs_sub(self, other: Self) -> Self {
            let d = self - other;
            if d > Self::ZERO {
                d
            } else {
                Self::ZERO
            }
        }

        fn cbrt(self) -> Self {
            // TODO: full-precision cbrt via Newton iteration
            Self::from_f64(self.to_f64().cbrt())
        }

        fn hypot(self, other: Self) -> Self {
            (self * self + other * other).sqrt()
        }

        fn sin(self) -> Self {
            // TODO: implement full-precision sin
            Self::from_f64(self.to_f64().sin())
        }

        fn cos(self) -> Self {
            // TODO: implement full-precision cos
            Self::from_f64(self.to_f64().cos())
        }

        fn tan(self) -> Self {
            // TODO: implement full-precision tan
            Self::from_f64(self.to_f64().tan())
        }

        fn asin(self) -> Self {
            // TODO: implement full-precision asin
            Self::from_f64(self.to_f64().asin())
        }

        fn acos(self) -> Self {
            // TODO: implement full-precision acos
            Self::from_f64(self.to_f64().acos())
        }

        fn atan(self) -> Self {
            // TODO: implement full-precision atan
            Self::from_f64(self.to_f64().atan())
        }

        fn atan2(self, other: Self) -> Self {
            // TODO: implement full-precision atan2
            Self::from_f64(self.to_f64().atan2(other.to_f64()))
        }

        fn sin_cos(self) -> (Self, Self) {
            // TODO: implement full-precision sin_cos
            let (s, c) = self.to_f64().sin_cos();
            (Self::from_f64(s), Self::from_f64(c))
        }

        fn exp_m1(self) -> Self {
            // exp(x) - 1, using the full-precision exp
            self.exp() - Self::one()
        }

        fn ln_1p(self) -> Self {
            // ln(1 + x), using the full-precision ln
            (Self::one() + self).ln()
        }

        fn sinh(self) -> Self {
            // TODO: implement full-precision sinh
            Self::from_f64(self.to_f64().sinh())
        }

        fn cosh(self) -> Self {
            // TODO: implement full-precision cosh
            Self::from_f64(self.to_f64().cosh())
        }

        fn tanh(self) -> Self {
            // TODO: implement full-precision tanh
            Self::from_f64(self.to_f64().tanh())
        }

        fn asinh(self) -> Self {
            // TODO: implement full-precision asinh
            Self::from_f64(self.to_f64().asinh())
        }

        fn acosh(self) -> Self {
            // TODO: implement full-precision acosh
            Self::from_f64(self.to_f64().acosh())
        }

        fn atanh(self) -> Self {
            // TODO: implement full-precision atanh
            Self::from_f64(self.to_f64().atanh())
        }

        fn integer_decode(self) -> (u64, i16, i8) {
            let bits = self.limbs[0].to_bits();
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
            // Each limb adds ~53 bits of precision, so epsilon ~ f64::EPSILON^N.
            // f64::EPSILON = 2^-52, so MultiFloat<N> epsilon ~ 2^(-52*N).
            let mut eps = 1.0_f64;
            for _ in 0..N {
                eps *= f64::EPSILON;
            }
            Self::from_f64(eps)
        }

        fn to_degrees(self) -> Self {
            self * Self::from_f64(180.0) / Self::pi()
        }

        fn to_radians(self) -> Self {
            self * Self::pi() / Self::from_f64(180.0)
        }
    }
}
