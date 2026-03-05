use std::ops::{Add, Sub};
use std::iter::{Sum, Product};
use super::core::{MultiFloat, two_sum, renormalize};

// --- From conversions ---

impl<const N: usize> From<f64> for MultiFloat<N> {
    #[inline(always)]
    fn from(x: f64) -> Self {
        let mut limbs = [0.0f64; N];
        limbs[0] = x;
        Self { limbs }
    }
}

impl<const N: usize> From<f32> for MultiFloat<N> {
    #[inline(always)]
    fn from(x: f32) -> Self {
        Self::from(x as f64)
    }
}

impl<const N: usize> From<i32> for MultiFloat<N> {
    #[inline(always)]
    fn from(x: i32) -> Self {
        Self::from(x as f64)
    }
}

impl<const N: usize> From<u32> for MultiFloat<N> {
    #[inline(always)]
    fn from(x: u32) -> Self {
        Self::from(x as f64)
    }
}

impl<const N: usize> From<i64> for MultiFloat<N> {
    fn from(x: i64) -> Self {
        let hi_part = (x >> 32) as f64 * (1u64 << 32) as f64;
        let lo_part = (x as u64 & 0xFFFFFFFF) as f64;
        let (s, e) = two_sum(hi_part, lo_part);
        let mut limbs = [0.0f64; N];
        limbs[0] = s;
        if N > 1 {
            limbs[1] = e;
        }
        Self { limbs }
    }
}

impl<const N: usize> From<u64> for MultiFloat<N> {
    fn from(x: u64) -> Self {
        let hi_part = (x >> 32) as f64 * (1u64 << 32) as f64;
        let lo_part = (x & 0xFFFFFFFF) as f64;
        let (s, e) = two_sum(hi_part, lo_part);
        let mut limbs = [0.0f64; N];
        limbs[0] = s;
        if N > 1 {
            limbs[1] = e;
        }
        Self { limbs }
    }
}

// --- Mixed f64 arithmetic ---

impl<const N: usize> Add<f64> for MultiFloat<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        let mut result = self;
        let (s, mut e) = two_sum(result.limbs[0], rhs);
        result.limbs[0] = s;
        for i in 1..N {
            let (s, e2) = two_sum(result.limbs[i], e);
            result.limbs[i] = s;
            e = e2;
        }
        renormalize(&mut result.limbs);
        result
    }
}

impl<const N: usize> Sub<f64> for MultiFloat<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        self + (-rhs)
    }
}

// --- Iterator traits ---

impl<const N: usize> Sum for MultiFloat<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a, const N: usize> Sum<&'a MultiFloat<N>> for MultiFloat<N> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + *b)
    }
}

impl<const N: usize> Product for MultiFloat<N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}

impl<'a, const N: usize> Product<&'a MultiFloat<N>> for MultiFloat<N> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * *b)
    }
}

// --- Cross-precision conversions ---

impl MultiFloat<4> {
    pub fn from_f128(x: crate::f128) -> Self {
        let mut limbs = [0.0f64; 4];
        limbs[0] = x.hi;
        limbs[1] = x.lo;
        Self { limbs }
    }

    pub fn to_f128(self) -> crate::f128 {
        crate::f128::new(self.limbs[0], self.limbs[1])
    }
}

impl MultiFloat<8> {
    pub fn from_f128(x: crate::f128) -> Self {
        let mut limbs = [0.0f64; 8];
        limbs[0] = x.hi;
        limbs[1] = x.lo;
        Self { limbs }
    }

    pub fn to_f128(self) -> crate::f128 {
        crate::f128::new(self.limbs[0], self.limbs[1])
    }

    pub fn from_f256(x: MultiFloat<4>) -> Self {
        let mut limbs = [0.0f64; 8];
        limbs[0] = x.limbs[0];
        limbs[1] = x.limbs[1];
        limbs[2] = x.limbs[2];
        limbs[3] = x.limbs[3];
        Self { limbs }
    }

    pub fn to_f256(self) -> MultiFloat<4> {
        let mut limbs = [0.0f64; 4];
        limbs[0] = self.limbs[0];
        limbs[1] = self.limbs[1];
        limbs[2] = self.limbs[2];
        limbs[3] = self.limbs[3];
        MultiFloat::<4> { limbs }
    }
}
