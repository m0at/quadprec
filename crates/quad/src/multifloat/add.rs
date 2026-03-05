use std::ops::{Add, Sub, Neg, AddAssign, SubAssign};
use super::core::{MultiFloat, two_sum, renormalize};

impl<const N: usize> Neg for MultiFloat<N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        let mut out = [0.0f64; N];
        let mut i = 0;
        while i < N {
            out[i] = -self.limbs[i];
            i += 1;
        }
        MultiFloat { limbs: out }
    }
}

impl<const N: usize> Add for MultiFloat<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Merge-expansion addition (Shewchuk-style):
        //   1. Pairwise two_sum of corresponding limbs -> 2N terms
        //   2. Sort all 2N terms by descending magnitude
        //   3. Grow-expansion: accumulate into a non-overlapping expansion
        //   4. Renormalize top N limbs
        const MAX_WORK: usize = 32;
        let work_len = 2 * N;
        debug_assert!(work_len <= MAX_WORK);
        let mut work = [0.0f64; MAX_WORK];

        let mut i = 0;
        while i < N {
            let (s, e) = two_sum(self.limbs[i], rhs.limbs[i]);
            work[2 * i] = s;
            work[2 * i + 1] = e;
            i += 1;
        }

        // Insertion sort by descending magnitude
        i = 1;
        while i < work_len {
            let val = work[i];
            let mag = val.abs();
            let mut j = i;
            while j > 0 && work[j - 1].abs() < mag {
                work[j] = work[j - 1];
                j -= 1;
            }
            work[j] = val;
            i += 1;
        }

        // Grow-expansion: accumulate sorted terms into a non-overlapping
        // expansion. exp[0] is most significant, exp[exp_len-1] least.
        // For each new term we walk from least-significant upward,
        // two_sum-ing at each level.
        let mut exp = [0.0f64; MAX_WORK];
        let mut exp_len: usize = 0;

        i = 0;
        while i < work_len {
            let term = work[i];
            i += 1;
            if term == 0.0 {
                continue;
            }
            if exp_len == 0 {
                exp[0] = term;
                exp_len = 1;
                continue;
            }
            // Walk from tail (least significant) toward head
            let mut carry = term;
            let mut j = exp_len;
            while j > 0 {
                j -= 1;
                let (s, e) = two_sum(exp[j], carry);
                exp[j] = e;
                carry = s;
            }
            if carry != 0.0 {
                // Shift right to make room at index 0
                let new_len = if exp_len < MAX_WORK { exp_len + 1 } else { MAX_WORK };
                let mut k = new_len - 1;
                while k > 0 {
                    exp[k] = exp[k - 1];
                    k -= 1;
                }
                exp[0] = carry;
                exp_len = new_len;
            }
        }

        let mut limbs = [0.0f64; N];
        i = 0;
        while i < N && i < exp_len {
            limbs[i] = exp[i];
            i += 1;
        }
        renormalize(&mut limbs);
        MultiFloat { limbs }
    }
}

impl<const N: usize> Sub for MultiFloat<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const N: usize> AddAssign for MultiFloat<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl<const N: usize> SubAssign for MultiFloat<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

#[cfg(test)]
mod tests {
    use super::super::core::MultiFloat;
    type F256 = MultiFloat<4>;
    type F512 = MultiFloat<8>;

    #[test]
    fn add_zero() {
        let a = F256::from_f64(1.0);
        let r = a + F256::ZERO;
        assert_eq!(r.limbs[0], 1.0);
    }

    #[test]
    fn add_small_cancellation() {
        let a = F256::from_f64(1.0);
        let b = F256::from_f64(1e-20);
        let c = a + b;
        let d = c - a;
        assert!((d.to_f64() - 1e-20).abs() < 1e-35);
    }

    #[test]
    fn neg_roundtrip() {
        let a = F256::from_f64(3.14);
        assert_eq!(a.limbs[0], (-(-a)).limbs[0]);
    }

    #[test]
    fn sub_self_is_zero() {
        let a = F256::from_f64(42.0);
        assert_eq!((a - a).to_f64(), 0.0);
    }

    #[test]
    fn f512_add() {
        let a = F512::from_f64(1.0);
        let b = F512::from_f64(1e-40);
        let d = (a + b) - a;
        assert!((d.to_f64() - 1e-40).abs() < 1e-55);
    }
}
