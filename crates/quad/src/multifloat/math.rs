use super::core::MultiFloat;

impl<const N: usize> MultiFloat<N> {
    #[inline]
    pub fn is_sign_negative(&self) -> bool {
        self.limbs[0].is_sign_negative()
    }

    #[inline]
    pub fn nan() -> Self {
        Self::from_f64(f64::NAN)
    }

    pub fn abs(self) -> Self {
        if self.limbs[0].is_sign_negative() {
            -self
        } else {
            self
        }
    }

    pub fn sqrt(self) -> Self {
        if self.limbs[0] < 0.0 {
            return Self::nan();
        }
        if self.limbs[0] == 0.0 {
            return Self::ZERO;
        }
        if !self.limbs[0].is_finite() {
            return Self::from_f64(self.limbs[0]);
        }

        let mut x = Self::from_f64(self.limbs[0].sqrt());
        let half = Self::from_f64(0.5);
        let steps = (N as f64).log2().ceil() as usize + 1;
        for _ in 0..steps {
            x = (x + self / x) * half;
        }
        x
    }

    pub fn trunc(self) -> Self {
        let t0 = self.limbs[0].trunc();
        if t0 != self.limbs[0] {
            return Self::from_f64(t0);
        }
        let mut result = [0.0_f64; N];
        result[0] = t0;
        for i in 1..N {
            let ti = self.limbs[i].trunc();
            result[i] = ti;
            if ti != self.limbs[i] {
                break;
            }
        }
        for i in (1..N).rev() {
            let s = result[i - 1] + result[i];
            let e = result[i] - (s - result[i - 1]);
            result[i - 1] = s;
            result[i] = e;
        }
        MultiFloat { limbs: result }
    }
}
