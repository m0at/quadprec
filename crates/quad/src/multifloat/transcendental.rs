use super::core::MultiFloat;

// LN2 limbs at quad-double precision (4 non-overlapping f64, ~212 bits)
const LN2_LIMBS_4: [f64; 4] = [
    6.931471805599452862e-01,
    2.319046813846299558e-17,
    5.707708438416212066e-34,
   -3.582432210601811423e-50,
];

const PI_LIMBS_4: [f64; 4] = [
    3.141592653589793116e+00,
    1.224646799147353207e-16,
   -2.994769809718339666e-33,
    1.112454220863365282e-49,
];

const E_LIMBS_4: [f64; 4] = [
    2.718281828459045091e+00,
    1.445646891729250158e-16,
   -2.127717108038176765e-33,
    1.515630159841219144e-49,
];

/// Build a MultiFloat<N> from a slice of limbs, zero-extending or truncating.
/// Renormalizes to guarantee the non-overlapping invariant.
fn from_limb_array<const N: usize>(arr: &[f64]) -> MultiFloat<N> {
    let mut limbs = [0.0f64; N];
    let n = arr.len().min(N);
    limbs[..n].copy_from_slice(&arr[..n]);
    MultiFloat::from_limbs(limbs)
}

/// Epsilon for convergence: roughly 2^(-52*N).
/// Uses f64::EPSILON = 2^-52, so epsilon::<4>() = 2^-208 ~ 2.4e-63.
fn epsilon<const N: usize>() -> f64 {
    let mut eps = 1.0_f64;
    for _ in 0..N {
        eps *= f64::EPSILON;
    }
    eps
}

/// Check if a MultiFloat term is negligible (below epsilon threshold).
/// Examines limbs[0] first (fast path), but also handles the edge case
/// where limbs[0] is zero and precision lives in lower limbs.
#[inline]
fn is_negligible<const N: usize>(term: &MultiFloat<N>, eps: f64) -> bool {
    let mag = term.limbs[0].abs();
    if mag > eps {
        return false;
    }
    // If leading limb is exactly zero, check all limbs
    if mag == 0.0 {
        for i in 1..N {
            if term.limbs[i].abs() > eps {
                return false;
            }
        }
    }
    true
}

/// Maximum Taylor terms for the given precision level.
fn max_taylor_terms<const N: usize>() -> usize {
    N * 20
}

/// Number of Newton iterations for ln: ceil(log2(N)) + 2.
fn newton_iters<const N: usize>() -> usize {
    let mut k = N;
    let mut bits = 0;
    while k > 1 {
        k = (k + 1) / 2;
        bits += 1;
    }
    bits + 3
}

/// Compute arctanh(x) = x + x^3/3 + x^5/5 + ... for |x| < 1.
/// Used to bootstrap ln(2) without depending on exp.
fn arctanh_series<const N: usize>(x: MultiFloat<N>) -> MultiFloat<N> {
    let x2 = x * x;
    let mut power = x; // x^(2k+1)
    let mut sum = x;
    let eps = epsilon::<N>();
    let max = max_taylor_terms::<N>() * 2;

    for i in 1..=max {
        power = power * x2;
        let denom = (2 * i + 1) as f64;
        let term = power / MultiFloat::<N>::from_f64(denom);
        sum = sum + term;
        if is_negligible(&term, eps) {
            break;
        }
    }
    sum
}

/// Compute arctan(x) for |x| < 1 via Taylor series:
/// arctan(x) = x - x^3/3 + x^5/5 - x^7/7 + ...
fn arctan_series<const N: usize>(x: MultiFloat<N>) -> MultiFloat<N> {
    let x2 = x * x;
    let mut power = x;
    let mut sum = x;
    let eps = epsilon::<N>();
    let max = max_taylor_terms::<N>() * 2;

    for i in 1..=max {
        power = power * x2;
        let denom = (2 * i + 1) as f64;
        let term = power / MultiFloat::<N>::from_f64(denom);
        if i % 2 == 1 {
            sum = sum - term;
        } else {
            sum = sum + term;
        }
        if is_negligible(&term, eps) {
            break;
        }
    }
    sum
}

/// Compute ln(2) at full N-limb precision using the identity:
/// ln(2) = 2 * arctanh(1/3) = 2 * (1/3 + 1/(3*3^2) + 1/(5*3^4) + ...)
/// This avoids any dependency on exp, breaking the circular dependency.
fn compute_ln2<const N: usize>() -> MultiFloat<N> {
    let third = MultiFloat::<N>::one() / MultiFloat::<N>::from_f64(3.0);
    let two = MultiFloat::<N>::from_f64(2.0);
    two * arctanh_series(third)
}

impl<const N: usize> MultiFloat<N> {
    /// Natural logarithm of 2 at full MultiFloat<N> precision.
    ///
    /// For N <= 4, returns hardcoded quad-double limbs.
    /// For N > 4, computes via arctanh series (no dependency on exp).
    pub fn ln2() -> Self {
        if N <= 4 {
            from_limb_array(&LN2_LIMBS_4)
        } else {
            compute_ln2::<N>()
        }
    }

    /// Pi at full MultiFloat<N> precision.
    ///
    /// For N <= 4, returns hardcoded quad-double limbs.
    /// For N > 4, uses Machin's formula: pi/4 = 4*arctan(1/5) - arctan(1/239).
    pub fn pi() -> Self {
        if N <= 4 {
            from_limb_array(&PI_LIMBS_4)
        } else {
            let four = Self::from_f64(4.0);
            let a = Self::one() / Self::from_f64(5.0);
            let b = Self::one() / Self::from_f64(239.0);
            four * (four * arctan_series(a) - arctan_series(b))
        }
    }

    /// Euler's number e at full MultiFloat<N> precision.
    ///
    /// For N <= 4, returns hardcoded quad-double limbs.
    /// For N > 4, computes exp(1).
    pub fn e() -> Self {
        if N <= 4 {
            from_limb_array(&E_LIMBS_4)
        } else {
            Self::one().exp()
        }
    }

    /// Compute e^self via argument reduction and Taylor series.
    ///
    /// Reduces self = k*ln(2) + r where |r| < ln(2)/2, then computes
    /// exp(r) = 1 + r + r^2/2! + r^3/3! + ... and scales by 2^k.
    pub fn exp(self) -> Self {
        if self.is_nan() {
            return self;
        }
        if self.is_zero() {
            return Self::one();
        }
        if !self.limbs[0].is_finite() {
            return if self.limbs[0] > 0.0 {
                Self::from_f64(f64::INFINITY)
            } else {
                Self::ZERO
            };
        }
        if self.limbs[0] > 709.0 {
            return Self::from_f64(f64::INFINITY);
        }
        if self.limbs[0] < -745.0 {
            return Self::ZERO;
        }

        let ln2 = Self::ln2();
        let k = (self.limbs[0] / ln2.limbs[0]).round();
        let r = self - Self::from_f64(k) * ln2;

        // Secondary argument reduction: compute exp(r/2^m) then square m times.
        // This reduces |r| further, improving Taylor convergence and accuracy.
        // Choose m so that |r/2^m| < ~0.01 for fast convergence.
        let m = if N <= 2 { 3 } else { (N as u32) * 2 };
        let scale_down = (1u64 << m) as f64;
        let rs = r / Self::from_f64(scale_down);

        // Taylor series for exp(rs)
        let mut term = Self::one();
        let mut sum = Self::one();
        let eps = epsilon::<N>();
        let max_terms = max_taylor_terms::<N>();

        for i in 1..=max_terms {
            term = term * rs / Self::from_f64(i as f64);
            sum = sum + term;
            if is_negligible(&term, eps) {
                break;
            }
        }

        // Square m times: exp(r) = exp(rs)^(2^m)
        for _ in 0..m {
            sum = sum * sum;
        }

        // Scale by 2^k
        let ki = k as i64;
        if ki >= -1022 && ki <= 1023 {
            let scale = f64::from_bits(((1023_i64 + ki) as u64) << 52);
            sum * Self::from_f64(scale)
        } else {
            // Split scaling to avoid overflow in the bit trick
            let half = ki / 2;
            let other = ki - half;
            let s1 = f64::from_bits(((1023_i64 + half) as u64) << 52);
            let s2 = f64::from_bits(((1023_i64 + other) as u64) << 52);
            sum * Self::from_f64(s1) * Self::from_f64(s2)
        }
    }

    /// Natural logarithm via Newton's method.
    ///
    /// Seed: x0 = f64::ln(self.limbs[0])
    /// Iteration: x_{k+1} = x_k + (self * exp(-x_k) - 1)
    /// Converges quadratically, needing ceil(log2(N)) + 2 steps.
    pub fn ln(self) -> Self {
        if self.is_nan() || self.limbs[0] < 0.0 {
            return Self::from_f64(f64::NAN);
        }
        if self.is_zero() {
            return Self::from_f64(f64::NEG_INFINITY);
        }
        if self.limbs[0] == f64::INFINITY {
            return Self::from_f64(f64::INFINITY);
        }
        // ln(1) = 0 exactly
        if self.limbs[0] == 1.0 {
            let mut is_one = true;
            for i in 1..N {
                if self.limbs[i] != 0.0 {
                    is_one = false;
                    break;
                }
            }
            if is_one {
                return Self::ZERO;
            }
        }

        let mut x = Self::from_f64(self.limbs[0].ln());
        let iters = newton_iters::<N>();
        for _ in 0..iters {
            let e = (Self::ZERO - x).exp();
            x = x + (self * e - Self::one());
        }
        x
    }

    /// Compute self^exp = exp(exp * ln(self)).
    ///
    /// Special cases:
    /// - x^0 = 1
    /// - 1^y = 1
    /// - 0^(positive) = 0, 0^(negative) = inf
    /// - (negative)^(non-integer) = NaN
    /// - (negative)^(integer) computed via |self|^exp with sign correction
    pub fn pow(self, exp: Self) -> Self {
        if exp.is_zero() {
            return Self::one();
        }
        // 1^y = 1
        if self.limbs[0] == 1.0 {
            let mut is_one = true;
            for i in 1..N {
                if self.limbs[i] != 0.0 {
                    is_one = false;
                    break;
                }
            }
            if is_one {
                return Self::one();
            }
        }
        if self.is_zero() {
            return if exp.limbs[0] > 0.0 {
                Self::ZERO
            } else {
                Self::from_f64(f64::INFINITY)
            };
        }
        if self.limbs[0] < 0.0 {
            let exp_f = exp.limbs[0];
            if exp_f == exp_f.round() && exp_f.abs() < (1u64 << 52) as f64 {
                let n = exp_f as i64;
                let result = (exp * self.abs().ln()).exp();
                return if n % 2 != 0 {
                    Self::ZERO - result
                } else {
                    result
                };
            }
            return Self::from_f64(f64::NAN);
        }
        (exp * self.ln()).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type QD = MultiFloat<4>;

    fn approx_eq(a: QD, b: QD, tol: f64) -> bool {
        (a - b).abs().limbs[0] < tol
    }

    #[test]
    fn exp_zero_is_one() {
        assert_eq!(QD::ZERO.exp().limbs[0], 1.0);
    }

    #[test]
    fn exp_one_is_e() {
        let e = QD::one().exp();
        let expected = std::f64::consts::E;
        assert!(
            (e.limbs[0] - expected).abs() < 1e-15,
            "exp(1) = {:e}",
            e.limbs[0]
        );
    }

    #[test]
    fn exp_neg_one_times_exp_one() {
        let e = QD::one().exp();
        let inv = (QD::ZERO - QD::one()).exp();
        let product = e * inv;
        assert!(
            approx_eq(product, QD::one(), 1e-50),
            "e * e^-1 error: {:e}",
            (product - QD::one()).limbs[0]
        );
    }

    #[test]
    fn exp_overflow() {
        let result = QD::from_f64(710.0).exp();
        assert!(result.limbs[0].is_infinite());
    }

    #[test]
    fn exp_underflow() {
        let result = QD::from_f64(-750.0).exp();
        assert_eq!(result.limbs[0], 0.0);
    }

    #[test]
    fn exp_nan() {
        assert!(QD::from_f64(f64::NAN).exp().is_nan());
    }

    #[test]
    fn ln_one_is_zero() {
        let result = QD::one().ln();
        assert_eq!(result.limbs[0], 0.0);
    }

    #[test]
    fn ln_e_is_one() {
        let e = QD::e();
        let result = e.ln();
        assert!(
            approx_eq(result, QD::one(), 1e-50),
            "ln(e) error: {:e}",
            (result - QD::one()).limbs[0]
        );
    }

    #[test]
    fn ln_negative_is_nan() {
        assert!(QD::from_f64(-1.0).ln().is_nan());
    }

    #[test]
    fn ln_zero_is_neg_inf() {
        let result = QD::ZERO.ln();
        assert!(result.limbs[0].is_infinite() && result.limbs[0] < 0.0);
    }

    #[test]
    fn exp_ln_roundtrip() {
        let x = QD::from_f64(3.14159265358979);
        let result = x.ln().exp();
        assert!(
            approx_eq(result, x, 1e-48),
            "exp(ln(pi)) error: {:e}",
            (result - x).limbs[0]
        );
    }

    #[test]
    fn ln_exp_roundtrip() {
        let x = QD::from_f64(2.5);
        let result = x.exp().ln();
        assert!(
            approx_eq(result, x, 1e-48),
            "ln(exp(2.5)) error: {:e}",
            (result - x).limbs[0]
        );
    }

    #[test]
    fn ln2_accuracy() {
        let ln2 = QD::ln2();
        let computed = QD::from_f64(2.0).ln();
        assert!(
            approx_eq(ln2, computed, 1e-50),
            "ln2 constant vs computed: {:e}",
            (ln2 - computed).limbs[0]
        );
    }

    #[test]
    fn pow_integer() {
        let two = QD::from_f64(2.0);
        let ten = QD::from_f64(10.0);
        let result = two.pow(ten);
        assert!(
            approx_eq(result, QD::from_f64(1024.0), 1e-40),
            "2^10 = {:e}",
            result.limbs[0]
        );
    }

    #[test]
    fn pow_zero_exp() {
        let result = QD::from_f64(42.0).pow(QD::ZERO);
        assert_eq!(result.limbs[0], 1.0);
    }

    #[test]
    fn pow_one_base() {
        let result = QD::one().pow(QD::from_f64(999.0));
        assert_eq!(result.limbs[0], 1.0);
    }

    #[test]
    fn pow_negative_base_even() {
        let result = QD::from_f64(-2.0).pow(QD::from_f64(4.0));
        assert!(
            approx_eq(result, QD::from_f64(16.0), 1e-40),
            "(-2)^4 = {:e}",
            result.limbs[0]
        );
    }

    #[test]
    fn pow_negative_base_odd() {
        let result = QD::from_f64(-2.0).pow(QD::from_f64(3.0));
        assert!(
            approx_eq(result, QD::from_f64(-8.0), 1e-40),
            "(-2)^3 = {:e}",
            result.limbs[0]
        );
    }

    #[test]
    fn pow_negative_base_noninteger_is_nan() {
        let result = QD::from_f64(-2.0).pow(QD::from_f64(0.5));
        assert!(result.limbs[0].is_nan());
    }

    #[test]
    fn pow_zero_base_positive_exp() {
        let result = QD::ZERO.pow(QD::from_f64(5.0));
        assert_eq!(result.limbs[0], 0.0);
    }

    #[test]
    fn pi_accuracy() {
        let pi = QD::pi();
        assert!(
            (pi.limbs[0] - std::f64::consts::PI).abs() < 1e-15,
            "pi.limbs[0] = {:e}",
            pi.limbs[0]
        );
    }

    #[test]
    fn e_accuracy() {
        let e = QD::e();
        assert!(
            (e.limbs[0] - std::f64::consts::E).abs() < 1e-15,
            "e.limbs[0] = {:e}",
            e.limbs[0]
        );
    }
}
