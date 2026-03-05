/// MultiFloat<N>: an N-component non-overlapping expansion of f64 values.
/// limbs[0] is most significant; each subsequent limb captures error from above.

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct MultiFloat<const N: usize> {
    pub limbs: [f64; N], // limbs[0] most significant, non-overlapping
}

#[allow(non_camel_case_types)]
pub type f256 = MultiFloat<4>;
#[allow(non_camel_case_types)]
pub type f512 = MultiFloat<8>;

// --- Error-free primitives (Dekker/Knuth) ---

/// Error-free addition: returns (s, e) with a + b = s + e exactly, |e| <= ulp(s)/2.
/// No constraint on relative magnitude of a, b.
#[inline(always)]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Fast error-free addition when |a| >= |b|.
/// Returns (s, e) with a + b = s + e exactly.
#[inline(always)]
pub(crate) fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// Error-free multiplication using FMA: returns (p, e) with a * b = p + e exactly.
#[inline(always)]
pub(crate) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

// --- Renormalization ---

/// In-place renormalization of N limbs into non-overlapping form.
///
/// After this, |limbs[i+1]| <= ulp(limbs[i]) / 2 for all valid i.
///
/// Algorithm from the QD library (qd_real::renorm):
///   Phase 1 (bottom-up): sweep from the least-significant pair upward
///     using two_sum (not quick_two_sum, since magnitudes may not be ordered).
///     This produces an intermediate array where each pair is error-free-summed.
///   Phase 2 (top-down): sweep from most-significant downward using
///     quick_two_sum (now magnitudes ARE ordered after phase 1). Non-zero
///     error terms are emitted as output limbs.
#[inline]
pub(crate) fn renormalize<const N: usize>(limbs: &mut [f64; N]) {
    if N <= 1 {
        return;
    }

    // Run two full passes of bottom-up / top-down to guarantee the
    // non-overlapping invariant even for heavily overlapping inputs.
    // The first pass sorts magnitudes; the second locks in the invariant.
    for _ in 0..2 {
        // Bottom-up: two_sum from least-significant upward.
        let (mut s, e) = two_sum(limbs[N - 2], limbs[N - 1]);
        limbs[N - 1] = e;
        for i in (0..N - 2).rev() {
            let (new_s, e) = two_sum(limbs[i], s);
            s = new_s;
            limbs[i + 1] = e;
        }
        limbs[0] = s;

        // Top-down: quick_two_sum, emitting non-zero errors as output limbs.
        let mut out = [0.0f64; N];
        s = limbs[0];
        let mut k = 0usize;
        for i in 1..N {
            let (new_s, e) = quick_two_sum(s, limbs[i]);
            s = new_s;
            if e != 0.0 {
                out[k] = s;
                s = e;
                k += 1;
            }
        }
        out[k] = s;
        *limbs = out;
    }
}

/// Reduce M input terms to N non-overlapping output limbs (M > N).
///
/// First performs a full bottom-up accumulation on all M terms,
/// then a top-down sweep outputting at most N limbs.
#[inline]
pub(crate) fn renormalize_from<const N: usize, const M: usize>(input: &[f64; M]) -> [f64; N] {
    debug_assert!(M > N, "renormalize_from requires M > N");

    if M == 0 || N == 0 {
        return [0.0; N];
    }

    // First pass: bottom-up two_sum over all M terms, then top-down into M slots.
    let mut t = *input;
    {
        let (mut s, e) = two_sum(t[M - 2], t[M - 1]);
        t[M - 1] = e;
        for i in (0..M - 2).rev() {
            let (new_s, e) = two_sum(t[i], s);
            s = new_s;
            t[i + 1] = e;
        }
        t[0] = s;

        // Top-down compaction within the M-element array.
        let mut buf = [0.0f64; M];
        s = t[0];
        let mut k = 0usize;
        for i in 1..M {
            let (new_s, e) = quick_two_sum(s, t[i]);
            s = new_s;
            if e != 0.0 {
                buf[k] = s;
                s = e;
                k += 1;
            }
        }
        buf[k] = s;
        t = buf;
    }

    // Second pass: same procedure to lock in the invariant,
    // but this time output only N limbs.
    {
        let (mut s, e) = two_sum(t[M - 2], t[M - 1]);
        t[M - 1] = e;
        for i in (0..M - 2).rev() {
            let (new_s, e) = two_sum(t[i], s);
            s = new_s;
            t[i + 1] = e;
        }
        t[0] = s;

        let mut out = [0.0f64; N];
        s = t[0];
        let mut k = 0usize;
        for i in 1..M {
            let (new_s, e) = quick_two_sum(s, t[i]);
            s = new_s;
            if e != 0.0 {
                out[k] = s;
                s = e;
                k += 1;
                if k >= N {
                    return out;
                }
            }
        }
        if k < N {
            out[k] = s;
        }
        out
    }
}

// --- MultiFloat implementation ---

impl<const N: usize> MultiFloat<N> {
    pub const ZERO: Self = Self { limbs: [0.0; N] };

    // ONE must be a function because const generics can't do array initialization
    // with non-zero values in const context without nightly features.
    // We provide it as an associated function.
    pub fn one() -> Self {
        let mut limbs = [0.0f64; N];
        limbs[0] = 1.0;
        Self { limbs }
    }

    #[inline(always)]
    pub fn from_f64(x: f64) -> Self {
        let mut limbs = [0.0f64; N];
        limbs[0] = x;
        Self { limbs }
    }

    /// Best f64 approximation: sum of the two most significant limbs.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        if N >= 2 {
            self.limbs[0] + self.limbs[1]
        } else {
            self.limbs[0]
        }
    }

    #[inline(always)]
    pub fn from_limbs(limbs: [f64; N]) -> Self {
        let mut mf = Self { limbs };
        renormalize(&mut mf.limbs);
        mf
    }

    /// Construct from limbs without renormalization (caller guarantees non-overlapping).
    #[inline(always)]
    pub fn from_limbs_unchecked(limbs: [f64; N]) -> Self {
        Self { limbs }
    }

    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.limbs[0].is_nan()
    }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.limbs[0].is_finite()
    }

    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.limbs[0] == 0.0
    }

    /// Renormalize in place.
    #[inline]
    pub fn renorm(&mut self) {
        renormalize(&mut self.limbs);
    }
}

// Provide ONE as associated constants for the common sizes via specialization-like approach.
// Since we can't do `const ONE` generically with non-zero array init, we use a trait.

impl MultiFloat<2> {
    pub const ONE: Self = Self { limbs: [1.0, 0.0] };
}

impl MultiFloat<4> {
    pub const ONE: Self = Self { limbs: [1.0, 0.0, 0.0, 0.0] };
}

impl MultiFloat<8> {
    pub const ONE: Self = Self { limbs: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
}

// --- Debug ---

impl<const N: usize> std::fmt::Debug for MultiFloat<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiFloat<{}>([", N)?;
        for (i, &l) in self.limbs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:e}", l)?;
        }
        write!(f, "])")
    }
}

impl<const N: usize> PartialEq for MultiFloat<N> {
    fn eq(&self, other: &Self) -> bool {
        self.limbs == other.limbs
    }
}

impl<const N: usize> PartialOrd for MultiFloat<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        for i in 0..N {
            match self.limbs[i].partial_cmp(&other.limbs[i]) {
                Some(std::cmp::Ordering::Equal) => continue,
                ord => return ord,
            }
        }
        Some(std::cmp::Ordering::Equal)
    }
}

impl<const N: usize> Default for MultiFloat<N> {
    fn default() -> Self {
        Self::ZERO
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renorm_identity() {
        // Already normalized: should be unchanged.
        let mut limbs = [1.0, 1e-20, 1e-40, 0.0];
        let orig = limbs;
        renormalize(&mut limbs);
        for i in 0..4 {
            assert_eq!(limbs[i], orig[i], "limb {i} changed unexpectedly");
        }
    }

    #[test]
    fn renorm_overlapping() {
        // Two large values that overlap.
        let mut limbs = [1.0, 1.0, 0.0, 0.0];
        renormalize(&mut limbs);
        assert_eq!(limbs[0], 2.0);
        assert_eq!(limbs[1], 0.0);
        assert_eq!(limbs[2], 0.0);
        assert_eq!(limbs[3], 0.0);
    }

    #[test]
    fn renorm_from_larger() {
        // 8 terms -> 4 limbs
        let input = [1.0, 1e-17, 1e-34, 1e-51, 1e-68, 0.0, 0.0, 0.0];
        let out: [f64; 4] = renormalize_from::<4, 8>(&input);
        // The most significant limb should be 1.0
        assert_eq!(out[0], 1.0);
        // Second limb should capture the 1e-17 term
        assert!((out[1] - 1e-17).abs() < 1e-32);
    }

    #[test]
    fn renorm_non_overlapping_invariant() {
        // After renorm, |limbs[i+1]| <= ulp(limbs[i]) / 2
        let mut limbs = [3.14159, 2.71828, 1.41421, 1.73205];
        renormalize(&mut limbs);
        for i in 0..3 {
            if limbs[i] == 0.0 {
                continue;
            }
            let ulp_i = ulp(limbs[i]);
            assert!(
                limbs[i + 1].abs() <= ulp_i / 2.0 + 1e-300, // tiny epsilon for zero case
                "Non-overlapping invariant violated: |limbs[{}]| = {:e} > ulp(limbs[{}])/2 = {:e}",
                i + 1,
                limbs[i + 1].abs(),
                i,
                ulp_i / 2.0
            );
        }
    }

    #[test]
    fn from_f64_roundtrip() {
        let mf = MultiFloat::<4>::from_f64(std::f64::consts::PI);
        assert_eq!(mf.to_f64(), std::f64::consts::PI);
        assert_eq!(mf.limbs[1], 0.0);
    }

    #[test]
    fn is_predicates() {
        assert!(MultiFloat::<4>::ZERO.is_zero());
        assert!(!MultiFloat::<4>::ZERO.is_nan());
        assert!(MultiFloat::<4>::ZERO.is_finite());
        assert!(MultiFloat::<4>::from_f64(f64::NAN).is_nan());
        assert!(!MultiFloat::<4>::from_f64(f64::INFINITY).is_finite());
    }

    #[test]
    fn one_constant() {
        let one = MultiFloat::<4>::ONE;
        assert_eq!(one.to_f64(), 1.0);
        assert_eq!(one.limbs[0], 1.0);
        assert_eq!(one.limbs[1], 0.0);
    }

    /// Compute ulp of a finite f64 for testing.
    fn ulp(x: f64) -> f64 {
        if x == 0.0 {
            return f64::MIN_POSITIVE;
        }
        let x = x.abs();
        let bits = x.to_bits();
        let next = f64::from_bits(bits + 1);
        next - x
    }

    /// Helper: check the non-overlapping invariant on limbs.
    fn assert_non_overlapping(limbs: &[f64], label: &str) {
        for i in 0..limbs.len() - 1 {
            if limbs[i] == 0.0 {
                continue;
            }
            let ulp_i = ulp(limbs[i]);
            assert!(
                limbs[i + 1].abs() <= ulp_i / 2.0 + 1e-300,
                "{label}: non-overlapping violated at [{i}]->[{}]: |{:e}| > ulp({:e})/2 = {:e}",
                i + 1,
                limbs[i + 1],
                limbs[i],
                ulp_i / 2.0
            );
        }
    }

    #[test]
    fn renorm_badly_conditioned_equal_limbs() {
        // All limbs equal and large -- heavily overlapping.
        let mut limbs = [1e10, 1e10, 1e10, 1e10];
        renormalize(&mut limbs);
        // Sum must be preserved exactly: 4e10
        let total: f64 = limbs.iter().sum();
        assert_eq!(total, 4e10, "Sum not preserved: {total:e}");
        assert_eq!(limbs[0], 4e10);
        assert_eq!(limbs[1], 0.0);
        assert_eq!(limbs[2], 0.0);
        assert_eq!(limbs[3], 0.0);
        assert_non_overlapping(&limbs, "badly_conditioned");
    }

    #[test]
    fn renorm_badly_conditioned_mixed() {
        // Mixture of positive/negative overlapping terms
        let mut limbs = [1.0, 0.5, -0.25, 0.125];
        let orig_sum: f64 = limbs.iter().sum();
        renormalize(&mut limbs);
        let new_sum: f64 = limbs.iter().sum();
        assert!(
            (new_sum - orig_sum).abs() < 1e-15,
            "Sum changed: {orig_sum:e} -> {new_sum:e}"
        );
        assert_non_overlapping(&limbs, "mixed_conditioned");
    }

    #[test]
    fn mul_1_plus_tiny_squared_full_precision() {
        // (1 + 1e-20)^2 = 1 + 2e-20 + 1e-40
        // With MultiFloat<4>, all three "pieces" should be captured.
        let tiny = 1e-20;
        let a = MultiFloat::<4>::from_limbs([1.0, tiny, 0.0, 0.0]);

        // Verify the input represents 1 + 1e-20
        let a_sum: f64 = a.limbs.iter().sum();
        assert!((a_sum - (1.0 + tiny)).abs() < 1e-35);

        let c = a * a;

        // The exact value is 1 + 2e-20 + 1e-40.
        // Sum all limbs to get the full-precision value.
        // Since limbs[0] = 1.0 absorbs the leading part, we check:
        //   sum(limbs) - 1.0 ≈ 2e-20 + 1e-40
        let sum: f64 = c.limbs.iter().sum();
        let expected = 1.0 + 2.0 * tiny + tiny * tiny;

        // The error should be well below the 4th limb's precision (~212 bits, ~1e-64)
        assert!(
            (sum - expected).abs() < 1e-55,
            "Precision loss: sum={sum:e}, expected={expected:e}, diff={:e}",
            sum - expected
        );

        // Check that limbs[0] captured the leading 1.0
        assert_eq!(c.limbs[0], 1.0, "limbs[0] should be 1.0");

        // Check that limbs[1] captured the cross-term ~2e-20
        assert!(
            (c.limbs[1] - 2.0 * tiny).abs() < 1e-35,
            "limbs[1] should be ~2e-20, got {:e}",
            c.limbs[1]
        );

        // Check that limbs[2] captured the tiny^2 term ~1e-40
        assert!(
            (c.limbs[2] - tiny * tiny).abs() < 1e-55,
            "limbs[2] should be ~1e-40, got {:e}",
            c.limbs[2]
        );

        // Verify non-overlapping invariant
        assert_non_overlapping(&c.limbs, "mul_1_plus_tiny");
    }
}
