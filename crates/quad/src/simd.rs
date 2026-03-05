use crate::f128;

// ---------------------------------------------------------------------------
// NEON-accelerated batch double-double operations (aarch64)
// ---------------------------------------------------------------------------

// ---- Scalar fallback helpers ----

#[inline(always)]
fn scalar_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

#[inline(always)]
fn scalar_quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

#[inline(always)]
fn scalar_two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

#[inline(always)]
fn scalar_dd_add(a_hi: f64, a_lo: f64, b_hi: f64, b_lo: f64) -> (f64, f64) {
    let (s1, e1) = scalar_two_sum(a_hi, b_hi);
    let e = e1 + (a_lo + b_lo);
    scalar_quick_two_sum(s1, e)
}

#[inline(always)]
fn scalar_dd_mul(a_hi: f64, a_lo: f64, b_hi: f64, b_lo: f64) -> (f64, f64) {
    let (p1, e1) = scalar_two_prod(a_hi, b_hi);
    let e = e1 + (a_hi * b_lo + a_lo * b_hi);
    scalar_quick_two_sum(p1, e)
}

// ---- NEON implementation ----

#[cfg(target_arch = "aarch64")]
mod neon {
    #![allow(unsafe_op_in_unsafe_fn)]
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn two_sum_x2(a: float64x2_t, b: float64x2_t) -> (float64x2_t, float64x2_t) {
        let s = vaddq_f64(a, b);
        let v = vsubq_f64(s, a);
        let e = vaddq_f64(
            vsubq_f64(a, vsubq_f64(s, v)),
            vsubq_f64(b, v),
        );
        (s, e)
    }

    #[inline(always)]
    unsafe fn quick_two_sum_x2(a: float64x2_t, b: float64x2_t) -> (float64x2_t, float64x2_t) {
        let s = vaddq_f64(a, b);
        let e = vsubq_f64(b, vsubq_f64(s, a));
        (s, e)
    }

    #[inline(always)]
    unsafe fn two_prod_x2(a: float64x2_t, b: float64x2_t) -> (float64x2_t, float64x2_t) {
        let p = vmulq_f64(a, b);
        let neg_p = vnegq_f64(p);
        let e = vfmaq_f64(neg_p, a, b);
        (p, e)
    }

    #[inline(always)]
    unsafe fn dd_add_x2(
        a_hi: float64x2_t, a_lo: float64x2_t,
        b_hi: float64x2_t, b_lo: float64x2_t,
    ) -> (float64x2_t, float64x2_t) {
        let (s1, e1) = two_sum_x2(a_hi, b_hi);
        let e = vaddq_f64(e1, vaddq_f64(a_lo, b_lo));
        quick_two_sum_x2(s1, e)
    }

    #[inline(always)]
    unsafe fn dd_mul_x2(
        a_hi: float64x2_t, a_lo: float64x2_t,
        b_hi: float64x2_t, b_lo: float64x2_t,
    ) -> (float64x2_t, float64x2_t) {
        let (p1, e1) = two_prod_x2(a_hi, b_hi);
        let cross = vaddq_f64(vmulq_f64(a_hi, b_lo), vmulq_f64(a_lo, b_hi));
        let e = vaddq_f64(e1, cross);
        quick_two_sum_x2(p1, e)
    }

    #[inline]
    pub(super) unsafe fn dd_add_batch_neon(
        a_hi: &[f64], a_lo: &[f64],
        b_hi: &[f64], b_lo: &[f64],
        out_hi: &mut [f64], out_lo: &mut [f64],
    ) {
        let n = a_hi.len();
        let pairs = n / 2;
        for i in 0..pairs {
            let off = i * 2;
            let ah = vld1q_f64(a_hi.as_ptr().add(off));
            let al = vld1q_f64(a_lo.as_ptr().add(off));
            let bh = vld1q_f64(b_hi.as_ptr().add(off));
            let bl = vld1q_f64(b_lo.as_ptr().add(off));
            let (rh, rl) = dd_add_x2(ah, al, bh, bl);
            vst1q_f64(out_hi.as_mut_ptr().add(off), rh);
            vst1q_f64(out_lo.as_mut_ptr().add(off), rl);
        }
    }

    #[inline]
    pub(super) unsafe fn dd_mul_batch_neon(
        a_hi: &[f64], a_lo: &[f64],
        b_hi: &[f64], b_lo: &[f64],
        out_hi: &mut [f64], out_lo: &mut [f64],
    ) {
        let n = a_hi.len();
        let pairs = n / 2;
        for i in 0..pairs {
            let off = i * 2;
            let ah = vld1q_f64(a_hi.as_ptr().add(off));
            let al = vld1q_f64(a_lo.as_ptr().add(off));
            let bh = vld1q_f64(b_hi.as_ptr().add(off));
            let bl = vld1q_f64(b_lo.as_ptr().add(off));
            let (rh, rl) = dd_mul_x2(ah, al, bh, bl);
            vst1q_f64(out_hi.as_mut_ptr().add(off), rh);
            vst1q_f64(out_lo.as_mut_ptr().add(off), rl);
        }
    }

    #[inline]
    pub(super) unsafe fn dd_dot_neon(
        a_hi: &[f64], a_lo: &[f64],
        b_hi: &[f64], b_lo: &[f64],
    ) -> (f64, f64) {
        let n = a_hi.len();
        let pairs = n / 2;
        let mut acc_hi = vdupq_n_f64(0.0);
        let mut acc_lo = vdupq_n_f64(0.0);
        for i in 0..pairs {
            let off = i * 2;
            let ah = vld1q_f64(a_hi.as_ptr().add(off));
            let al = vld1q_f64(a_lo.as_ptr().add(off));
            let bh = vld1q_f64(b_hi.as_ptr().add(off));
            let bl = vld1q_f64(b_lo.as_ptr().add(off));
            let (ph, pl) = dd_mul_x2(ah, al, bh, bl);
            let (sh, sl) = dd_add_x2(acc_hi, acc_lo, ph, pl);
            acc_hi = sh;
            acc_lo = sl;
        }
        let mut buf_hi = [0.0f64; 2];
        let mut buf_lo = [0.0f64; 2];
        vst1q_f64(buf_hi.as_mut_ptr(), acc_hi);
        vst1q_f64(buf_lo.as_mut_ptr(), acc_lo);
        super::scalar_dd_add(buf_hi[0], buf_lo[0], buf_hi[1], buf_lo[1])
    }
}

// ---- Public API ----

/// Batch double-double addition: out[i] = a[i] + b[i].
///
/// All slices must have the same length. Panics otherwise.
pub fn dd_add_batch(
    a_hi: &[f64], a_lo: &[f64],
    b_hi: &[f64], b_lo: &[f64],
    out_hi: &mut [f64], out_lo: &mut [f64],
) {
    let n = a_hi.len();
    assert!(n == a_lo.len() && n == b_hi.len() && n == b_lo.len()
        && n == out_hi.len() && n == out_lo.len(),
        "dd_add_batch: all slices must have the same length");

    let done;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::dd_add_batch_neon(a_hi, a_lo, b_hi, b_lo, out_hi, out_lo); }
        done = (n / 2) * 2;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        done = 0;
    }

    // Scalar remainder
    for i in done..n {
        let (rh, rl) = scalar_dd_add(a_hi[i], a_lo[i], b_hi[i], b_lo[i]);
        out_hi[i] = rh;
        out_lo[i] = rl;
    }
}

/// Batch double-double multiplication: out[i] = a[i] * b[i].
///
/// All slices must have the same length. Panics otherwise.
pub fn dd_mul_batch(
    a_hi: &[f64], a_lo: &[f64],
    b_hi: &[f64], b_lo: &[f64],
    out_hi: &mut [f64], out_lo: &mut [f64],
) {
    let n = a_hi.len();
    assert!(n == a_lo.len() && n == b_hi.len() && n == b_lo.len()
        && n == out_hi.len() && n == out_lo.len(),
        "dd_mul_batch: all slices must have the same length");

    let done;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::dd_mul_batch_neon(a_hi, a_lo, b_hi, b_lo, out_hi, out_lo); }
        done = (n / 2) * 2;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        done = 0;
    }

    // Scalar remainder
    for i in done..n {
        let (rh, rl) = scalar_dd_mul(a_hi[i], a_lo[i], b_hi[i], b_lo[i]);
        out_hi[i] = rh;
        out_lo[i] = rl;
    }
}

/// Dot product of two double-double arrays, returning an f128.
pub fn dd_dot(
    a_hi: &[f64], a_lo: &[f64],
    b_hi: &[f64], b_lo: &[f64],
) -> f128 {
    let n = a_hi.len();
    assert!(n == a_lo.len() && n == b_hi.len() && n == b_lo.len(),
        "dd_dot: all slices must have the same length");

    let (mut sum_hi, mut sum_lo);

    #[cfg(target_arch = "aarch64")]
    {
        let (sh, sl) = unsafe { neon::dd_dot_neon(a_hi, a_lo, b_hi, b_lo) };
        sum_hi = sh;
        sum_lo = sl;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        sum_hi = 0.0;
        sum_lo = 0.0;
    }

    // Scalar remainder (odd element)
    let done;
    #[cfg(target_arch = "aarch64")]
    { done = (n / 2) * 2; }
    #[cfg(not(target_arch = "aarch64"))]
    { done = 0; }

    for i in done..n {
        let (ph, pl) = scalar_dd_mul(a_hi[i], a_lo[i], b_hi[i], b_lo[i]);
        let (sh, sl) = scalar_dd_add(sum_hi, sum_lo, ph, pl);
        sum_hi = sh;
        sum_lo = sl;
    }

    f128::new(sum_hi, sum_lo)
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_dd_add(a: f128, b: f128) -> f128 { a + b }
    fn ref_dd_mul(a: f128, b: f128) -> f128 { a * b }

    #[test]
    fn test_dd_add_batch_even() {
        let a: Vec<f128> = (0..8).map(|i| f128::from_f64(1.0 + i as f64 * 0.1)).collect();
        let b: Vec<f128> = (0..8).map(|i| f128::from_f64(2.0 + i as f64 * 0.3)).collect();
        let a_hi: Vec<f64> = a.iter().map(|x| x.hi).collect();
        let a_lo: Vec<f64> = a.iter().map(|x| x.lo).collect();
        let b_hi: Vec<f64> = b.iter().map(|x| x.hi).collect();
        let b_lo: Vec<f64> = b.iter().map(|x| x.lo).collect();
        let mut out_hi = vec![0.0; 8];
        let mut out_lo = vec![0.0; 8];

        dd_add_batch(&a_hi, &a_lo, &b_hi, &b_lo, &mut out_hi, &mut out_lo);

        for i in 0..8 {
            let expected = ref_dd_add(a[i], b[i]);
            assert_eq!(out_hi[i], expected.hi, "add hi mismatch at {i}");
            assert_eq!(out_lo[i], expected.lo, "add lo mismatch at {i}");
        }
    }

    #[test]
    fn test_dd_add_batch_odd() {
        let a: Vec<f128> = (0..7).map(|i| f128::from_f64(1.0 + i as f64 * 0.1)).collect();
        let b: Vec<f128> = (0..7).map(|i| f128::from_f64(2.0 + i as f64 * 0.3)).collect();
        let a_hi: Vec<f64> = a.iter().map(|x| x.hi).collect();
        let a_lo: Vec<f64> = a.iter().map(|x| x.lo).collect();
        let b_hi: Vec<f64> = b.iter().map(|x| x.hi).collect();
        let b_lo: Vec<f64> = b.iter().map(|x| x.lo).collect();
        let mut out_hi = vec![0.0; 7];
        let mut out_lo = vec![0.0; 7];

        dd_add_batch(&a_hi, &a_lo, &b_hi, &b_lo, &mut out_hi, &mut out_lo);

        for i in 0..7 {
            let expected = ref_dd_add(a[i], b[i]);
            assert_eq!(out_hi[i], expected.hi, "add hi mismatch at {i}");
            assert_eq!(out_lo[i], expected.lo, "add lo mismatch at {i}");
        }
    }

    #[test]
    fn test_dd_mul_batch_even() {
        let a: Vec<f128> = (0..8).map(|i| f128::from_f64(1.0 + i as f64 * 0.1)).collect();
        let b: Vec<f128> = (0..8).map(|i| f128::from_f64(2.0 + i as f64 * 0.3)).collect();
        let a_hi: Vec<f64> = a.iter().map(|x| x.hi).collect();
        let a_lo: Vec<f64> = a.iter().map(|x| x.lo).collect();
        let b_hi: Vec<f64> = b.iter().map(|x| x.hi).collect();
        let b_lo: Vec<f64> = b.iter().map(|x| x.lo).collect();
        let mut out_hi = vec![0.0; 8];
        let mut out_lo = vec![0.0; 8];

        dd_mul_batch(&a_hi, &a_lo, &b_hi, &b_lo, &mut out_hi, &mut out_lo);

        for i in 0..8 {
            let expected = ref_dd_mul(a[i], b[i]);
            assert_eq!(out_hi[i], expected.hi, "mul hi mismatch at {i}");
            assert_eq!(out_lo[i], expected.lo, "mul lo mismatch at {i}");
        }
    }

    #[test]
    fn test_dd_mul_batch_odd() {
        let a: Vec<f128> = (0..5).map(|i| f128::from_f64(1.0 + i as f64 * 0.7)).collect();
        let b: Vec<f128> = (0..5).map(|i| f128::from_f64(0.5 + i as f64 * 0.2)).collect();
        let a_hi: Vec<f64> = a.iter().map(|x| x.hi).collect();
        let a_lo: Vec<f64> = a.iter().map(|x| x.lo).collect();
        let b_hi: Vec<f64> = b.iter().map(|x| x.hi).collect();
        let b_lo: Vec<f64> = b.iter().map(|x| x.lo).collect();
        let mut out_hi = vec![0.0; 5];
        let mut out_lo = vec![0.0; 5];

        dd_mul_batch(&a_hi, &a_lo, &b_hi, &b_lo, &mut out_hi, &mut out_lo);

        for i in 0..5 {
            let expected = ref_dd_mul(a[i], b[i]);
            assert_eq!(out_hi[i], expected.hi, "mul hi mismatch at {i}");
            assert_eq!(out_lo[i], expected.lo, "mul lo mismatch at {i}");
        }
    }

    #[test]
    fn test_dd_dot() {
        let a: Vec<f128> = (0..10).map(|i| f128::from_f64(1.0 + i as f64 * 0.1)).collect();
        let b: Vec<f128> = (0..10).map(|i| f128::from_f64(2.0 + i as f64 * 0.3)).collect();
        let a_hi: Vec<f64> = a.iter().map(|x| x.hi).collect();
        let a_lo: Vec<f64> = a.iter().map(|x| x.lo).collect();
        let b_hi: Vec<f64> = b.iter().map(|x| x.hi).collect();
        let b_lo: Vec<f64> = b.iter().map(|x| x.lo).collect();

        let result = dd_dot(&a_hi, &a_lo, &b_hi, &b_lo);

        // Reference: scalar accumulation
        let mut expected = f128::ZERO;
        for i in 0..10 {
            expected = expected + a[i] * b[i];
        }

        // hi parts must match exactly; lo parts may differ by a few ulps
        // due to different accumulation order (SIMD reduces lane pairs then
        // horizontally, scalar accumulates sequentially).
        assert_eq!(result.hi, expected.hi, "dot hi mismatch");
        let lo_err = (result.lo - expected.lo).abs();
        assert!(lo_err < 1e-28,
            "dot lo mismatch: got {}, expected {}, diff {lo_err:e}",
            result.lo, expected.lo);
    }

    #[test]
    fn test_dd_dot_odd() {
        let a: Vec<f128> = (0..11).map(|i| f128::from_f64(0.5 + i as f64)).collect();
        let b: Vec<f128> = (0..11).map(|i| f128::from_f64(1.0 - i as f64 * 0.05)).collect();
        let a_hi: Vec<f64> = a.iter().map(|x| x.hi).collect();
        let a_lo: Vec<f64> = a.iter().map(|x| x.lo).collect();
        let b_hi: Vec<f64> = b.iter().map(|x| x.hi).collect();
        let b_lo: Vec<f64> = b.iter().map(|x| x.lo).collect();

        let result = dd_dot(&a_hi, &a_lo, &b_hi, &b_lo);

        let mut expected = f128::ZERO;
        for i in 0..11 {
            expected = expected + a[i] * b[i];
        }

        assert_eq!(result.hi, expected.hi, "dot hi mismatch (odd)");
        let lo_err = (result.lo - expected.lo).abs();
        assert!(lo_err < 1e-28,
            "dot lo mismatch (odd): got {}, expected {}, diff {lo_err:e}",
            result.lo, expected.lo);
    }

    #[test]
    fn test_dd_dot_empty() {
        let result = dd_dot(&[], &[], &[], &[]);
        assert_eq!(result.hi, 0.0);
        assert_eq!(result.lo, 0.0);
    }

    #[test]
    fn test_dd_dot_single() {
        let a = f128::from_f64(3.0);
        let b = f128::from_f64(7.0);
        let result = dd_dot(&[a.hi], &[a.lo], &[b.hi], &[b.lo]);
        let expected = a * b;
        assert_eq!(result.hi, expected.hi);
        assert_eq!(result.lo, expected.lo);
    }

    #[test]
    fn test_dd_add_batch_with_lo_parts() {
        // Use values that have nonzero lo parts
        let a = [f128::new(1.0, 1e-20), f128::new(2.0, -3e-18), f128::new(0.5, 1e-25)];
        let b = [f128::new(3.0, 2e-19), f128::new(1.0, 5e-17), f128::new(0.25, -1e-24)];
        let a_hi: Vec<f64> = a.iter().map(|x| x.hi).collect();
        let a_lo: Vec<f64> = a.iter().map(|x| x.lo).collect();
        let b_hi: Vec<f64> = b.iter().map(|x| x.hi).collect();
        let b_lo: Vec<f64> = b.iter().map(|x| x.lo).collect();
        let mut out_hi = vec![0.0; 3];
        let mut out_lo = vec![0.0; 3];

        dd_add_batch(&a_hi, &a_lo, &b_hi, &b_lo, &mut out_hi, &mut out_lo);

        for i in 0..3 {
            let expected = a[i] + b[i];
            assert_eq!(out_hi[i], expected.hi, "add hi mismatch at {i}");
            assert_eq!(out_lo[i], expected.lo, "add lo mismatch at {i}");
        }
    }
}
