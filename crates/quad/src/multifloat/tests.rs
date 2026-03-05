#[cfg(test)]
mod tests {
    use crate::multifloat::*;

    // Approximate machine epsilons for each precision level.
    // f256 (4 limbs, ~212 mantissa bits): epsilon ~ 2^{-212} ~ 1e-63
    // f512 (8 limbs, ~424 mantissa bits): epsilon ~ 2^{-424} ~ 1e-127
    const F256_EPS: f64 = 1e-63;
    const F512_EPS: f64 = 1e-127;

    // Looser tolerances for operations that accumulate error (transcendentals, etc.)
    const F256_TOL: f64 = 1e-55;
    const F512_TOL: f64 = 1e-110;

    fn f256_err(a: f256, b: f256) -> f64 {
        (a - b).abs().limbs[0].abs()
    }

    fn f512_err(a: f512, b: f512) -> f64 {
        (a - b).abs().limbs[0].abs()
    }

    // ===================================================================
    // 1. Basic arithmetic precision
    // ===================================================================

    #[test]
    fn f256_add_sub_recovers_small() {
        // (1 + 1e-40) - 1 should recover 1e-40 (impossible in f64)
        let one = f256::from_f64(1.0);
        let tiny = f256::from_f64(1e-40);
        let result = (one + tiny) - one;
        let err = f256_err(result, tiny);
        assert!(err < F256_EPS, "(1+1e-40)-1 error in f256: {err:e}");
    }

    #[test]
    fn f512_add_sub_recovers_very_small() {
        // (1 + 1e-80) - 1 should recover 1e-80 (impossible in f64 or f256, possible in f512)
        let one = f512::from_f64(1.0);
        let tiny = f512::from_f64(1e-80);
        let result = (one + tiny) - one;
        let err = f512_err(result, tiny);
        assert!(err < F512_EPS, "(1+1e-80)-1 error in f512: {err:e}");
    }

    #[test]
    fn f256_one_third_roundtrip() {
        let one = f256::ONE;
        let three = f256::from_f64(3.0);
        let third = one / three;
        let result = third * three;
        let err = f256_err(result, one);
        assert!(err < F256_EPS, "f256: 1/3 * 3 error: {err:e}");
    }

    #[test]
    fn f512_one_third_roundtrip() {
        let one = f512::ONE;
        let three = f512::from_f64(3.0);
        let third = one / three;
        let result = third * three;
        let err = f512_err(result, one);
        assert!(err < F512_EPS, "f512: 1/3 * 3 error: {err:e}");
    }

    // ===================================================================
    // 2. Multiplication precision
    // ===================================================================

    #[test]
    fn f256_mul_one_plus_eps_squared() {
        // (1 + eps)^2 - 1 should be approximately 2*eps for tiny eps
        let eps = 1e-40_f64;
        let one = f256::ONE;
        let val = one + f256::from_f64(eps);
        let sq = val * val;
        let diff = sq - one;
        let expected = f256::from_f64(2.0 * eps);
        let err = f256_err(diff, expected);
        assert!(err < eps * eps * 10.0, "f256 (1+eps)^2 - 1 error: {err:e}");
    }

    #[test]
    fn f512_mul_one_plus_eps_squared() {
        let eps = 1e-80_f64;
        let one = f512::ONE;
        let val = one + f512::from_f64(eps);
        let sq = val * val;
        let diff = sq - one;
        let expected = f512::from_f64(2.0 * eps);
        let err = f512_err(diff, expected);
        assert!(err < eps * eps * 10.0, "f512 (1+eps)^2 - 1 error: {err:e}");
    }

    #[test]
    fn f256_large_times_small() {
        // Large * small: the inputs 1e30 and 1e-30 are NOT exact f64 values,
        // so their product will differ from 1.0 by the input representation error.
        // We just check it's close and the mul doesn't amplify the error.
        let large = f256::from_f64(1e30);
        let small = f256::from_f64(1e-30);
        let product = large * small;
        let err = f256_err(product, f256::ONE);
        assert!(err < 1e-15, "f256 large*small error: {err:e}");
    }

    #[test]
    fn f512_large_times_small() {
        let large = f512::from_f64(1e100);
        let small = f512::from_f64(1e-100);
        let product = large * small;
        let err = f512_err(product, f512::ONE);
        assert!(err < 1e-15, "f512 large*small error: {err:e}");
    }

    // ===================================================================
    // 3. Division precision
    // ===================================================================

    #[test]
    fn f256_div_one_seventh_roundtrip() {
        let one = f256::ONE;
        let seven = f256::from_f64(7.0);
        let seventh = one / seven;
        let result = seventh * seven;
        let err = f256_err(result, one);
        assert!(err < F256_EPS, "f256: 1/7 * 7 error: {err:e}");
    }

    #[test]
    fn f512_div_one_seventh_roundtrip() {
        let one = f512::ONE;
        let seven = f512::from_f64(7.0);
        let seventh = one / seven;
        let result = seventh * seven;
        let err = f512_err(result, one);
        assert!(err < F512_EPS, "f512: 1/7 * 7 error: {err:e}");
    }

    #[test]
    fn f256_div_self_is_one() {
        let x = f256::from_f64(std::f64::consts::PI);
        let result = x / x;
        let err = f256_err(result, f256::ONE);
        assert!(err < F256_EPS, "f256 pi/pi error: {err:e}");
    }

    #[test]
    fn f512_div_self_is_one() {
        let x = f512::from_f64(std::f64::consts::PI);
        let result = x / x;
        let err = f512_err(result, f512::ONE);
        assert!(err < F512_EPS, "f512 pi/pi error: {err:e}");
    }

    // ===================================================================
    // 4. Square root
    // ===================================================================

    #[test]
    fn f256_sqrt_two_squared() {
        let two = f256::from_f64(2.0);
        let s = two.sqrt();
        let check = s * s;
        let err = f256_err(check, two);
        assert!(err < F256_EPS, "f256 sqrt(2)^2 error: {err:e}");
    }

    #[test]
    fn f512_sqrt_two_squared() {
        let two = f512::from_f64(2.0);
        let s = two.sqrt();
        let check = s * s;
        let err = f512_err(check, two);
        assert!(err < F512_EPS, "f512 sqrt(2)^2 error: {err:e}");
    }

    #[test]
    fn f256_sqrt_large_number() {
        let val = f256::from_f64(1e30);
        let s = val.sqrt();
        let check = s * s;
        let rel_err = f256_err(check, val) / 1e30;
        assert!(rel_err < F256_EPS, "f256 sqrt(1e30)^2 relative error: {rel_err:e}");
    }

    #[test]
    fn f512_sqrt_large_number() {
        let val = f512::from_f64(1e100);
        let s = val.sqrt();
        let check = s * s;
        let rel_err = f512_err(check, val) / 1e100;
        assert!(rel_err < F512_EPS, "f512 sqrt(1e100)^2 relative error: {rel_err:e}");
    }

    #[test]
    fn f256_sqrt_edge_cases() {
        assert_eq!(f256::ZERO.sqrt().to_f64(), 0.0);
        assert!(f256::from_f64(-1.0).sqrt().is_nan());
        assert!(f256::from_f64(f64::INFINITY).sqrt().to_f64().is_infinite());
    }

    #[test]
    fn f512_sqrt_edge_cases() {
        assert_eq!(f512::ZERO.sqrt().to_f64(), 0.0);
        assert!(f512::from_f64(-1.0).sqrt().is_nan());
        assert!(f512::from_f64(f64::INFINITY).sqrt().to_f64().is_infinite());
    }

    // ===================================================================
    // 5. Transcendentals
    // ===================================================================

    #[test]
    fn f256_exp_zero_is_one() {
        assert_eq!(f256::ZERO.exp(), f256::ONE);
    }

    #[test]
    fn f512_exp_zero_is_one() {
        assert_eq!(f512::ZERO.exp(), f512::ONE);
    }

    #[test]
    fn f256_exp_times_exp_neg() {
        let e = f256::ONE.exp();
        let inv = (f256::ZERO - f256::ONE).exp();
        let product = e * inv;
        let err = f256_err(product, f256::ONE);
        assert!(err < F256_TOL, "f256 exp(1)*exp(-1) error: {err:e}");
    }

    #[test]
    fn f512_exp_times_exp_neg() {
        let e = f512::ONE.exp();
        let inv = (f512::ZERO - f512::ONE).exp();
        let product = e * inv;
        let err = f512_err(product, f512::ONE);
        assert!(err < F512_TOL, "f512 exp(1)*exp(-1) error: {err:e}");
    }

    #[test]
    fn f256_ln_exp_roundtrip() {
        let x = f256::from_f64(2.5);
        let result = x.exp().ln();
        let err = f256_err(result, x);
        assert!(err < F256_TOL, "f256 ln(exp(2.5)) error: {err:e}");
    }

    #[test]
    fn f512_ln_exp_roundtrip() {
        let x = f512::from_f64(2.5);
        let result = x.exp().ln();
        let err = f512_err(result, x);
        assert!(err < F512_TOL, "f512 ln(exp(2.5)) error: {err:e}");
    }

    #[test]
    fn f256_exp_ln_roundtrip() {
        let x = f256::from_f64(3.14159);
        let result = x.ln().exp();
        let err = f256_err(result, x);
        assert!(err < F256_TOL, "f256 exp(ln(pi)) error: {err:e}");
    }

    #[test]
    fn f512_exp_ln_roundtrip() {
        let x = f512::from_f64(3.14159);
        let result = x.ln().exp();
        let err = f512_err(result, x);
        assert!(err < F512_TOL, "f512 exp(ln(pi)) error: {err:e}");
    }

    #[test]
    fn f256_ln_one_is_zero() {
        let result = f256::ONE.ln();
        assert_eq!(result, f256::ZERO, "f256 ln(1) should be exactly 0");
    }

    #[test]
    fn f512_ln_one_is_zero() {
        let result = f512::ONE.ln();
        assert_eq!(result, f512::ZERO, "f512 ln(1) should be exactly 0");
    }

    #[test]
    fn f256_pow_two_ten() {
        let two = f256::from_f64(2.0);
        let ten = f256::from_f64(10.0);
        let result = two.pow(ten);
        let expected = f256::from_f64(1024.0);
        let err = f256_err(result, expected);
        assert!(err < F256_TOL, "f256 2^10 error: {err:e}");
    }

    #[test]
    fn f512_pow_two_ten() {
        let two = f512::from_f64(2.0);
        let ten = f512::from_f64(10.0);
        let result = two.pow(ten);
        let expected = f512::from_f64(1024.0);
        let err = f512_err(result, expected);
        assert!(err < F512_TOL, "f512 2^10 error: {err:e}");
    }

    // ===================================================================
    // 6. Catastrophic cancellation resistance
    // ===================================================================

    #[test]
    fn f256_kahan_sum_100k() {
        let n = 100_000;
        let one = f256::ONE;
        let mut sum = f256::ZERO;
        for _ in 0..n {
            sum = sum + one;
        }
        assert_eq!(sum.to_f64(), n as f64, "f256 Kahan sum of 100000 ones");
    }

    #[test]
    fn f512_kahan_sum_100k() {
        let n = 100_000;
        let one = f512::ONE;
        let mut sum = f512::ZERO;
        for _ in 0..n {
            sum = sum + one;
        }
        assert_eq!(sum.to_f64(), n as f64, "f512 Kahan sum of 100000 ones");
    }

    #[test]
    fn f256_widely_separated_magnitudes() {
        // (a + b) - a = b for widely separated magnitudes
        let a = f256::from_f64(1e20);
        let b = f256::from_f64(1e-30);
        let result = (a + b) - a;
        let err = f256_err(result, b);
        assert!(err < F256_EPS, "f256 separated magnitude error: {err:e}");
    }

    #[test]
    fn f512_widely_separated_magnitudes() {
        let a = f512::from_f64(1e50);
        let b = f512::from_f64(1e-70);
        let result = (a + b) - a;
        let err = f512_err(result, b);
        assert!(err < F512_EPS, "f512 separated magnitude error: {err:e}");
    }

    // ===================================================================
    // 7. Parsing (FromStr)
    // ===================================================================

    #[test]
    fn f256_parse_pi_precision() {
        let pi_str = "3.14159265358979323846264338327950288419716939937510";
        let pi: f256 = pi_str.parse().unwrap();
        // The f64 approximation of pi
        let f64_pi = std::f64::consts::PI;
        // Our parsed value should be at least as good as f64 pi
        assert!((pi.limbs[0] - f64_pi).abs() < 1e-15,
            "f256 parsed pi leading limb should match f64 pi");
        // And should have non-zero lower limbs capturing extra precision
        let has_extra_precision = pi.limbs[1] != 0.0 || pi.limbs[2] != 0.0;
        assert!(has_extra_precision,
            "f256 parsed pi should have non-zero lower limbs for extra precision");
    }

    #[test]
    fn f512_parse_pi_precision() {
        let pi_str = "3.14159265358979323846264338327950288419716939937510";
        let pi: f512 = pi_str.parse().unwrap();
        assert!((pi.limbs[0] - std::f64::consts::PI).abs() < 1e-15);
        let has_extra_precision = pi.limbs.iter().skip(1).any(|&l| l != 0.0);
        assert!(has_extra_precision,
            "f512 parsed pi should have non-zero lower limbs");
    }

    #[test]
    fn f256_parse_scientific_notation() {
        let v: f256 = "1.23e-10".parse().unwrap();
        assert!((v.to_f64() - 1.23e-10_f64).abs() < 1e-25);

        let v2: f256 = "5.0E3".parse().unwrap();
        assert!((v2.to_f64() - 5000.0).abs() < 1e-10);
    }

    #[test]
    fn f512_parse_scientific_notation() {
        let v: f512 = "1.23e-10".parse().unwrap();
        assert!((v.to_f64() - 1.23e-10_f64).abs() < 1e-25);
    }

    #[test]
    fn f256_parse_negative() {
        let v: f256 = "-42.5".parse().unwrap();
        assert!((v.to_f64() - (-42.5)).abs() < F256_EPS);
    }

    #[test]
    fn f512_parse_negative() {
        let v: f512 = "-42.5".parse().unwrap();
        assert!((v.to_f64() - (-42.5)).abs() < F512_EPS);
    }

    #[test]
    fn f256_parse_invalid() {
        assert!("".parse::<f256>().is_err());
        assert!("abc".parse::<f256>().is_err());
        assert!("1.2.3".parse::<f256>().is_err());
        assert!("-".parse::<f256>().is_err());
        assert!("1.0e".parse::<f256>().is_err());
    }

    #[test]
    fn f512_parse_invalid() {
        assert!("".parse::<f512>().is_err());
        assert!("abc".parse::<f512>().is_err());
        assert!("1.2.3".parse::<f512>().is_err());
    }

    // ===================================================================
    // 8. Comparison
    // ===================================================================

    #[test]
    fn f256_ordering_close_values() {
        let a = f256::from_f64(1.0);
        let b = a + f256::from_f64(1e-50);
        assert!(b > a, "f256: 1 + 1e-50 should be > 1");
        assert!(a < b);
        assert!(a != b);
    }

    #[test]
    fn f512_ordering_close_values() {
        let a = f512::from_f64(1.0);
        let b = a + f512::from_f64(1e-100);
        assert!(b > a, "f512: 1 + 1e-100 should be > 1");
        assert!(a < b);
        assert!(a != b);
    }

    #[test]
    fn f256_partial_eq() {
        let a = f256::from_f64(42.0);
        let b = f256::from_f64(42.0);
        assert_eq!(a, b);

        let c = f256::from_f64(43.0);
        assert_ne!(a, c);
    }

    #[test]
    fn f512_partial_eq() {
        let a = f512::from_f64(42.0);
        let b = f512::from_f64(42.0);
        assert_eq!(a, b);

        let c = f512::from_f64(43.0);
        assert_ne!(a, c);
    }

    #[test]
    fn f256_ordering_negative() {
        let neg = f256::from_f64(-1.0);
        let pos = f256::from_f64(1.0);
        assert!(neg < pos);
        assert!(pos > neg);
    }

    // ===================================================================
    // 9. Conversions
    // ===================================================================

    #[test]
    fn f256_from_f64_roundtrip() {
        let val = std::f64::consts::E;
        let mf = f256::from_f64(val);
        assert_eq!(mf.to_f64(), val);
        assert_eq!(mf.limbs[0], val);
        assert_eq!(mf.limbs[1], 0.0);
    }

    #[test]
    fn f512_from_f64_roundtrip() {
        let val = std::f64::consts::E;
        let mf = f512::from_f64(val);
        assert_eq!(mf.to_f64(), val);
    }

    #[test]
    fn f256_from_i64_large() {
        // Values > 2^53 should not lose precision
        let x: i64 = (1i64 << 53) + 1;
        let f = f256::from(x);
        // Reconstruct the integer from the limbs
        let back = f.limbs[0] as i64 + f.limbs[1] as i64;
        assert_eq!(back, x, "f256 i64 roundtrip failed for 2^53+1");
    }

    #[test]
    fn f512_from_i64_large() {
        let x: i64 = (1i64 << 53) + 1;
        let f = f512::from(x);
        let back = f.limbs[0] as i64 + f.limbs[1] as i64;
        assert_eq!(back, x, "f512 i64 roundtrip failed for 2^53+1");
    }

    #[test]
    fn f256_to_f64_best_approx() {
        // to_f64 returns the best f64 approximation (sum of top 2 limbs)
        let pi = f256::from_f64(std::f64::consts::PI);
        assert_eq!(pi.to_f64(), std::f64::consts::PI);
    }

    #[test]
    fn f256_from_f128_conversion() {
        let hi = std::f64::consts::PI;
        let lo = 1.2246467991473532e-16; // correction term for pi
        let dd = crate::f128::new(hi, lo);
        let mf = f256::from_f128(dd);
        assert_eq!(mf.limbs[0], hi);
        assert_eq!(mf.limbs[1], lo);
        // Round-trip back
        let back = mf.to_f128();
        assert_eq!(back.hi, hi);
        assert_eq!(back.lo, lo);
    }

    #[test]
    fn f512_from_f256_conversion() {
        let val = f256::from_f64(std::f64::consts::PI);
        let wide = f512::from_f256(val);
        assert_eq!(wide.limbs[0], val.limbs[0]);
        assert_eq!(wide.limbs[1], val.limbs[1]);
        assert_eq!(wide.limbs[2], val.limbs[2]);
        assert_eq!(wide.limbs[3], val.limbs[3]);
        // Convert back
        let back = wide.to_f256();
        assert_eq!(back.limbs, val.limbs);
    }

    // ===================================================================
    // 10. Linear algebra
    // ===================================================================

    #[test]
    fn f256_dot_orthogonal_is_zero() {
        // Two orthogonal vectors: [1, 0, 0] . [0, 1, 0] = 0
        let a = vec![f256::ONE, f256::ZERO, f256::ZERO];
        let b = vec![f256::ZERO, f256::ONE, f256::ZERO];
        let result = mf_dot(&a, &b);
        assert_eq!(result.to_f64(), 0.0, "f256 dot product of orthogonal vectors");
    }

    #[test]
    fn f512_dot_orthogonal_is_zero() {
        let a = vec![f512::ONE, f512::ZERO, f512::ZERO];
        let b = vec![f512::ZERO, f512::ONE, f512::ZERO];
        let result = mf_dot(&a, &b);
        assert_eq!(result.to_f64(), 0.0, "f512 dot product of orthogonal vectors");
    }

    #[test]
    fn f256_gemm_2x2() {
        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        let a = [1.0, 2.0, 3.0, 4.0].map(|v| f256::from_f64(v));
        let b = [5.0, 6.0, 7.0, 8.0].map(|v| f256::from_f64(v));
        let mut c = vec![f256::ZERO; 4];
        mf_gemm(&a, &b, &mut c, 2, 2, 2);
        let expected = [19.0, 22.0, 43.0, 50.0];
        for i in 0..4 {
            let err = (c[i].to_f64() - expected[i]).abs();
            assert!(err < 1e-28, "f256 2x2 gemm mismatch at {i}: {err:e}");
        }
    }

    #[test]
    fn f256_gemm_3x3() {
        // 3x3 identity * A = A
        let n = 3;
        let mut eye = vec![f256::ZERO; n * n];
        for i in 0..n { eye[i * n + i] = f256::ONE; }
        let a: Vec<f256> = (1..=9).map(|v| f256::from_f64(v as f64)).collect();
        let mut c = vec![f256::ZERO; n * n];
        mf_gemm(&eye, &a, &mut c, n, n, n);
        for i in 0..n * n {
            let err = (c[i].to_f64() - a[i].to_f64()).abs();
            assert!(err < 1e-28, "f256 I*A mismatch at {i}: {err:e}");
        }
    }

    #[test]
    fn f256_cholesky_identity() {
        let n = 3;
        let mut eye = vec![f256::ZERO; n * n];
        for i in 0..n { eye[i * n + i] = f256::ONE; }
        mf_cholesky(&mut eye, n).expect("identity should be PD");
        // Cholesky of identity should be identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let val = eye[i * n + j].to_f64();
                // Only lower triangle matters; upper triangle is not zeroed
                if i >= j {
                    assert!((val - expected).abs() < F256_EPS,
                        "f256 cholesky(I) [{i}][{j}] = {val}, expected {expected}");
                }
            }
        }
    }

    #[test]
    fn f256_cholesky_solve_simple() {
        // Solve A*x = b where A = [[4, 2], [2, 3]], b = [1, 1]
        // Solution: x = [1/8, 1/4] = [0.125, 0.25]
        let n = 2;
        let mut a = vec![
            f256::from_f64(4.0), f256::from_f64(2.0),
            f256::from_f64(2.0), f256::from_f64(3.0),
        ];
        let a_copy = a.clone();
        mf_cholesky(&mut a, n).expect("Should be PD");
        let b = vec![f256::ONE, f256::ONE];
        let mut x = vec![f256::ZERO; n];
        mf_cholesky_solve(&a, &b, &mut x, n);

        // Verify by computing A*x and comparing to b
        let mut ax = vec![f256::ZERO; n];
        mf_gemv(&a_copy, &x, &mut ax, n, n);
        for i in 0..n {
            let err = f256_err(ax[i], b[i]);
            assert!(err < F256_EPS, "f256 cholesky solve residual[{i}] = {err:e}");
        }
    }

    #[test]
    fn f512_cholesky_solve_3x3() {
        let n = 3;
        let mut a = vec![
            f512::from_f64(4.0), f512::from_f64(2.0), f512::from_f64(0.0),
            f512::from_f64(2.0), f512::from_f64(5.0), f512::from_f64(1.0),
            f512::from_f64(0.0), f512::from_f64(1.0), f512::from_f64(6.0),
        ];
        let a_copy = a.clone();
        mf_cholesky(&mut a, n).expect("Should be PD");
        let b = vec![f512::ONE, f512::ONE, f512::ONE];
        let mut x = vec![f512::ZERO; n];
        mf_cholesky_solve(&a, &b, &mut x, n);

        let mut ax = vec![f512::ZERO; n];
        mf_gemv(&a_copy, &x, &mut ax, n, n);
        for i in 0..n {
            let err = f512_err(ax[i], b[i]);
            assert!(err < F512_EPS, "f512 cholesky solve residual[{i}] = {err:e}");
        }
    }

    // ===================================================================
    // 11. Cross-precision: verify f256 beats f128, f512 beats f256
    // ===================================================================

    #[test]
    fn f256_more_precise_than_f128() {
        // f256 (4 limbs, ~212 bits) should achieve ~63 digits of precision
        let f256_val = f256::ONE / f256::from_f64(7.0);
        let f256_err_val = f256_err(f256_val * f256::from_f64(7.0), f256::ONE);
        // Error should be well below f64 epsilon (~1e-16)
        assert!(f256_err_val < 1e-50,
            "f256 1/7*7 error should be < 1e-50: {f256_err_val:e}");
    }

    #[test]
    fn f512_more_precise_than_f256() {
        // Compute 1/7 * 7 - 1 in both f256 and f512
        let f256_seventh = f256::ONE / f256::from_f64(7.0);
        let f256_err_val = f256_err(f256_seventh * f256::from_f64(7.0), f256::ONE);

        let f512_seventh = f512::ONE / f512::from_f64(7.0);
        let f512_err_val = f512_err(f512_seventh * f512::from_f64(7.0), f512::ONE);

        assert!(f512_err_val < f256_err_val,
            "f512 should be more precise than f256: f512={f512_err_val:e} vs f256={f256_err_val:e}");
    }

    #[test]
    fn f256_sqrt_more_precise_than_f128() {
        // f256 sqrt(2)^2 should have error well below f64 epsilon
        let two_256 = f256::from_f64(2.0);
        let s256 = two_256.sqrt();
        let err_256 = f256_err(s256 * s256, two_256);
        assert!(err_256 < 1e-50,
            "f256 sqrt(2)^2 error should be < 1e-50: {err_256:e}");
    }

    #[test]
    fn f512_sqrt_more_precise_than_f256() {
        let two_256 = f256::from_f64(2.0);
        let s256 = two_256.sqrt();
        let err_256 = f256_err(s256 * s256, two_256);

        let two_512 = f512::from_f64(2.0);
        let s512 = two_512.sqrt();
        let err_512 = f512_err(s512 * s512, two_512);

        assert!(err_512 < err_256,
            "f512 sqrt should be more precise than f256: f512={err_512:e} vs f256={err_256:e}");
    }

    // ===================================================================
    // Additional edge case tests
    // ===================================================================

    #[test]
    fn f256_neg_roundtrip() {
        let a = f256::from_f64(std::f64::consts::PI);
        let b = -(-a);
        assert_eq!(a, b, "f256 double negation should be identity");
    }

    #[test]
    fn f512_neg_roundtrip() {
        let a = f512::from_f64(std::f64::consts::PI);
        let b = -(-a);
        assert_eq!(a, b, "f512 double negation should be identity");
    }

    #[test]
    fn f256_sub_self_is_zero() {
        let a = f256::from_f64(42.0);
        let r = a - a;
        assert_eq!(r.to_f64(), 0.0, "f256 a - a should be 0");
    }

    #[test]
    fn f512_sub_self_is_zero() {
        let a = f512::from_f64(42.0);
        let r = a - a;
        assert_eq!(r.to_f64(), 0.0, "f512 a - a should be 0");
    }

    #[test]
    fn f256_mul_by_zero() {
        let a = f256::from_f64(42.0);
        let r = a * f256::ZERO;
        assert_eq!(r.to_f64(), 0.0, "f256 x * 0 should be 0");
    }

    #[test]
    fn f256_sum_trait() {
        let vals: Vec<f256> = (1..=100).map(|i| f256::from_f64(i as f64)).collect();
        let total: f256 = vals.iter().sum();
        assert_eq!(total.to_f64(), 5050.0, "f256 sum 1..100 should be 5050");
    }

    #[test]
    fn f512_sum_trait() {
        let vals: Vec<f512> = (1..=100).map(|i| f512::from_f64(i as f64)).collect();
        let total: f512 = vals.iter().sum();
        assert_eq!(total.to_f64(), 5050.0, "f512 sum 1..100 should be 5050");
    }

    #[test]
    fn f256_is_predicates() {
        assert!(f256::ZERO.is_zero());
        assert!(!f256::ZERO.is_nan());
        assert!(f256::ZERO.is_finite());
        assert!(f256::from_f64(f64::NAN).is_nan());
        assert!(!f256::from_f64(f64::INFINITY).is_finite());
    }

    #[test]
    fn f512_is_predicates() {
        assert!(f512::ZERO.is_zero());
        assert!(!f512::ZERO.is_nan());
        assert!(f512::ZERO.is_finite());
        assert!(f512::from_f64(f64::NAN).is_nan());
        assert!(!f512::from_f64(f64::INFINITY).is_finite());
    }

    #[test]
    fn f256_recip_precision() {
        let x = f256::from_f64(7.0);
        let r = x.recip();
        let product = r * x;
        let err = f256_err(product, f256::ONE);
        assert!(err < F256_EPS, "f256 recip(7) * 7 error: {err:e}");
    }

    #[test]
    fn f512_recip_precision() {
        let x = f512::from_f64(7.0);
        let r = x.recip();
        let product = r * x;
        let err = f512_err(product, f512::ONE);
        assert!(err < F512_EPS, "f512 recip(7) * 7 error: {err:e}");
    }

    #[test]
    fn f256_mixed_f64_arithmetic() {
        let a = f256::from_f64(1.0);
        let b = a + 1e-30;
        let c = b - 1e-30;
        let err = f256_err(c, a);
        assert!(err < F256_EPS, "f256 mixed f64 add/sub error: {err:e}");

        let d = f256::from_f64(3.0) * 2.0;
        assert_eq!(d.to_f64(), 6.0);
    }

    #[test]
    fn f256_rem_basic() {
        let ten = f256::from_f64(10.0);
        let three = f256::from_f64(3.0);
        let r = ten % three;
        let err = f256_err(r, f256::ONE);
        assert!(err < F256_EPS, "f256 10 % 3 error: {err:e}");
    }

    #[test]
    fn f256_ln_negative_is_nan() {
        let result = f256::from_f64(-1.0).ln();
        assert!(result.is_nan(), "f256 ln(-1) should be NaN");
    }

    #[test]
    fn f256_ln_zero_is_neg_inf() {
        let result = f256::ZERO.ln();
        assert!(result.limbs[0].is_infinite() && result.limbs[0] < 0.0,
            "f256 ln(0) should be -inf");
    }

    #[test]
    fn f256_exp_large_positive_overflows() {
        let result = f256::from_f64(710.0).exp();
        assert!(result.limbs[0].is_infinite(), "f256 exp(710) should overflow");
    }

    #[test]
    fn f256_exp_large_negative_underflows() {
        let result = f256::from_f64(-750.0).exp();
        assert_eq!(result.limbs[0], 0.0, "f256 exp(-750) should underflow to 0");
    }

    #[test]
    fn f256_pow_edge_cases() {
        // x^0 = 1
        let x = f256::from_f64(42.0);
        assert_eq!(x.pow(f256::ZERO), f256::ONE);

        // Negative base with non-integer exponent is NaN
        let neg = f256::from_f64(-2.0);
        let half = f256::from_f64(0.5);
        assert!(neg.pow(half).is_nan());
    }

    #[test]
    fn f256_from_u64_large() {
        let x: u64 = (1u64 << 53) + 1;
        let f = f256::from(x);
        let back = f.limbs[0] as u64 + f.limbs[1] as u64;
        assert_eq!(back, x, "f256 u64 roundtrip failed for 2^53+1");
    }

    #[test]
    fn f256_from_i32() {
        let f = f256::from(42i32);
        assert_eq!(f.to_f64(), 42.0);
    }

    #[test]
    fn f256_from_f32() {
        let f = f256::from(3.14f32);
        assert!((f.to_f64() - 3.14f32 as f64).abs() < 1e-6);
    }
}
