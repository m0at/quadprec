use quad::f128;
use rayon::prelude::*;

/// Neumaier compensated summation for f64 values.
/// Error is O(eps) instead of O(n*eps).
pub fn neumaier_sum_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64; // running compensation

    for &v in values {
        let t = sum + v;
        if sum.abs() >= v.abs() {
            comp += (sum - t) + v;
        } else {
            comp += (v - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// Neumaier compensated summation for f128 values.
pub fn neumaier_sum(values: &[f128]) -> f128 {
    let mut sum = f128::ZERO;
    let mut comp = f128::ZERO;

    for &v in values {
        let t = sum + v;
        if sum.abs().hi >= v.abs().hi {
            comp += (sum - t) + v;
        } else {
            comp += (v - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// Parallel compensated sum: partition into chunks, Neumaier-sum each chunk,
/// then Neumaier-sum the partial results.
pub fn par_neumaier_sum_f64(values: &[f64]) -> f64 {
    const CHUNK: usize = 4096;
    if values.len() < CHUNK {
        return neumaier_sum_f64(values);
    }
    let partials: Vec<f64> = values
        .par_chunks(CHUNK)
        .map(|chunk| neumaier_sum_f64(chunk))
        .collect();
    neumaier_sum_f64(&partials)
}

/// Parallel compensated sum for f128.
pub fn par_neumaier_sum(values: &[f128]) -> f128 {
    const CHUNK: usize = 4096;
    if values.len() < CHUNK {
        return neumaier_sum(values);
    }
    let partials: Vec<f128> = values
        .par_chunks(CHUNK)
        .map(|chunk| neumaier_sum(chunk))
        .collect();
    neumaier_sum(&partials)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_f64_precision() {
        // Sum 1.0 + 1e-16 * 10_000_000 times. Naive sum loses the small terms.
        let n = 10_000_000;
        let mut values = vec![1e-16_f64; n];
        values[0] = 1.0;

        let naive: f64 = values.iter().sum();
        let compensated = neumaier_sum_f64(&values);
        let exact = 1.0 + (n - 1) as f64 * 1e-16;

        // Compensated should be closer to exact than naive
        let naive_err = (naive - exact).abs();
        let comp_err = (compensated - exact).abs();
        assert!(comp_err < naive_err || comp_err < 1e-15,
            "Compensated ({comp_err:e}) should beat naive ({naive_err:e})");
    }

    #[test]
    fn par_sum_matches_serial() {
        let values: Vec<f64> = (0..100_000).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let serial = neumaier_sum_f64(&values);
        let parallel = par_neumaier_sum_f64(&values);
        assert!((serial - parallel).abs() < 1e-14,
            "Serial {serial} != parallel {parallel}");
    }

    #[test]
    fn neumaier_f128_basic() {
        let values: Vec<f128> = (1..=1000).map(|i| f128::from_f64(1.0 / i as f64)).collect();
        let result = neumaier_sum(&values);
        // Harmonic sum H_1000 ≈ 7.485470...
        assert!((result.to_f64() - 7.485470860550343).abs() < 1e-12);
    }
}
