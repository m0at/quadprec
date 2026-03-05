use quad::f128;
use quadprec_solver::{solve_spd, solve_spd_f128};
use std::time::Instant;

fn main() {
    println!("quadprec solver demo");
    println!("====================\n");

    println!("--- Part 1: f64 input via solve_spd() ---");
    println!("    (solution quality limited by f64 input rounding * kappa)\n");

    for &n in &[8, 10, 12, 13] {
        run_f64_input("Hilbert", &hilbert_f64(n), n);
    }
    println!();
    for &(n, z) in &[(60, 0.05), (100, 0.04), (200, 0.03), (400, 0.02)] {
        run_f64_input(&format!("Overlap z={z}"), &overlap_f64(n, z), n);
    }

    println!("\n--- Part 2: f128 input via solve_spd_f128() ---");
    println!("    (matrix assembled in quad precision — the real payoff)\n");

    for &n in &[12, 13, 14, 15, 18, 20] {
        run_f128_input("Hilbert", n);
    }

    println!("\n--- Part 3: Scaling (well-conditioned, shows Rayon parallelism) ---\n");
    for &n in &[200, 500, 1000] {
        run_f64_input("Shifted-Hilbert", &shifted_hilbert(n, 5.0), n);
    }
}

fn run_f64_input(label: &str, a: &[f64], n: usize) {
    let mut b = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            b[i] += a[i * n + j];
        }
    }
    let t0 = Instant::now();
    match solve_spd(a, &b, n) {
        Ok(sol) => {
            let dt = t0.elapsed();
            println!(
                "  {label:>20}  n={n:<4}  kappa={:<10.1e}  ||r||={:<10.2e}  {:?}  {:.1}ms  ({} refine)",
                sol.kappa, sol.residual_norm,
                sol.strategy, dt.as_secs_f64() * 1000.0, sol.refinement_iters,
            );
        }
        Err(e) => println!("  {label:>20}  n={n:<4}  FAILED: {e}"),
    }
}

fn run_f128_input(label: &str, n: usize) {
    // Assemble Hilbert matrix in f128 — each element computed with full quad precision
    let a128 = hilbert_f128(n);
    // b = A * ones computed in f128 (no rounding)
    let ones = vec![f128::ONE; n];
    let mut b128 = vec![f128::ZERO; n];
    for i in 0..n {
        for j in 0..n {
            b128[i] = b128[i] + a128[i * n + j] * ones[j];
        }
    }

    let t0 = Instant::now();
    match solve_spd_f128(&a128, &b128, n) {
        Ok(sol) => {
            let dt = t0.elapsed();
            let err: f64 = sol.x.iter()
                .map(|xi| (*xi - f128::ONE).to_f64().powi(2))
                .sum::<f64>().sqrt();
            println!(
                "  {label:>20}  n={n:<4}  kappa={:<10.1e}  ||x-1||={:<10.2e}  ||r||={:<10.2e}  {:.1}ms",
                sol.kappa, err, sol.residual_norm,
                dt.as_secs_f64() * 1000.0,
            );
        }
        Err(e) => println!("  {label:>20}  n={n:<4}  FAILED: {e}"),
    }
}

fn hilbert_f64(n: usize) -> Vec<f64> {
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = 1.0 / (i + j + 1) as f64;
        }
    }
    h
}

fn hilbert_f128(n: usize) -> Vec<f128> {
    let mut h = vec![f128::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            // Compute 1/(i+j+1) in f128 for full precision
            h[i * n + j] = f128::ONE / f128::from_f64((i + j + 1) as f64);
        }
    }
    h
}

fn overlap_f64(n: usize, zeta_diffuse: f64) -> Vec<f64> {
    let mut s = vec![0.0f64; n * n];
    let spacing = 2.0;
    let positions: Vec<f64> = (0..n).map(|i| (i / 2) as f64 * spacing).collect();
    let exponents: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { zeta_diffuse })
        .collect();
    for i in 0..n {
        for j in 0..=i {
            let zi = exponents[i];
            let zj = exponents[j];
            let zsum = zi + zj;
            let dr = positions[i] - positions[j];
            let val = (4.0 * zi * zj / (zsum * zsum)).sqrt().sqrt()
                * (-(zi * zj / zsum) * dr * dr).exp();
            s[i * n + j] = val;
            s[j * n + i] = val;
        }
    }
    s
}

fn shifted_hilbert(n: usize, shift: f64) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = 1.0 / (i + j + 1) as f64;
        }
        a[i * n + i] += shift;
    }
    a
}
