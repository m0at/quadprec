# quadprec

Software-emulated quad-precision linear algebra in Rust. Solves ill-conditioned
SPD systems (overlap matrices, Hilbert matrices) where f64 Cholesky fails.

## The problem

Basis sets like aug-cc-pVDZ produce overlap matrices with condition numbers
kappa(S) > 10^16 at large system sizes. f64 Cholesky fails at kappa ~ 10^13.
GPU hardware is cutting f64 support (Nvidia Blackwell halved f64 TFLOPS vs Hopper).

## The solution

Double-double arithmetic: each value is a pair of f64s giving ~32 decimal digits.
This is 100% CPU-friendly (no GPU needed), parallelized via Rayon, and handles
kappa up to ~10^30 before losing all precision.

```
cargo run --release -p demo
```

## Quick start

```rust
use quadprec_solver::solve_spd;

// Your ill-conditioned SPD matrix (row-major f64)
let a: Vec<f64> = assemble_overlap_matrix();
let b: Vec<f64> = compute_rhs();
let n = basis_size;

let sol = solve_spd(&a, &b, n).unwrap();
// sol.x      — solution in f128 (use this for ill-conditioned systems)
// sol.x_f64  — solution truncated to f64 (convenience)
// sol.kappa  — estimated condition number
// sol.strategy — Float64 / MixedPrecision / Float128

// Auto-selects the cheapest strategy:
// - kappa < 10^10:  pure f64 (fast)
// - kappa 10^10-16: f64 solve + iterative refinement in f128
// - kappa > 10^16:  f64 Cholesky fails, full f128 factorization
```

For matrices assembled in quad precision (the real payoff for SCF):

```rust
use quad::f128;
use quadprec_solver::solve_spd_f128;

let a: Vec<f128> = assemble_overlap_f128();
let b: Vec<f128> = compute_rhs_f128();
let sol = solve_spd_f128(&a, &b, n).unwrap();
```

## Demo output

```
--- f128 input via solve_spd_f128() ---
    (matrix assembled in quad precision)

  Hilbert  n=12   kappa=1.1e13   ||x-1||=9.89e-18   ||r||=1.18e-31
  Hilbert  n=14   kappa=2.9e15   ||x-1||=5.10e-14   ||r||=5.59e-32
  Hilbert  n=18   kappa=1.9e20   ||x-1||=5.46e-8    ||r||=1.09e-31
  Hilbert  n=20   kappa=4.9e22   ||x-1||=9.52e-5    ||r||=1.07e-31
```

f64 Cholesky fails at n=13. f128 keeps going through n=20 (kappa = 10^22).

## Crate structure

```
crates/
  quad/              # f128 type + BLAS-like ops (dot, gemv, cholesky)
  compensated-sum/   # Neumaier compensated summation (f64 and f128)
  solver/            # solve_spd() — auto-selects precision strategy
  demo/              # cargo run --release -p demo
```

## Performance

Double-double arithmetic is ~10x slower than native f64 per FLOP. But:
- Cholesky is O(n^3) — for n=20 (Hilbert), it's sub-millisecond
- For large well-conditioned systems, the solver uses f64 (no overhead)
- Rayon parallelizes the off-diagonal Cholesky update and matvec

The precision-critical path is typically a small fraction of total SCF wall time.

## Requirements

- Rust 2024 edition
- Any platform with FMA support (all modern x86-64 and ARM)
