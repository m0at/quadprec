# quadprec

Quad-precision linear algebra in Rust for solving ill-conditioned symmetric positive definite systems — specifically the overlap matrix problem that kills SCF convergence in quantum chemistry.

```
cargo run --release -p demo
```

## Why your SCF diverges

In Hartree-Fock and DFT, the Roothaan-Hall equation is a generalized eigenvalue problem:

$$\mathbf{F}\mathbf{C} = \mathbf{S}\mathbf{C}\boldsymbol{\varepsilon}$$

where $\mathbf{S}$ is the overlap matrix of atomic orbital basis functions:

$$S_{\mu\nu} = \langle \phi_\mu | \phi_\nu \rangle = \int \phi_\mu(\mathbf{r})\, \phi_\nu(\mathbf{r})\, d\mathbf{r}$$

The standard approach orthogonalizes the basis via $\mathbf{S} = \mathbf{L}\mathbf{L}^T$ (Cholesky) or $\mathbf{X} = \mathbf{S}^{-1/2}$ (canonical orthogonalization). Both require inverting or factoring $\mathbf{S}$.

**The problem is the condition number** $\kappa(\mathbf{S})$. When you use augmented or diffuse basis sets (aug-cc-pVDZ, aug-cc-pVTZ), nearby diffuse Gaussians overlap almost completely:

$$S_{\mu\nu} = \left(\frac{4\alpha_\mu\alpha_\nu}{(\alpha_\mu + \alpha_\nu)^2}\right)^{d/4} \exp\!\left(-\frac{\alpha_\mu\alpha_\nu}{\alpha_\mu + \alpha_\nu}|\mathbf{R}_\mu - \mathbf{R}_\nu|^2\right)$$

For diffuse exponents $\alpha \sim 0.01$, this approaches 1.0 for adjacent atoms, making $\mathbf{S}$ nearly singular. The condition number grows with system size:

| System | $N_\text{basis}$ | $\kappa(\mathbf{S})$ | f64 status |
|--------|:-:|:-:|:-:|
| Small molecule | ~100 | $10^4$ | Fine |
| Protein fragment | ~750 | $10^{13}$ | Cholesky fails |
| Periodic slab | ~2000 | $10^{20}$+ | Impossible |

**f64 has ~16 decimal digits.** When $\kappa(\mathbf{S}) > 10^{13}$, Cholesky loses all significant digits. When $\kappa(\mathbf{S}) > 10^{16}$, Cholesky produces a negative pivot and refuses to continue. Your SCF never starts.

This is a [well-known failure mode](https://server.ccl.net/chemistry/resources/messages/2013/11/07.006-dir/). Current workarounds:
- **Remove diffuse functions** — sacrifices accuracy for the properties you needed them for
- **Use MOLOPT basis sets** (CP2K) — sidesteps the problem at the theory level, not the compute level
- **Canonical orthogonalization with threshold** — discards near-null eigenvectors, changes your basis

None of these solve the underlying numerical problem. They all compromise the physics.

## Why this fixes it

This crate implements **double-double arithmetic**: each scalar is stored as a pair of `f64` values $(h, \ell)$ where the true value is $h + \ell$ exactly, with $|\ell| \leq \frac{1}{2}\,\text{ulp}(h)$.

$$\text{f64: } \epsilon \approx 10^{-16} \quad(16\text{ digits})$$

$$\text{f128 (double-double): } \epsilon \approx 10^{-31} \quad(31\text{ digits})$$

Every arithmetic operation uses error-free transformations (Dekker/Knuth):

$$\texttt{TwoSum}(a,b): \quad s + e = a + b \quad\text{exactly}$$

$$\texttt{TwoProd}(a,b): \quad p + e = a \times b \quad\text{exactly (via FMA)}$$

This gives you ~31 decimal digits per scalar. The Cholesky factorization now handles:

| $\kappa(\mathbf{S})$ | f64 | f128 (this crate) |
|:-:|:-:|:-:|
| $10^{13}$ | Fails | $\|\mathbf{x} - \mathbf{x}_\text{true}\| \sim 10^{-18}$ |
| $10^{15}$ | Fails | $\|\mathbf{x} - \mathbf{x}_\text{true}\| \sim 10^{-14}$ |
| $10^{20}$ | Fails | $\|\mathbf{x} - \mathbf{x}_\text{true}\| \sim 10^{-8}$ |
| $10^{22}$ | Fails | $\|\mathbf{x} - \mathbf{x}_\text{true}\| \sim 10^{-5}$ |

The solver auto-selects the cheapest precision strategy:

1. **$\kappa < 10^{10}$**: Pure f64 Cholesky. No overhead.
2. **$10^{10} < \kappa < 10^{16}$**: f64 Cholesky + iterative refinement in f128. Factor once in f64 (fast), compute residual $\mathbf{r} = \mathbf{b} - \mathbf{A}\mathbf{x}$ in f128 (exact), solve correction $\mathbf{A}\,\delta\mathbf{x} = \mathbf{r}$ in f128. Converges in 1–3 iterations.
3. **$\kappa > 10^{16}$**: Full f128 Cholesky. f64 can't even factor; f128 handles it.

## How to use it

### Drop-in solve (f64 input)

```rust
use quadprec_solver::solve_spd;

let a: Vec<f64> = your_overlap_matrix;  // row-major, n x n
let b: Vec<f64> = your_rhs;
let n = basis_size;

let sol = solve_spd(&a, &b, n)?;

sol.x       // Vec<f128> — full quad-precision solution
sol.x_f64   // Vec<f64>  — truncated (loses digits when kappa > 10^10)
sol.kappa   // f64       — estimated condition number
sol.strategy // Float64 | MixedPrecision | Float128
```

**Note:** When $\kappa(\mathbf{S})$ is large, the solution accuracy is limited by the precision of your *input*. If $\mathbf{A}$ and $\mathbf{b}$ are assembled in f64, you lose $\log_{10}\kappa$ digits before the solve even starts. For the full payoff, assemble in f128.

### Full-precision assembly (the real payoff)

```rust
use quad::f128;
use quadprec_solver::solve_spd_f128;

// Assemble S in quad precision — diffuse pairs get full 31-digit accuracy
let mut s = vec![f128::ZERO; n * n];
for (i, j) in shell_pairs {
    let alpha_i = f128::from_f64(exponents[i]);
    let alpha_j = f128::from_f64(exponents[j]);
    let r2 = compute_distance_sq_f128(centers[i], centers[j]);
    s[i * n + j] = overlap_integral_f128(alpha_i, alpha_j, r2);
    s[j * n + i] = s[i * n + j];
}

let sol = solve_spd_f128(&s, &b, n)?;
// sol.x has ~31 - log10(kappa) correct digits
```

### Using `f128` directly

```rust
use quad::f128;

let a = f128::from_f64(1.0);
let b = f128::from_f64(1e-20);
let c = a + b;
let d = c - a;
// d.to_f64() == 1e-20 (exact — f64 would give 0.0)

// Supports: + - * / sqrt() recip() abs()
// All operator-overloaded, all #[inline(always)]
```

## Integrating with your SCF code

The precision-critical path in SCF is small. You don't need f128 everywhere:

```
Your existing code (f64)             This crate (f128)
┌───────────────────────┐            ┌────────────────────────┐
│ Electron integrals    │            │                        │
│ (Coulomb, exchange)   │            │ Overlap matrix S       │
│                       │            │ Cholesky / eigensolve  │
│ Fock matrix assembly  │◄──────────►│ Iterative refinement   │
│                       │            │ Orthogonalization      │
│ Density matrix P      │            │                        │
│ SCF convergence check │            │                        │
└───────────────────────┘            └────────────────────────┘
        ~90% of wall time                  ~10% of wall time
```

Concretely, to integrate into an existing Hartree-Fock or DFT code:

1. **Replace your overlap matrix assembly** with f128-precision computation for shell pairs involving diffuse functions. Use `quad::f128` arithmetic.

2. **Replace your Cholesky/orthogonalization** with `solve_spd_f128()` or use `quad::cholesky()` directly on the f128 overlap matrix.

3. **Keep everything else in f64.** Coulomb/exchange integrals, Fock matrix construction, density matrix updates — all stay in f64. These are not precision-limited, they're compute-limited.

4. **Use iterative refinement** (built into `solve_spd()`) for the generalized eigenvalue problem: solve in f64, correct in f128, converge.

If your code uses [libint2](https://github.com/evaleev/libint) or [libcint](https://github.com/sunqm/libcint) for integral evaluation, you can FFI to them for everything except the overlap integrals, which you recompute in f128.

## Demo output

```
--- Part 2: f128 input via solve_spd_f128() ---
    (matrix assembled in quad precision — the real payoff)

  Hilbert  n=12   kappa=1.1e13   ||x-1||=9.89e-18   ||r||=1.18e-31
  Hilbert  n=13   kappa=1.8e14   ||x-1||=1.05e-16   ||r||=5.70e-32
  Hilbert  n=14   kappa=2.9e15   ||x-1||=5.10e-14   ||r||=5.59e-32
  Hilbert  n=15   kappa=4.7e16   ||x-1||=5.22e-12   ||r||=9.14e-32
  Hilbert  n=18   kappa=1.9e20   ||x-1||=5.46e-8    ||r||=1.09e-31
  Hilbert  n=20   kappa=4.9e22   ||x-1||=9.52e-5    ||r||=1.07e-31
```

f64 Cholesky fails at n=13. f128 keeps going through n=20 ($\kappa = 5 \times 10^{22}$). Residual stays at $\sim 10^{-31}$ throughout — the solver is producing answers accurate to the limits of 31-digit arithmetic.

## Why not just use a GPU?

GPUs are going the wrong direction for this problem:

| | FP64 | FP16/FP8 |
|---|---|---|
| H100 (Hopper) | 34 TFLOPS | 1979 TFLOPS |
| B200 (Blackwell) | 40 TFLOPS | 9000+ TFLOPS |
| **Ratio shift** | **1.2x** | **4.5x** |

The silicon budget is pouring into low-precision AI modes. FP64 is an afterthought. And **Apple Silicon GPUs have no FP64 at all** — Metal only supports FP32 and FP16.

Double-double arithmetic runs entirely on the CPU, uses only f64 hardware (which every CPU has), and parallelizes via [Rayon](https://github.com/rayon-rs/rayon) work-stealing. On an M-series Mac, you get 10 performance cores doing f128 Cholesky while the GPU sits idle (or handles the precision-tolerant work).

## Crate structure

```
crates/
  quad/              # f128 type + BLAS-like ops (dot, gemv, cholesky, triangular solve)
  compensated-sum/   # Neumaier compensated summation for f64 and f128
  solver/            # solve_spd() — auto-selects precision, iterative refinement
  demo/              # cargo run --release -p demo
```

## Requirements

- Rust 2024 edition
- Any platform with FMA (fused multiply-add) support — all modern x86-64 and ARM processors
- No external dependencies beyond `rayon`

## References

- T.J. Dekker, "A floating-point technique for extending the available precision" (1971)
- D.E. Knuth, *The Art of Computer Programming*, Vol 2 — error-free transformations
- [SCF convergence failure with aug-cc-pVDZ](https://server.ccl.net/chemistry/resources/messages/2013/11/07.006-dir/) — the exact failure this solves
- [CP2K SCF convergence problems with large basis sets](https://groups.google.com/g/cp2k/c/6JDb5uy5r0M)
- Y. Hida, X.S. Li, D.H. Bailey, "Algorithms for Quad-Double Precision Floating Point Arithmetic" (2001)
