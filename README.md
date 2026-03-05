# hyperprec

Software-emulated extended precision floating point in Rust: **f128**, **f256**, and **f512**.

```rust
use hyperprec::{f128, f256, f512};

let x = f256::from_f64(1.0) + f256::from_f64(1e-40);
let y = x - f256::from_f64(1.0);
// y.to_f64() == 1e-40 (exact -- f64 would give 0.0)
```

## Precision

| Type | Limbs | Significand bits | Decimal digits | Relative cost vs f64 |
|------|:-----:|:----------------:|:--------------:|:--------------------:|
| `f64`  | 1 | 53   | ~16  | 1x |
| `f128` | 2 | 106  | ~31  | ~10x |
| `f256` | 4 | 212  | ~63  | ~40-60x |
| `f512` | 8 | 424  | ~127 | ~150-200x |

All types are built from non-overlapping expansions of `f64` limbs, using error-free transformations:

$$\texttt{TwoSum}(a,b): \quad s + e = a + b \quad\text{exactly}$$

$$\texttt{TwoProd}(a,b): \quad p + e = a \times b \quad\text{exactly (via FMA)}$$

`f128` is a hand-tuned double-double type. `f256` and `f512` are `MultiFloat<4>` and `MultiFloat<8>` respectively -- a generic $N$-limb expansion with the same error-free primitives.

## Usage

### f128 (double-double, ~31 digits)

```rust
use hyperprec::f128;

let a = f128::from_f64(1.0);
let b = f128::from_f64(1e-20);
let c = a + b;
let d = c - a;
assert!((d.to_f64() - 1e-20).abs() < 1e-35);

// Parse from string to full precision
let pi: f128 = "3.14159265358979323846264338327950288".parse().unwrap();

// Transcendentals
let e = f128::ONE.exp();
let ln2 = f128::from_f64(2.0).ln();
let pow = f128::from_f64(2.0).pow(f128::from_f64(10.0)); // 1024.0
```

### f256 (quad-double, ~63 digits)

```rust
use hyperprec::f256;

let one = f256::ONE;
let seven = f256::from_f64(7.0);
let seventh = one / seven;
let back = seventh * seven;
// (back - one).limbs[0].abs() < 1e-60

// Constants at full precision
let pi = f256::pi();
let e = f256::e();
let ln2 = f256::ln2();
```

### f512 (octo-double, ~127 digits)

```rust
use hyperprec::f512;

let one = f512::ONE;
let seven = f512::from_f64(7.0);
let seventh = one / seven;
let back = seventh * seven;
// (back - one).limbs[0].abs() < 1e-100
```

### Linear algebra

All linear algebra operations are available for both `f128` and `MultiFloat<N>`:

```rust
use hyperprec::{f128, dot, gemv, gemm, gemm_atb, cholesky, solve_cholesky, jacobi_eigen};
use hyperprec::multifloat::{mf_dot, mf_gemv, mf_gemm, mf_gemm_atb, mf_cholesky, mf_cholesky_solve};

// f128 Cholesky solve
let n = 4;
let mut a = vec![f128::ZERO; n * n];
for i in 0..n {
    for j in 0..n {
        a[i * n + j] = f128::from_f64(1.0 / (i + j + 1) as f64);
    }
    a[i * n + i] = a[i * n + i] + f128::from_f64(1.0);
}
let a_copy = a.clone();
cholesky(&mut a, n).unwrap();
let b = vec![f128::ONE; n];
let x = solve_cholesky(&a, &b, n);

// GEMM: C = A * B (row-major, m x k * k x n -> m x n)
let mut c = vec![f128::ZERO; m * n];
gemm(&a_mat, &b_mat, &mut c, m, n, k);

// Eigenvalue decomposition (symmetric)
let (eigenvalues, eigenvectors) = jacobi_eigen(&sym_matrix, n, 200).unwrap();
```

## Supported operations

**Arithmetic:** `+`, `-`, `*`, `/`, `%`, `+=`, `-=`, `*=`, `/=`, unary `-`

**Mixed precision:** `f128 + f64`, `f128 * f64`, etc. (avoids unnecessary promotion). `MultiFloat<N> * f64`, `MultiFloat<N> / f64`.

**Math:** `abs`, `sqrt`, `recip`, `trunc`, `exp`, `ln`, `pow`

**Constants:** `ZERO`, `ONE`, `pi()`, `e()`, `ln2()` -- all computed to full $N$-limb precision

**Conversions:** `From<f64>`, `From<f32>`, `From<i32>`, `From<u32>`, `From<i64>`, `From<u64>`, `FromStr`, `Display`, `Debug`

**Iterator traits:** `Sum`, `Product`

**Comparisons:** `PartialEq`, `PartialOrd`

**Linear algebra (f128):** `dot`, `gemv`, `gemm`, `gemm_atb`, `cholesky`, `cholesky_blocked`, `forward_solve`, `backward_solve`, `solve_cholesky`, `matvec`, `cond_estimate`, `jacobi_eigen`

**Linear algebra (MultiFloat\<N\>):** `mf_dot`, `mf_gemv`, `mf_gemm`, `mf_gemm_atb`, `mf_cholesky`, `mf_cholesky_solve`

**Optional features:**
- `serde` -- lossless serialization/deserialization (JSON roundtrip preserves all limbs)
- `num-traits` -- `Zero`, `One`, `Num`, `Float`, `FromPrimitive`, `ToPrimitive`, `NumCast`

## Use cases

### Ill-conditioned linear algebra

The primary motivation. When $\kappa(\mathbf{A}) > 10^{13}$, f64 Cholesky loses all significant digits. f128 extends the working range to $\kappa \sim 10^{28}$; f256 to $\kappa \sim 10^{60}$.

### Overlap matrices in quantum chemistry

In Hartree-Fock and DFT, the Roothaan-Hall equation is a generalized eigenvalue problem:

$$\mathbf{F}\mathbf{C} = \mathbf{S}\mathbf{C}\boldsymbol{\varepsilon}$$

where $\mathbf{S}$ is the overlap matrix of atomic orbital basis functions:

$$S_{\mu\nu} = \langle \phi_\mu | \phi_\nu \rangle = \int \phi_\mu(\mathbf{r})\, \phi_\nu(\mathbf{r})\, d\mathbf{r}$$

The standard approach orthogonalizes the basis via $\mathbf{S} = \mathbf{L}\mathbf{L}^T$ (Cholesky) or $\mathbf{X} = \mathbf{S}^{-1/2}$ (canonical orthogonalization). Both require inverting or factoring $\mathbf{S}$.

When you use augmented or diffuse basis sets (aug-cc-pVDZ, aug-cc-pVTZ), nearby diffuse Gaussians overlap almost completely:

$$S_{\mu\nu} = \left(\frac{4\alpha_\mu\alpha_\nu}{(\alpha_\mu + \alpha_\nu)^2}\right)^{d/4} \exp\!\left(-\frac{\alpha_\mu\alpha_\nu}{\alpha_\mu + \alpha_\nu}|\mathbf{R}_\mu - \mathbf{R}_\nu|^2\right)$$

For diffuse exponents $\alpha \sim 0.01$, this approaches 1.0 for adjacent atoms, making $\mathbf{S}$ nearly singular:

| System | $N_\text{basis}$ | $\kappa(\mathbf{S})$ | f64 status |
|--------|:-:|:-:|:-:|
| Small molecule | ~100 | $10^4$ | Fine |
| Protein fragment | ~750 | $10^{13}$ | Cholesky fails |
| Periodic slab | ~2000 | $10^{20}$+ | Impossible |

**f64 has ~16 decimal digits.** When $\kappa(\mathbf{S}) > 10^{13}$, Cholesky loses all significant digits. f128 (~31 digits) handles the problem directly:

| $\kappa(\mathbf{S})$ | f64 | f128 |
|:-:|:-:|:-:|
| $10^{13}$ | Fails | $\|\mathbf{x} - \mathbf{x}_\text{true}\| \sim 10^{-18}$ |
| $10^{15}$ | Fails | $\|\mathbf{x} - \mathbf{x}_\text{true}\| \sim 10^{-14}$ |
| $10^{20}$ | Fails | $\|\mathbf{x} - \mathbf{x}_\text{true}\| \sim 10^{-8}$ |

For $\kappa > 10^{28}$, use f256. For $\kappa > 10^{60}$, use f512.

### Other applications

- **Geometric predicates** -- orientation tests, in-circle tests where sign correctness matters
- **Long-time numerical integration** -- ODE/PDE solvers where error accumulates over millions of steps
- **Number theory / computational mathematics** -- high-precision evaluation of special functions
- **Compensated summation** -- when Kahan summation is not enough

## Hilbert matrix demo

The Hilbert matrix $H_{ij} = 1/(i+j-1)$ is a classic ill-conditioned test. f64 Cholesky fails at $n=13$. f128 Cholesky with full-precision assembly:

```
  Hilbert  n=12   kappa=1.1e13   ||x-1||=9.89e-18   ||r||=1.18e-31
  Hilbert  n=13   kappa=1.8e14   ||x-1||=1.05e-16   ||r||=5.70e-32
  Hilbert  n=14   kappa=2.9e15   ||x-1||=5.10e-14   ||r||=5.59e-32
  Hilbert  n=15   kappa=4.7e16   ||x-1||=5.22e-12   ||r||=9.14e-32
  Hilbert  n=18   kappa=1.9e20   ||x-1||=5.46e-8    ||r||=1.09e-31
  Hilbert  n=20   kappa=4.9e22   ||x-1||=9.52e-5    ||r||=1.07e-31
```

Residual stays at $\sim 10^{-31}$ throughout -- the solver produces answers accurate to the limits of 31-digit arithmetic.

## Architecture

The crate provides two layers:

**`f128`** -- a hand-tuned double-double type (`hi: f64, lo: f64`). All arithmetic is `#[inline(always)]`. This is the fast path for when ~31 digits suffice.

**`MultiFloat<N>`** -- a generic $N$-limb non-overlapping expansion:

```rust
pub struct MultiFloat<const N: usize> {
    pub limbs: [f64; N],  // limbs[0] most significant, non-overlapping
}

pub type f256 = MultiFloat<4>;
pub type f512 = MultiFloat<8>;
```

Each limb captures error from the limb above. After every operation, the limb array is renormalized to maintain the non-overlapping invariant: $|\text{limbs}[i+1]| \leq \frac{1}{2}\,\text{ulp}(\text{limbs}[i])$.

Multiplication computes the full convolution of cross-products $a_i \cdot b_j$ for $i+j < N$ using error-free `two_prod`, accumulates into a $2N$ working array, and renormalizes down to $N$ output limbs.

Division uses Newton iteration for the reciprocal ($\lceil\log_2 N\rceil + 2$ steps), starting from an f64 seed. Each step doubles the number of correct bits.

Transcendentals (`exp`, `ln`, `pow`) use argument reduction + Taylor series (for `exp`) and Newton iteration (for `ln`), with iteration counts scaled to the precision level $N$.

## Crate structure

```
crates/
  quad/              # f128, f256, f512, MultiFloat<N>, linalg, SIMD kernels
  compensated-sum/   # Neumaier compensated summation for f64 and f128
  solver/            # solve_spd() -- auto-selects precision strategy
  overlap/           # Mixed-precision overlap integral assembly
  demo/              # cargo run --release -p demo
```

## Requirements

- **Rust 2024 edition** (workspace version 0.3.0)
- **FMA support** -- all modern x86-64 (Haswell+) and ARM (all AArch64) processors. The `two_prod` primitive requires hardware FMA for correctness.
- **rayon** -- used for parallel GEMV, GEMM, and Cholesky
- Optional: `serde`, `num-traits` via feature flags

```toml
[dependencies]
hyperprec = "0.3"

# With optional features:
hyperprec = { version = "0.3", features = ["serde", "num-traits"] }
```

## References

- T.J. Dekker, "A floating-point technique for extending the available precision" (1971)
- D.E. Knuth, *The Art of Computer Programming*, Vol 2 -- error-free transformations
- Y. Hida, X.S. Li, D.H. Bailey, "Algorithms for Quad-Double Precision Floating Point Arithmetic" (2001)
- [QD Library](https://www.davidhbailey.com/dhbsoftware/) -- the original quad-double C++/Fortran implementation

## License

MIT
