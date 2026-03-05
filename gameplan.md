# Gameplan: Mixed-Precision SCF Pipeline via Rust + Rayon + ANE

## Problem Statement

Overlap matrix condition numbers kappa(S) in augmented Gaussian basis sets (e.g. aug-cc-pVDZ) exceed the inverse of 64-bit machine epsilon (~10^16) for systems beyond ~750 atoms. GPU hardware is trending *away* from high-precision support (Nvidia reduced f64 TFLOPS from Hopper to Blackwell). Scientific computing needs a heterogeneous compute strategy that routes work to the right precision/hardware tier.

---

## Architecture: Three-Tier Precision Pipeline

```
Tier 0 (ANE / float16)          Tier 1 (CPU / float64)         Tier 2 (CPU / float128-emu)
+-----------------------+       +----------------------+       +------------------------+
| ML surrogate forces   |       | Coulomb/exchange     |       | Overlap matrix S       |
| Density fitting       |       | integrals            |       | Eigensolve of S        |
| Preconditioner gen    |       | Fock matrix assembly  |       | Compensated summation  |
| Initial guess (SAD)   |       | Density matrix P     |       | Iterative refinement   |
+-----------------------+       +----------------------+       +------------------------+
         |                               |                               |
         +---------- Rayon threadpool orchestrates all tiers ------------+
```

---

## Swarm Subtasks

Each subtask is independently workable. Dependencies noted where they exist.

### SUB-0: Quad-Precision Core in Rust

**Owner:** numerics lead
**Goal:** Software-emulated float128 (double-double or quad-double) arithmetic library in Rust, optimized for linear algebra primitives.

* Implement `f128` type using double-double representation (Dekker/Knuth TwoSum + TwoProd)
* Benchmark against `qd` (C++ quad-double library) and GCC's `__float128`
* Target operations: dot product, matrix-vector multiply, Cholesky factorization
* Parallelize inner loops with Rayon; the key insight is that double-double arithmetic is ~4x the FLOP count of f64 but fully CPU-friendly (no GPU needed)
* Deliverable: `crate::quad` with BLAS-like API

### SUB-1: Compensated Summation Kernels

**Owner:** numerics lead (can run parallel with SUB-0)
**Goal:** Kahan/Neumaier/pairwise summation routines that keep error below O(epsilon) instead of O(n * epsilon) for large accumulations.

* Implement for both f64 and f128-emu
* Critical path: overlap matrix element computation involves summing many Gaussian primitive contributions
* Rayon-parallel reduce with per-thread compensated accumulators, then compensated merge
* Deliverable: `crate::compensated_sum` with Rayon-aware reduce

### SUB-2: ANE Backend via Reverse-Engineered API

**Owner:** hardware/systems lead (your ANEtransformers repo)
**Goal:** Wrap ANE dispatch for float16 matrix operations usable from Rust via FFI.

* Stabilize the probed ANE API surface from your repo into a Rust FFI crate
* Expose: matmul, elementwise ops, activation functions (enough for ML surrogate inference)
* Key constraint: ANE has no error correction, so it can only be used for tasks where 1-2 ULP error in float16 is acceptable
* Benchmark throughput vs. Metal GPU compute for equivalent operations
* Deliverable: `crate::ane_backend` with safe Rust wrappers

### SUB-3: ML Surrogate for Initial Guess / Preconditioner

**Owner:** ML/chemistry lead
**Depends on:** SUB-2 (for ANE dispatch)
**Goal:** Train a lightweight model that produces a good initial density matrix guess or preconditioner for the SCF, runnable on ANE.

* SchNet/PaiNN-style architecture quantized to float16 for ANE inference
* Training data: converged density matrices from small-system DFT runs
* Output: initial density matrix P_0 that reduces SCF iteration count by 3-5x
* Secondary output: approximate inverse overlap matrix S^{-1} as preconditioner (this is where ANE contributes to the precision problem indirectly)
* Deliverable: trained model + ANE inference pipeline

### SUB-4: Overlap Matrix Assembly with Mixed Precision

**Owner:** chemistry/numerics lead
**Depends on:** SUB-0, SUB-1
**Goal:** Assemble the overlap matrix S using quad-precision where needed, f64 elsewhere.

* Screen basis function pairs by distance: well-separated pairs contribute negligibly and can use f64
* Near-field / diffuse function pairs (the ones that cause ill-conditioning) get f128-emu treatment
* Rayon work-stealing over shell pairs
* Condition number monitoring: compute kappa(S) estimate cheaply (e.g. via power iteration on S^{-1}S) after assembly
* Deliverable: `assemble_overlap_mixed()` function

### SUB-5: Iterative Refinement SCF Loop

**Owner:** numerics/chemistry lead
**Depends on:** SUB-0, SUB-3, SUB-4
**Goal:** SCF loop that uses iterative refinement to recover precision lost in f64 Fock matrix operations.

* Core idea: solve F*C = S*C*E in f64, then compute residual r = F*C - S*C*E in f128-emu, solve correction in f128-emu
* This lets the bulk of the Fock matrix work stay in f64 (or even on GPU) while only the correction step needs quad precision
* DIIS extrapolation with compensated summation for the error vector accumulation
* Deliverable: `scf_iterative_refine()` main loop

### SUB-6: Rayon Orchestration Layer

**Owner:** systems lead
**Depends on:** all above
**Goal:** Top-level scheduler that routes work across ANE, CPU-f64, and CPU-f128 tiers.

* Rayon threadpool configuration: pin precision-critical threads to performance cores, let ANE dispatch happen asynchronously
* Memory layout: overlap/Fock matrices in cache-friendly blocked format, with blocks tagged by precision tier
* Profiling harness: instrument which fraction of wall time is spent in each precision tier
* Deliverable: `crate::scheduler` with configurable tier routing

---

## Milestones

| Phase | Target                 | Validates                                                                                      |
| ----- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| M0    | SUB-0 + SUB-1 complete | Quad-precision linear algebra works and is < 6x slower than f64 BLAS                           |
| M1    | SUB-2 complete         | ANE dispatch from Rust is stable, throughput is measured                                       |
| M2    | SUB-4 complete         | Overlap matrix for 1430-atom graphene cutout has kappa(S) < 10^12 effectively (via mixed prec) |
| M3    | SUB-3 + SUB-5 complete | Full SCF converges for aug-cc-pVDZ on 1430-atom system                                         |
| M4    | SUB-6 + integration    | End-to-end benchmark: wall time vs. pure-f64 (expect 2-4x overhead, not 10-20x of naive quad)  |

---

## Key Risks

**ANE API stability:** The reverse-engineered API could break with any macOS update. Mitigation: abstract behind a trait so Metal compute shaders can substitute.

**Double-double precision may not suffice:** For kappa(S) > 10^16, double-double gives ~32 decimal digits which is enough. But for truly pathological cases (kappa > 10^30), you'd need quad-double. Keep the `f128` type generic over representation width.

**Rust ecosystem gaps:** No mature quantum chemistry integral library in Rust. You'll need to either FFI to libint2/libcint or rewrite the critical integral kernels. FFI is pragmatic for M0-M3; native Rust is the long-term play.

**Benchmark validity:** Comparing against CP2K/MOLOPT (which sidesteps the problem by using better-conditioned basis sets) is apples-to-oranges. The real comparison is: can you run aug-cc-pVDZ on 2790 atoms where nobody else can?

---

## Why This Might Actually Work

The post's framing ("scientists are screwed") assumes you need the GPU for everything. But the precision-critical path (overlap matrix, eigensolve) is a small fraction of total SCF wall time. If you can keep it on CPU with smart parallelism (Rayon) and only pay the quad-precision penalty on that slice, while offloading the precision-tolerant bulk to ANE/GPU, the total overhead stays manageable. The mixed-precision iterative refinement approach (SUB-5) is the key insight: do the expensive work in f64, correct in f128, converge in fewer iterations than pure-f64 would (because pure-f64 diverges for these systems).
