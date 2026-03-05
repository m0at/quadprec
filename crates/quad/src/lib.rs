mod dd;
mod linalg;
pub mod simd;
pub mod multifloat;

#[cfg(feature = "num-traits")]
mod num_impl;

pub use dd::{f128, ParseF128Error};
pub use linalg::{
    dot, gemv, cholesky, cholesky_blocked, forward_solve, backward_solve, solve_cholesky,
    matvec, cond_estimate, CholeskyError, gemm, gemm_atb, jacobi_eigen,
};
pub use multifloat::{MultiFloat, f256, f512};
