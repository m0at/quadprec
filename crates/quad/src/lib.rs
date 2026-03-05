mod dd;
mod linalg;

pub use dd::f128;
pub use linalg::{
    dot, gemv, cholesky, forward_solve, backward_solve, solve_cholesky,
    matvec, cond_estimate, CholeskyError,
};
