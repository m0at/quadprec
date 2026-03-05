//! Apple Accelerate BLAS/LAPACK bindings for f64 Cholesky.
//!
//! LAPACK uses column-major (Fortran) order. Our solver stores matrices in
//! row-major order. For a symmetric matrix A, row-major A is the same as
//! column-major A^T. So row-major upper triangle == column-major lower
//! triangle. We pass uplo='U' to dpotrf/dpotrs which tells LAPACK to read
//! the upper triangle of the column-major layout, which corresponds to the
//! lower triangle of our row-major layout — but for SPD matrices A = A^T,
//! so both are valid factorizations of A.

unsafe extern "C" {
    unsafe fn dpotrf_(uplo: *const u8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
    unsafe fn dpotrs_(
        uplo: *const u8,
        n: *const i32,
        nrhs: *const i32,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
}

/// In-place Cholesky factorization using Accelerate's dpotrf.
///
/// `a` is an n-by-n matrix in row-major order. On success the factored form
/// is stored in-place (upper triangle in row-major = lower triangle in
/// column-major, which is what dpotrf with uplo='U' produces).
pub fn cholesky_accelerate(a: &mut [f64], n: usize) -> Result<(), i32> {
    let n_i32 = n as i32;
    let mut info: i32 = 0;
    let uplo = b'U';
    unsafe {
        dpotrf_(&uplo, &n_i32, a.as_mut_ptr(), &n_i32, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err(info)
    }
}

/// Solve A*x = b given the Cholesky factorization from `cholesky_accelerate`.
///
/// `l` is the factored matrix (n-by-n, row-major). `b` is overwritten with
/// the solution x.
pub fn solve_cholesky_accelerate(l: &[f64], b: &mut [f64], n: usize) {
    let n_i32 = n as i32;
    let nrhs: i32 = 1;
    let mut info: i32 = 0;
    let uplo = b'U';
    unsafe {
        dpotrs_(
            &uplo,
            &n_i32,
            &nrhs,
            l.as_ptr(),
            &n_i32,
            b.as_mut_ptr(),
            &n_i32,
            &mut info,
        );
    }
    assert_eq!(info, 0, "dpotrs failed with info={info}");
}
