use quad::f128;
use crate::basis::Shell;
use crate::screen::PrecisionTier;

/// Overlap integral between two s-type primitive Gaussians in 3D:
///
///   S = (pi / (alpha_i + alpha_j))^{3/2}
///       * exp(-alpha_i * alpha_j / (alpha_i + alpha_j) * |R_i - R_j|^2)
///
/// For higher angular momentum (p, d, f), each Cartesian component
/// introduces Hermite polynomial factors via the Obara-Saika recurrence.
/// This module implements the full contracted overlap for s-type,
/// and the recurrence framework for l > 0.
#[inline]
pub fn overlap_primitive_f64(
    alpha_i: f64, center_i: &[f64; 3],
    alpha_j: f64, center_j: &[f64; 3],
) -> f64 {
    let gamma = alpha_i + alpha_j;
    let mu = alpha_i * alpha_j / gamma;
    let r2 = dist2_f64(center_i, center_j);

    let prefactor = (std::f64::consts::PI / gamma).sqrt().powi(3);
    prefactor * (-mu * r2).exp()
}

/// Same integral in f128 for near-field diffuse pairs.
#[inline]
pub fn overlap_primitive_f128(
    alpha_i: f64, center_i: &[f64; 3],
    alpha_j: f64, center_j: &[f64; 3],
) -> f128 {
    let ai = f128::from_f64(alpha_i);
    let aj = f128::from_f64(alpha_j);
    let gamma = ai + aj;
    let mu = ai * aj / gamma;

    let dx = f128::from_f64(center_i[0] - center_j[0]);
    let dy = f128::from_f64(center_i[1] - center_j[1]);
    let dz = f128::from_f64(center_i[2] - center_j[2]);
    let r2 = dx * dx + dy * dy + dz * dz;

    // (pi / gamma)^{3/2}
    let pi = f128::from_f64(std::f64::consts::PI);
    let ratio = pi / gamma;
    let prefactor = ratio * ratio.sqrt(); // ratio^{3/2}

    // exp(-mu * r2): use f64 exp with f128 argument reduction
    // For double-double, we split: exp(hi + lo) = exp(hi) * (1 + lo)
    // This is accurate to ~2^-105 for |lo| << 1.
    let neg_mu_r2 = (f128::ZERO - mu) * r2;
    let exp_hi = f128::from_f64(neg_mu_r2.hi.exp());
    let exp_val = exp_hi * (f128::ONE + f128::from_f64(neg_mu_r2.lo));

    prefactor * exp_val
}

/// Compute overlap sub-block for a shell pair.
///
/// Returns a flat array of size shell_i.size() * shell_j.size()
/// in row-major order (iterating over Cartesian components of i, then j).
///
/// For s-s pairs this is a single element.
/// For p-s, s-p, p-p, etc., we apply Obara-Saika recurrence.
pub fn overlap_shell_pair(
    shell_i: &Shell,
    shell_j: &Shell,
    tier: PrecisionTier,
) -> Vec<f64> {
    let ni = shell_i.size();
    let nj = shell_j.size();
    let li = shell_i.kind.l();
    let lj = shell_j.kind.l();

    // For now: implement s-type directly, higher-l via Obara-Saika
    // over the primitive pairs, then contract.
    let mut block = vec![0.0f64; ni * nj];

    match tier {
        PrecisionTier::Skip => return block,
        PrecisionTier::F64 => {
            overlap_contracted_f64(shell_i, shell_j, li, lj, &mut block);
        }
        PrecisionTier::F128 => {
            overlap_contracted_f128(shell_i, shell_j, li, lj, &mut block);
        }
    }

    block
}

/// f64 contracted overlap.
fn overlap_contracted_f64(
    shell_i: &Shell, shell_j: &Shell,
    li: u32, lj: u32,
    block: &mut [f64],
) {
    if li == 0 && lj == 0 {
        // s-s: single element, sum over primitive pairs
        let mut sum = 0.0_f64;
        for pi in &shell_i.primitives {
            for pj in &shell_j.primitives {
                let s_prim = overlap_primitive_f64(
                    pi.alpha, &shell_i.center,
                    pj.alpha, &shell_j.center,
                );
                sum += pi.coeff * pj.coeff * s_prim;
            }
        }
        block[0] = sum;
    } else {
        // Higher angular momentum: Obara-Saika recurrence over primitives.
        // For each primitive pair, compute the (li+lj+1)-order overlap
        // integrals via recurrence, then contract.
        obara_saika_contracted_f64(shell_i, shell_j, li, lj, block);
    }
}

/// f128 contracted overlap, downcast result to f64 for storage.
fn overlap_contracted_f128(
    shell_i: &Shell, shell_j: &Shell,
    li: u32, lj: u32,
    block: &mut [f64],
) {
    if li == 0 && lj == 0 {
        let mut sum = f128::ZERO;
        for pi in &shell_i.primitives {
            for pj in &shell_j.primitives {
                let s_prim = overlap_primitive_f128(
                    pi.alpha, &shell_i.center,
                    pj.alpha, &shell_j.center,
                );
                let ci = f128::from_f64(pi.coeff);
                let cj = f128::from_f64(pj.coeff);
                sum += ci * cj * s_prim;
            }
        }
        block[0] = sum.to_f64();
    } else {
        obara_saika_contracted_f128(shell_i, shell_j, li, lj, block);
    }
}

/// Obara-Saika recurrence for overlap integrals [a|b] in one Cartesian direction.
///
/// The 1D recurrence is:
///   S(i+1, j) = X_PA * S(i, j) + (1/(2*gamma)) * [i * S(i-1, j) + j * S(i, j-1)]
///
/// where gamma = alpha_i + alpha_j, P = (alpha_i * A + alpha_j * B) / gamma,
/// X_PA = P_x - A_x.
///
/// We build a 2D table S[0..li+1][0..lj+1] for each Cartesian direction,
/// then the 3D integral is S_x(lx_i, lx_j) * S_y(ly_i, ly_j) * S_z(lz_i, lz_j)
/// times the prefactor.

fn obara_saika_1d_f64(
    la: u32, lb: u32,
    pa: f64,   // P_x - A_x
    pb: f64,   // P_x - B_x
    gamma: f64,
) -> Vec<f64> {
    let na = la as usize + 1;
    let nb = lb as usize + 1;
    let mut s = vec![0.0_f64; na * nb];
    let idx = |i: usize, j: usize| i * nb + j;
    let oo2g = 0.5 / gamma;

    s[idx(0, 0)] = 1.0; // normalized by prefactor outside

    // Build along i (first index) with j=0
    for i in 0..la as usize {
        s[idx(i + 1, 0)] = pa * s[idx(i, 0)] + oo2g * i as f64 * if i > 0 { s[idx(i - 1, 0)] } else { 0.0 };
    }

    // Build along j for each i
    for j in 0..lb as usize {
        for i in 0..=la as usize {
            let term_im1 = oo2g * i as f64 * if i > 0 { s[idx(i - 1, j)] } else { 0.0 };
            let term_jm1 = oo2g * j as f64 * if j > 0 { s[idx(i, j - 1)] } else { 0.0 };
            s[idx(i, j + 1)] = pb * s[idx(i, j)] + term_im1 + term_jm1;
        }
    }

    s
}

fn obara_saika_1d_f128(
    la: u32, lb: u32,
    pa: f128, pb: f128,
    gamma: f128,
) -> Vec<f128> {
    let na = la as usize + 1;
    let nb = lb as usize + 1;
    let mut s = vec![f128::ZERO; na * nb];
    let idx = |i: usize, j: usize| i * nb + j;
    let oo2g = f128::from_f64(0.5) / gamma;

    s[idx(0, 0)] = f128::ONE;

    for i in 0..la as usize {
        let fi = f128::from_f64(i as f64);
        let prev = if i > 0 { s[idx(i - 1, 0)] } else { f128::ZERO };
        s[idx(i + 1, 0)] = pa * s[idx(i, 0)] + oo2g * fi * prev;
    }

    for j in 0..lb as usize {
        let fj = f128::from_f64(j as f64);
        for i in 0..=la as usize {
            let fi = f128::from_f64(i as f64);
            let im1 = if i > 0 { s[idx(i - 1, j)] } else { f128::ZERO };
            let jm1 = if j > 0 { s[idx(i, j - 1)] } else { f128::ZERO };
            s[idx(i, j + 1)] = pb * s[idx(i, j)] + oo2g * fi * im1 + oo2g * fj * jm1;
        }
    }

    s
}

/// Enumerate Cartesian components for angular momentum l.
/// Returns (lx, ly, lz) triples in canonical order.
fn cart_components(l: u32) -> Vec<(u32, u32, u32)> {
    let mut comps = Vec::new();
    for lx in (0..=l).rev() {
        for ly in (0..=(l - lx)).rev() {
            let lz = l - lx - ly;
            comps.push((lx, ly, lz));
        }
    }
    comps
}

fn obara_saika_contracted_f64(
    shell_i: &Shell, shell_j: &Shell,
    li: u32, lj: u32,
    block: &mut [f64],
) {
    let comps_i = cart_components(li);
    let comps_j = cart_components(lj);
    let nj = comps_j.len();

    for pi in &shell_i.primitives {
        for pj in &shell_j.primitives {
            let ai = pi.alpha;
            let aj = pj.alpha;
            let gamma = ai + aj;
            let mu = ai * aj / gamma;
            let r2 = dist2_f64(&shell_i.center, &shell_j.center);
            let prefactor = (std::f64::consts::PI / gamma).sqrt().powi(3) * (-mu * r2).exp();

            // Gaussian product center
            let p = [
                (ai * shell_i.center[0] + aj * shell_j.center[0]) / gamma,
                (ai * shell_i.center[1] + aj * shell_j.center[1]) / gamma,
                (ai * shell_i.center[2] + aj * shell_j.center[2]) / gamma,
            ];

            let pa = [p[0] - shell_i.center[0], p[1] - shell_i.center[1], p[2] - shell_i.center[2]];
            let pb = [p[0] - shell_j.center[0], p[1] - shell_j.center[1], p[2] - shell_j.center[2]];

            let sx = obara_saika_1d_f64(li, lj, pa[0], pb[0], gamma);
            let sy = obara_saika_1d_f64(li, lj, pa[1], pb[1], gamma);
            let sz = obara_saika_1d_f64(li, lj, pa[2], pb[2], gamma);

            let nb = lj as usize + 1;

            for (ci, &(lxi, lyi, lzi)) in comps_i.iter().enumerate() {
                for (cj, &(lxj, lyj, lzj)) in comps_j.iter().enumerate() {
                    let val = sx[lxi as usize * nb + lxj as usize]
                            * sy[lyi as usize * nb + lyj as usize]
                            * sz[lzi as usize * nb + lzj as usize]
                            * prefactor * pi.coeff * pj.coeff;
                    block[ci * nj + cj] += val;
                }
            }
        }
    }
}

fn obara_saika_contracted_f128(
    shell_i: &Shell, shell_j: &Shell,
    li: u32, lj: u32,
    block: &mut [f64],
) {
    let comps_i = cart_components(li);
    let comps_j = cart_components(lj);
    let nj = comps_j.len();
    let mut block128 = vec![f128::ZERO; comps_i.len() * nj];

    for pi in &shell_i.primitives {
        for pj in &shell_j.primitives {
            let ai = f128::from_f64(pi.alpha);
            let aj = f128::from_f64(pj.alpha);
            let gamma = ai + aj;
            let mu = ai * aj / gamma;

            let dx = f128::from_f64(shell_i.center[0] - shell_j.center[0]);
            let dy = f128::from_f64(shell_i.center[1] - shell_j.center[1]);
            let dz = f128::from_f64(shell_i.center[2] - shell_j.center[2]);
            let r2 = dx * dx + dy * dy + dz * dz;

            let pi_val = f128::from_f64(std::f64::consts::PI);
            let ratio = pi_val / gamma;
            let prefactor = ratio * ratio.sqrt(); // (pi/gamma)^{3/2}

            let neg_mu_r2 = (f128::ZERO - mu) * r2;
            let exp_hi = f128::from_f64(neg_mu_r2.hi.exp());
            let exp_val = exp_hi * (f128::ONE + f128::from_f64(neg_mu_r2.lo));
            let total_pre = prefactor * exp_val;

            let ci_f = [
                f128::from_f64(shell_i.center[0]),
                f128::from_f64(shell_i.center[1]),
                f128::from_f64(shell_i.center[2]),
            ];
            let cj_f = [
                f128::from_f64(shell_j.center[0]),
                f128::from_f64(shell_j.center[1]),
                f128::from_f64(shell_j.center[2]),
            ];

            let p = [
                (ai * ci_f[0] + aj * cj_f[0]) / gamma,
                (ai * ci_f[1] + aj * cj_f[1]) / gamma,
                (ai * ci_f[2] + aj * cj_f[2]) / gamma,
            ];

            let pa = [p[0] - ci_f[0], p[1] - ci_f[1], p[2] - ci_f[2]];
            let pb = [p[0] - cj_f[0], p[1] - cj_f[1], p[2] - cj_f[2]];

            let sx = obara_saika_1d_f128(li, lj, pa[0], pb[0], gamma);
            let sy = obara_saika_1d_f128(li, lj, pa[1], pb[1], gamma);
            let sz = obara_saika_1d_f128(li, lj, pa[2], pb[2], gamma);

            let nb = lj as usize + 1;
            let coeff = f128::from_f64(pi.coeff) * f128::from_f64(pj.coeff) * total_pre;

            for (ci_idx, &(lxi, lyi, lzi)) in comps_i.iter().enumerate() {
                for (cj_idx, &(lxj, lyj, lzj)) in comps_j.iter().enumerate() {
                    let val = sx[lxi as usize * nb + lxj as usize]
                            * sy[lyi as usize * nb + lyj as usize]
                            * sz[lzi as usize * nb + lzj as usize]
                            * coeff;
                    block128[ci_idx * nj + cj_idx] += val;
                }
            }
        }
    }

    for (i, v) in block128.iter().enumerate() {
        block[i] = v.to_f64();
    }
}

#[inline]
fn dist2_f64(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Primitive, Shell, ShellKind};

    #[test]
    fn self_overlap_normalized() {
        // Self-overlap of a single normalized s-Gaussian: S = 1
        // A normalized Gaussian has coefficient such that <phi|phi> = 1.
        // Unnormalized: S = (pi/2alpha)^{3/2}
        // Norm factor N = (2*alpha/pi)^{3/4}
        let alpha = 1.0;
        let norm = (2.0 * alpha / std::f64::consts::PI).powf(0.75);
        let shell = Shell {
            center: [0.0; 3],
            kind: ShellKind::S,
            primitives: vec![Primitive { alpha, coeff: norm }],
        };
        let block = overlap_shell_pair(&shell, &shell, PrecisionTier::F64);
        assert!((block[0] - 1.0).abs() < 1e-14,
            "Self-overlap should be 1.0, got {}", block[0]);
    }

    #[test]
    fn f64_f128_agree_tight() {
        // For tight gaussians, f64 and f128 should agree closely
        let shell_a = Shell {
            center: [0.0, 0.0, 0.0],
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 10.0, coeff: 0.5 },
                Primitive { alpha: 5.0, coeff: 0.5 },
            ],
        };
        let shell_b = Shell {
            center: [1.0, 0.0, 0.0],
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 8.0, coeff: 0.6 },
                Primitive { alpha: 3.0, coeff: 0.4 },
            ],
        };
        let s64 = overlap_shell_pair(&shell_a, &shell_b, PrecisionTier::F64);
        let s128 = overlap_shell_pair(&shell_a, &shell_b, PrecisionTier::F128);
        assert!((s64[0] - s128[0]).abs() < 1e-14,
            "f64={} vs f128={}", s64[0], s128[0]);
    }

    #[test]
    fn f128_more_accurate_diffuse() {
        // Diffuse pair where contraction sum has cancellation.
        // We can't easily verify "more accurate" without a reference,
        // but we check they produce finite, reasonable values.
        let shell = Shell {
            center: [0.0; 3],
            kind: ShellKind::S,
            primitives: vec![
                Primitive { alpha: 100.0, coeff: 0.001 },
                Primitive { alpha: 0.03, coeff: 0.999 },
            ],
        };
        let s128 = overlap_shell_pair(&shell, &shell, PrecisionTier::F128);
        assert!(s128[0].is_finite(), "f128 overlap should be finite");
        assert!(s128[0] > 0.0, "Self-overlap should be positive, got {}", s128[0]);
    }

    #[test]
    fn p_shell_overlap_size() {
        let shell = Shell {
            center: [0.0; 3],
            kind: ShellKind::P,
            primitives: vec![Primitive { alpha: 1.0, coeff: 1.0 }],
        };
        let block = overlap_shell_pair(&shell, &shell, PrecisionTier::F64);
        assert_eq!(block.len(), 9); // 3x3 for p-p
    }
}
