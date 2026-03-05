/// A single primitive Gaussian: c * exp(-alpha * |r - R|^2)
#[derive(Clone, Debug)]
pub struct Primitive {
    pub alpha: f64,
    pub coeff: f64,
}

/// Angular momentum type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellKind {
    S, P, D, F,
}

impl ShellKind {
    pub fn l(self) -> u32 {
        match self {
            ShellKind::S => 0,
            ShellKind::P => 1,
            ShellKind::D => 2,
            ShellKind::F => 3,
        }
    }

    /// Number of Cartesian components: (l+1)(l+2)/2
    pub fn n_cart(self) -> usize {
        let l = self.l() as usize;
        (l + 1) * (l + 2) / 2
    }
}

/// A contracted shell: fixed center, angular momentum, list of primitives.
#[derive(Clone, Debug)]
pub struct Shell {
    pub center: [f64; 3],
    pub kind: ShellKind,
    pub primitives: Vec<Primitive>,
}

impl Shell {
    /// Minimum exponent in the contraction (the most diffuse primitive).
    pub fn alpha_min(&self) -> f64 {
        self.primitives.iter().map(|p| p.alpha).fold(f64::INFINITY, f64::min)
    }

    /// Maximum exponent in the contraction (the tightest primitive).
    pub fn alpha_max(&self) -> f64 {
        self.primitives.iter().map(|p| p.alpha).fold(0.0_f64, f64::max)
    }

    /// Basis function offset in the full AO matrix.
    pub fn size(&self) -> usize {
        self.kind.n_cart()
    }
}

/// Collection of shells forming a basis set for a molecule.
#[derive(Clone, Debug)]
pub struct BasisSet {
    pub shells: Vec<Shell>,
}

impl BasisSet {
    pub fn n_ao(&self) -> usize {
        self.shells.iter().map(|s| s.size()).sum()
    }

    /// Compute the AO offset for each shell.
    pub fn shell_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.shells.len());
        let mut off = 0;
        for s in &self.shells {
            offsets.push(off);
            off += s.size();
        }
        offsets
    }
}
