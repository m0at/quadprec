use std::fmt;
use std::str::FromStr;
use super::core::MultiFloat;

// --- Display ---

impl<const N: usize> fmt::Display for MultiFloat<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prec = N * 16;
        let approx = if N >= 2 {
            self.limbs[0] + self.limbs[1]
        } else {
            self.limbs[0]
        };
        write!(f, "{:.prec$e}", approx, prec = prec)
    }
}

// --- FromStr ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseMultiFloatError;

impl fmt::Display for ParseMultiFloatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid MultiFloat literal")
    }
}

impl std::error::Error for ParseMultiFloatError {}

impl<const N: usize> FromStr for MultiFloat<N> {
    type Err = ParseMultiFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err(ParseMultiFloatError);
        }

        let mut chars = s.as_bytes();
        let negative = match chars[0] {
            b'-' => { chars = &chars[1..]; true }
            b'+' => { chars = &chars[1..]; false }
            _ => false,
        };
        if chars.is_empty() {
            return Err(ParseMultiFloatError);
        }

        // Split on 'e' or 'E'
        let (mantissa, exp_part) = {
            let mut split = None;
            for (i, &c) in chars.iter().enumerate() {
                if c == b'e' || c == b'E' {
                    split = Some(i);
                    break;
                }
            }
            match split {
                Some(i) => (&chars[..i], Some(&chars[i + 1..])),
                None => (chars, None),
            }
        };
        if mantissa.is_empty() {
            return Err(ParseMultiFloatError);
        }

        let ten = Self::from_f64(10.0);
        let mut acc = Self::ZERO;
        let mut saw_dot = false;
        let mut any_digit = false;
        let mut pow10_neg = Self::from_f64(1.0);

        for &c in mantissa {
            if c == b'.' {
                if saw_dot {
                    return Err(ParseMultiFloatError);
                }
                saw_dot = true;
                continue;
            }
            if c < b'0' || c > b'9' {
                return Err(ParseMultiFloatError);
            }
            any_digit = true;
            let digit = Self::from_f64((c - b'0') as f64);
            if !saw_dot {
                acc = acc * ten + digit;
            } else {
                pow10_neg = pow10_neg / ten;
                acc = acc + digit * pow10_neg;
            }
        }
        if !any_digit {
            return Err(ParseMultiFloatError);
        }

        if let Some(exp_bytes) = exp_part {
            if exp_bytes.is_empty() {
                return Err(ParseMultiFloatError);
            }
            let exp_str = std::str::from_utf8(exp_bytes).map_err(|_| ParseMultiFloatError)?;
            let exp: i32 = exp_str.parse().map_err(|_| ParseMultiFloatError)?;
            if exp > 0 {
                for _ in 0..exp {
                    acc = acc * ten;
                }
            } else if exp < 0 {
                for _ in 0..(-exp) {
                    acc = acc / ten;
                }
            }
        }

        if negative {
            acc = -acc;
        }
        Ok(acc)
    }
}
