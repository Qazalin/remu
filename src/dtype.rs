use half::f16;
use num_traits::float::FloatCore;

pub trait IEEEClass<T> {
    fn exponent(&self) -> T;
}
impl IEEEClass<u32> for f32 {
    fn exponent(&self) -> u32 {
        (self.to_bits() & 0b01111111100000000000000000000000) >> 23
    }
}
impl IEEEClass<u16> for f16 {
    fn exponent(&self) -> u16 {
        (self.to_bits() & 0b0111110000000000) >> 10
    }
}
impl IEEEClass<u64> for f64 {
    fn exponent(&self) -> u64 {
        (self.to_bits() & 0b0111111111110000000000000000000000000000000000000000000000000000) >> 52
    }
}

pub trait VOPModifier<T> {
    fn negate(&self, pos: usize, modifier: usize) -> T;
    fn absolute(&self, pos: usize, modifier: usize) -> T;
}
impl<T> VOPModifier<T> for T
where
    T: FloatCore,
{
    fn negate(&self, pos: usize, modifier: usize) -> T {
        match (modifier >> pos) & 1 {
            1 => match self.is_zero() {
                true => *self,
                false => -*self,
            },
            _ => *self,
        }
    }
    fn absolute(&self, pos: usize, modifier: usize) -> T {
        match (modifier >> pos) & 1 {
            1 => self.abs(),
            _ => *self,
        }
    }
}

pub fn extract_mantissa(x: f64) -> f64 {
    if x.is_infinite() || x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let mantissa_mask: u64 = 0x000FFFFFFFFFFFFF;
    let bias: u64 = 1023;
    let normalized_mantissa_bits = (bits & mantissa_mask) | ((bias - 1) << 52);
    return f64::from_bits(normalized_mantissa_bits);
}
pub fn ldexp(x: f64, exp: i32) -> f64 {
    x * 2f64.powi(exp)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_extract_mantissa() {
        assert_eq!(extract_mantissa(2.0f64), 0.5);
    }

    #[test]
    fn test_normal_exponent() {
        assert_eq!(2.5f32.exponent(), 128);
        assert_eq!(1.17549435e-38f32.exponent(), 1);
        assert_eq!(f32::INFINITY.exponent(), 255);
        assert_eq!(f32::NEG_INFINITY.exponent(), 255);
    }

    #[test]
    fn test_denormal_exponent() {
        assert_eq!(1.0e-40f32.exponent(), 0);
        assert_eq!(1.0e-42f32.exponent(), 0);
        assert_eq!(1.0e-44f32.exponent(), 0);
        assert_eq!((1.17549435e-38f32 / 2.0).exponent(), 0);
    }

    #[test]
    fn test_normal_exponent_f16() {
        assert_eq!(f16::from_f32(3.14f32).exponent(), 16);
        assert_eq!(f16::NEG_INFINITY.exponent(), 31);
        assert_eq!(f16::INFINITY.exponent(), 31);
    }
}

#[cfg(test)]
mod test_modifiers {
    use super::*;

    #[test]
    fn test_neg() {
        assert_eq!(0.3_f32.negate(0, 0b001), -0.3_f32);
        assert_eq!(0.3_f32.negate(1, 0b010), -0.3_f32);
        assert_eq!(0.3_f32.negate(2, 0b100), -0.3_f32);
        assert_eq!(0.3_f32.negate(0, 0b110), 0.3_f32);
        assert_eq!(0.3_f32.negate(1, 0b010), -0.3_f32);
        assert_eq!(0.0_f32.negate(0, 0b001).to_bits(), 0);
    }
}
