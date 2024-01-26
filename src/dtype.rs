use half::f16;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_exponent() {
        assert_eq!(2.5f32.exponent(), 128);
        assert_eq!(1.17549435e-38.exponent(), 1);
        assert_eq!(f32::INFINITY.exponent(), 255);
        assert_eq!(f32::NEG_INFINITY.exponent(), 255);
    }

    #[test]
    fn test_denormal_exponent() {
        assert_eq!(1.0e-40f32.exponent(), 0);
        assert_eq!(1.0e-42.exponent(), 0);
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
