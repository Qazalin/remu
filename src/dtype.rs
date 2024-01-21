pub trait DType: Copy + 'static + core::fmt::Debug {}
macro_rules! impl_dtype_for {
    ($($t:ty),*) => {
        $(impl DType for $t {})*
    };
}
impl_dtype_for!(u8, u16, u32, u64);

pub trait IEEEClass {
    fn exponent(&self) -> u32;
}

impl IEEEClass for f32 {
    fn exponent(&self) -> u32 {
        (self.to_bits() & 0b01111111100000000000000000000000) >> 23
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
}
