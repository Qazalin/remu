pub trait Negate {
    fn negate(&self, pos: usize, modifier: u32) -> f32;
}

impl Negate for f32 {
    fn negate(&self, pos: usize, modifier: u32) -> f32 {
        match (modifier >> pos as u32) & 1 {
            1 => -(*self),
            _ => *self,
        }
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
    }
}
