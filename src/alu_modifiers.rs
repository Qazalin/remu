pub trait VOPModifier<T> {
    fn negate(&self, pos: usize, modifier: usize) -> T;
    fn absolute(&self, pos: usize, modifier: usize) -> T;
}

impl VOPModifier<f32> for f32 {
    fn negate(&self, pos: usize, modifier: usize) -> f32 {
        match (modifier >> pos) & 1 {
            1 => -(*self),
            _ => *self,
        }
    }
    fn absolute(&self, pos: usize, modifier: usize) -> f32 {
        match (modifier >> pos) & 1 {
            1 => f32::abs(*self),
            _ => *self,
        }
    }
}

impl VOPModifier<f64> for f64 {
    fn negate(&self, pos: usize, modifier: usize) -> f64 {
        match (modifier >> pos) & 1 {
            1 => -(*self),
            _ => *self,
        }
    }
    fn absolute(&self, pos: usize, modifier: usize) -> f64 {
        match (modifier >> pos) & 1 {
            1 => f64::abs(*self),
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
