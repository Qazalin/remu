use num_traits::float::FloatCore;

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
