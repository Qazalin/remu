pub trait DType: Copy + 'static {}
macro_rules! impl_dtype_for {
    ($($t:ty),*) => {
        $(impl DType for $t {})*
    };
}
impl_dtype_for!(u8, u16, u32, u64);
