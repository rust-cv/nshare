/// Converts a 2d type to a luma image type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToImageLuma {
    type Out;

    fn into_image_luma(self) -> Self::Out;
}
