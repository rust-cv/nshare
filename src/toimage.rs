#[cfg(feature = "image")]
mod ndarray_impl;

/// Converts a 2d type to a luma image type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToImageLuma {
    type Out;

    fn into_image_luma(self) -> Self::Out;
}

/// Converts a 3d array type to an RgbaImage type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToImageRgba {
    type Out;

    fn into_image_rgba(self) -> Self::Out;
}

/// Converts a 3d array type to an RgbaImage type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToImageRgb {
    type Out;

    fn into_image_rgb(self) -> Self::Out;
}

/// Converts a 2d array type to a GrayImage type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToImageGray {
    type Out;

    fn into_image_gray(self) -> Self::Out;
}
