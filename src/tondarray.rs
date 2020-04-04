#[cfg(feature = "image")]
mod image_impl;
#[cfg(feature = "nalgebra")]
mod nalgebra_impl;

/// Converts a 1d type to a ndarray 1d array type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToNdarray1 {
    type Out;

    fn to_ndarray1(self) -> Self::Out;
}

/// Converts a 2d type to a ndarray 2d array type.
///
/// Coordinates are in (row, col).
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToNdarray2 {
    type Out;

    fn to_ndarray2(self) -> Self::Out;
}

/// Converts a 3d type to a ndarray 2d array type.
///
/// Coordinates are in `(channel, row, col)`, where channel is typically a color channel,
/// or they are in `(z, y, x)`.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToNdarray3 {
    type Out;

    fn to_ndarray3(self) -> Self::Out;
}

/// Converts a refferenced 1d type to a ndarray 1d array type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait RefNdarray1 {
    type Out;

    fn ref_ndarray1(self) -> Self::Out;
}

/// Converts a refferenced 2d type to a ndarray 2d array type.
///
/// Coordinates are in (row, col).
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait RefNdarray2 {
    type Out;

    fn ref_ndarray2(self) -> Self::Out;
}

/// Converts a refferenced 3d type to a ndarray 2d array type.
///
/// Coordinates are in `(channel, row, col)`, where channel is typically a color channel,
/// or they are in `(z, y, x)`.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait RefNdarray3 {
    type Out;

    fn ref_ndarray3(self) -> Self::Out;
}
