#[cfg(feature = "image")]
mod image_impl;
#[cfg(feature = "nalgebra")]
mod nalgebra_impl;

/// Converts a 1d type to a ndarray 1d array type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait IntoNdarray1 {
    type Out;

    fn into_ndarray1(self) -> Self::Out;
}

/// Converts a 2d type to a ndarray 2d array type.
///
/// Coordinates are in (row, col).
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait IntoNdarray2 {
    type Out;

    fn into_ndarray2(self) -> Self::Out;
}

/// Converts a 3d type to a ndarray 2d array type.
///
/// Coordinates are in `(channel, row, col)`, where channel is typically a color channel,
/// or they are in `(z, y, x)`.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait IntoNdarray3 {
    type Out;

    fn into_ndarray3(self) -> Self::Out;
}

/// Borrows a 1d type to a ndarray 1d array type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait AsNdarray1 {
    type Out<'a>
    where
        Self: 'a;

    fn as_ndarray1(&self) -> Self::Out<'_>;
}

/// Borrows a 2d type to a ndarray 2d array type.
///
/// Coordinates are in (row, col).
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait AsNdarray2 {
    type Out<'a>
    where
        Self: 'a;

    fn as_ndarray2(&self) -> Self::Out<'_>;
}

/// Borrows a 3d type to a ndarray 2d array type.
///
/// Coordinates are in `(channel, row, col)`, where channel is typically a color channel,
/// or they are in `(z, y, x)`.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait AsNdarray3 {
    type Out<'a>
    where
        Self: 'a;

    fn as_ndarray3(&self) -> Self::Out<'_>;
}

/// Mutably borrows a 1d type to a ndarray 1d array type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait AsNdarray1Mut {
    type Out<'a>
    where
        Self: 'a;

    fn as_ndarray1_mut(&mut self) -> Self::Out<'_>;
}

/// Mutably borrows a 2d type to a ndarray 2d array type.
///
/// Coordinates are in (row, col).
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait AsNdarray2Mut {
    type Out<'a>
    where
        Self: 'a;

    fn as_ndarray2_mut(&mut self) -> Self::Out<'_>;
}

/// Mutably borrows a 3d type to a ndarray 2d array type.
///
/// Coordinates are in `(channel, row, col)`, where channel is typically a color channel,
/// or they are in `(z, y, x)`.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait AsNdarray3Mut {
    type Out<'a>
    where
        Self: 'a;

    fn as_ndarray3_mut(&mut self) -> Self::Out<'_>;
}
