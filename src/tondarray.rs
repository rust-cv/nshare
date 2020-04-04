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

/// Implementations for nalgebra types being converted to ndarray types.
#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use super::*;
    use nalgebra::{Dim, Matrix, Scalar, SliceStorage};
    use ndarray::{ArrayView2, ShapeBuilder};

    /// ```
    /// use nshare::ToNdarray2;
    /// use nalgebra::Matrix4;
    ///
    /// let m = Matrix4::new(
    ///     0.1, 0.2, 0.3, 0.4,
    ///     0.5, 0.6, 0.7, 0.8,
    ///     1.1, 1.2, 1.3, 1.4,
    ///     1.5, 1.6, 1.7, 1.8,
    /// );
    /// let arr = m.row(1).to_ndarray2();
    /// assert!(arr.iter().eq(&[0.5, 0.6, 0.7, 0.8]));
    /// assert_eq!(arr.dim(), (1, 4));
    /// ```
    impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> ToNdarray2
        for Matrix<N, R, C, SliceStorage<'a, N, R, C, RStride, CStride>>
    {
        type Out = ArrayView2<'a, N>;

        fn to_ndarray2(self) -> Self::Out {
            unsafe {
                ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
            }
        }
    }
}
