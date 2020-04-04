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
    use nalgebra::{dimension::U1, Dim, Matrix, Scalar, SliceStorage, SliceStorageMut, Vector};
    use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ShapeBuilder};

    /// ```
    /// use nshare::ToNdarray1;
    /// use nalgebra::Vector4;
    ///
    /// let m = Vector4::new(
    ///     0.1, 0.2, 0.3, 0.4f32,
    /// );
    /// let arr = m.rows(0, 4).to_ndarray1();
    /// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
    /// assert_eq!(arr.dim(), 4);
    /// ```
    impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> ToNdarray1
        for Vector<N, R, SliceStorage<'a, N, R, U1, RStride, CStride>>
    {
        type Out = ArrayView1<'a, N>;

        fn to_ndarray1(self) -> Self::Out {
            unsafe {
                ArrayView1::from_shape_ptr(
                    (self.shape().0,).strides((self.strides().0,)),
                    self.as_ptr(),
                )
            }
        }
    }

    /// ```
    /// use nshare::ToNdarray1;
    /// use nalgebra::{Vector4, dimension::U2};
    ///
    /// let mut m = Vector4::new(
    ///     0.1, 0.2, 0.3, 0.4,
    /// );
    /// let arr = m.rows_generic_with_step_mut::<U2>(0, U2, 1).to_ndarray1();
    /// for n in arr {
    ///     *n = 0.0;
    /// }
    /// assert!(m.iter().eq(&[0.0, 0.2, 0.0, 0.4]));
    /// ```
    impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> ToNdarray1
        for Matrix<N, R, U1, SliceStorageMut<'a, N, R, U1, RStride, CStride>>
    {
        type Out = ArrayViewMut1<'a, N>;

        fn to_ndarray1(self) -> Self::Out {
            unsafe {
                ArrayViewMut1::from_shape_ptr(
                    (self.shape().0,).strides((self.strides().0,)),
                    self.as_ptr() as *mut N,
                )
            }
        }
    }

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

    /// ```
    /// use nshare::ToNdarray2;
    /// use nalgebra::Matrix4;
    ///
    /// let mut m = Matrix4::new(
    ///     0.1, 0.2, 0.3, 0.4,
    ///     0.5, 0.6, 0.7, 0.8,
    ///     1.1, 1.2, 1.3, 1.4,
    ///     1.5, 1.6, 1.7, 1.8,
    /// );
    /// let arr = m.row_mut(1).to_ndarray2();
    /// for n in arr {
    ///     *n = 0.0;
    /// }
    /// assert!(m.row(1).iter().eq(&[0.0; 4]));
    /// ```
    impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> ToNdarray2
        for Matrix<N, R, C, SliceStorageMut<'a, N, R, C, RStride, CStride>>
    {
        type Out = ArrayViewMut2<'a, N>;

        fn to_ndarray2(self) -> Self::Out {
            unsafe {
                ArrayViewMut2::from_shape_ptr(
                    self.shape().strides(self.strides()),
                    self.as_ptr() as *mut N,
                )
            }
        }
    }

    #[cfg(feature = "std")]
    mod std_impl {
        use super::*;
        use nalgebra::{allocator::Allocator, DefaultAllocator, Dynamic, VecStorage};
        use ndarray::Array2;

        /// ```
        /// use nshare::ToNdarray2;
        /// use nalgebra::{Matrix, dimension::{U4, Dynamic}};
        /// use ndarray::s;
        ///
        /// let m = Matrix::<f32, Dynamic, Dynamic, _>::from_vec(4, 4, vec![
        ///     0.1, 0.2, 0.3, 0.4,
        ///     0.5, 0.6, 0.7, 0.8,
        ///     1.1, 1.2, 1.3, 1.4,
        ///     1.5, 1.6, 1.7, 1.8,
        /// ]);
        /// let arr = m.to_ndarray2();
        /// assert!(arr.slice(s![0, ..]).iter().eq(&[0.1, 0.2, 0.3, 0.4]));
        /// assert!(arr.slice(s![.., 0]).iter().eq(&[0.1, 0.5, 1.1, 1.5]));
        /// ```
        impl<'a, N: Scalar> ToNdarray2 for Matrix<N, Dynamic, Dynamic, VecStorage<N, Dynamic, Dynamic>>
        where
            DefaultAllocator:
                Allocator<N, Dynamic, Dynamic, Buffer = VecStorage<N, Dynamic, Dynamic>>,
        {
            type Out = Array2<N>;

            fn to_ndarray2(self) -> Self::Out {
                Array2::from_shape_vec(self.shape(), self.data.into()).unwrap()
            }
        }
    }
}
