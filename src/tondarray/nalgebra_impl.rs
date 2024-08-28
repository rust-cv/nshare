//! Implementations for nalgebra types being converted to ndarray types.

use super::*;
use nalgebra::{
    dimension::U1,
    storage::{Storage, StorageMut},
    Dim, Matrix, Scalar, Vector, ViewStorage, ViewStorageMut,
};
use ndarray::ShapeBuilder;

/// ```
/// use nshare::RefNdarray1;
/// use nalgebra::Vector4;
/// use ndarray::s;
///
/// let m = Vector4::new(
///     0.1, 0.2, 0.3, 0.4f32,
/// );
/// let arr = m.ref_ndarray1();
/// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(arr.dim(), 4);
/// ```
impl<'a, N: Scalar, R: Dim, S> RefNdarray1 for &'a Vector<N, R, S>
where
    S: Storage<N, R, U1>,
{
    type Out = ndarray::ArrayView1<'a, N>;

    fn ref_ndarray1(self) -> Self::Out {
        unsafe {
            ndarray::ArrayView1::from_shape_ptr(
                (self.shape().0,).strides((self.strides().0,)),
                self.as_ptr(),
            )
        }
    }
}

/// ```
/// use nshare::MutNdarray1;
/// use nalgebra::Vector4;
/// use ndarray::s;
///
/// let mut m = Vector4::new(
///     0.1, 0.2, 0.3, 0.4f32,
/// );
/// // Set everything to 0.
/// m.mut_ndarray1().fill(0.0);
/// assert!(m.iter().eq(&[0.0; 4]));
/// ```
impl<'a, N: Scalar, R: Dim, S> MutNdarray1 for &'a mut Vector<N, R, S>
where
    S: StorageMut<N, R, U1>,
{
    type Out = ndarray::ArrayViewMut1<'a, N>;

    fn mut_ndarray1(self) -> Self::Out {
        unsafe {
            ndarray::ArrayViewMut1::from_shape_ptr(
                (self.shape().0,).strides((self.strides().0,)),
                self.as_ptr() as *mut N,
            )
        }
    }
}

/// ```
/// use nshare::ToNdarray1;
/// use nalgebra::Vector4;
///
/// let m = Vector4::new(
///     0.1, 0.2, 0.3, 0.4f32,
/// );
/// let arr = m.rows(0, 4).into_ndarray1();
/// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(arr.dim(), 4);
/// ```
impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> ToNdarray1
    for Vector<N, R, ViewStorage<'a, N, R, U1, RStride, CStride>>
{
    type Out = ndarray::ArrayView1<'a, N>;

    fn into_ndarray1(self) -> Self::Out {
        unsafe {
            ndarray::ArrayView1::from_shape_ptr(
                (self.shape().0,).strides((self.strides().0,)),
                self.as_ptr(),
            )
        }
    }
}

/// ```
/// use nshare::ToNdarray1;
/// use nalgebra::{Vector4, dimension::U2, Const};
///
/// let mut m = Vector4::new(
///     0.1, 0.2, 0.3, 0.4,
/// );
/// let arr = m.rows_generic_with_step_mut::<Const<2>>(0, Const::<2>, 1).into_ndarray1().fill(0.0);
/// assert!(m.iter().eq(&[0.0, 0.2, 0.0, 0.4]));
/// ```
impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> ToNdarray1
    for Matrix<N, R, U1, ViewStorageMut<'a, N, R, U1, RStride, CStride>>
{
    type Out = ndarray::ArrayViewMut1<'a, N>;

    fn into_ndarray1(self) -> Self::Out {
        unsafe {
            ndarray::ArrayViewMut1::from_shape_ptr(
                (self.shape().0,).strides((self.strides().0,)),
                self.as_ptr() as *mut N,
            )
        }
    }
}

/// ```
/// use nshare::RefNdarray2;
/// use nalgebra::Matrix4;
/// use ndarray::s;
///
/// let m = Matrix4::new(
///     0.1, 0.2, 0.3, 0.4,
///     0.5, 0.6, 0.7, 0.8,
///     1.1, 1.2, 1.3, 1.4,
///     1.5, 1.6, 1.7, 1.8,
/// );
/// let arr = m.ref_ndarray2();
/// assert!(arr.slice(s![1, ..]).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(arr.dim(), (4, 4));
/// ```
impl<'a, N: Scalar, R: Dim, C: Dim, S> RefNdarray2 for &'a Matrix<N, R, C, S>
where
    S: Storage<N, R, C>,
{
    type Out = ndarray::ArrayView2<'a, N>;

    fn ref_ndarray2(self) -> Self::Out {
        unsafe {
            ndarray::ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
        }
    }
}

/// ```
/// use nshare::MutNdarray2;
/// use nalgebra::Matrix4;
/// use ndarray::s;
///
/// let mut m = Matrix4::new(
///     0.1, 0.2, 0.3, 0.4,
///     0.5, 0.6, 0.7, 0.8,
///     1.1, 1.2, 1.3, 1.4,
///     1.5, 1.6, 1.7, 1.8,
/// );
/// let arr = m.mut_ndarray2().slice_mut(s![1, ..]).fill(0.0);
/// assert!(m.row(1).iter().eq(&[0.0; 4]));
/// ```
impl<'a, N: Scalar, R: Dim, C: Dim, S> MutNdarray2 for &'a mut Matrix<N, R, C, S>
where
    S: StorageMut<N, R, C>,
{
    type Out = ndarray::ArrayViewMut2<'a, N>;

    fn mut_ndarray2(self) -> Self::Out {
        unsafe {
            ndarray::ArrayViewMut2::from_shape_ptr(
                self.shape().strides(self.strides()),
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
/// let arr = m.row(1).into_ndarray2();
/// assert!(arr.iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(arr.dim(), (1, 4));
/// ```
impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> ToNdarray2
    for Matrix<N, R, C, ViewStorage<'a, N, R, C, RStride, CStride>>
{
    type Out = ndarray::ArrayView2<'a, N>;

    fn into_ndarray2(self) -> Self::Out {
        unsafe {
            ndarray::ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
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
/// let arr = m.row_mut(1).into_ndarray2().fill(0.0);
/// assert!(m.row(1).iter().eq(&[0.0; 4]));
/// ```
impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> ToNdarray2
    for Matrix<N, R, C, ViewStorageMut<'a, N, R, C, RStride, CStride>>
{
    type Out = ndarray::ArrayViewMut2<'a, N>;

    fn into_ndarray2(self) -> Self::Out {
        unsafe {
            ndarray::ArrayViewMut2::from_shape_ptr(
                self.shape().strides(self.strides()),
                self.as_ptr() as *mut N,
            )
        }
    }
}

#[cfg(feature = "nalgebra_std")]
mod std_impl {
    use super::*;
    use nalgebra::{allocator::Allocator, DVector, DefaultAllocator, Dyn, VecStorage};
    use ndarray::{Array1, Array2};
    /// ```
    /// use nshare::ToNdarray1;
    /// use nalgebra::DVector;
    /// use ndarray::s;
    ///
    /// let m = DVector::from_vec(vec![
    ///     0.1, 0.2, 0.3, 0.4,
    /// ]);
    /// let arr = m.into_ndarray1();
    /// assert_eq!(arr.dim(), 4);
    /// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
    /// ```
    impl<N: Scalar> ToNdarray1 for DVector<N> {
        type Out = Array1<N>;

        fn into_ndarray1(self) -> Self::Out {
            Array1::from_shape_vec((self.shape().0,), self.data.into()).unwrap()
        }
    }

    /// ```
    /// use nshare::ToNdarray2;
    /// use nalgebra::{Matrix, dimension::{U4, Dynamic}};
    /// use ndarray::s;
    ///
    /// // Note: from_vec takes data column-by-column !
    /// let m = Matrix::<f32, Dynamic, Dynamic, _>::from_vec(3, 4, vec![
    ///     0.1, 0.2, 0.3,
    ///     0.5, 0.6, 0.7,
    ///     1.1, 1.2, 1.3,
    ///     1.5, 1.6, 1.7,
    /// ]);
    /// let arr = m.into_ndarray2();
    /// assert!(arr.slice(s![.., 0]).iter().eq(&[0.1, 0.2, 0.3]));
    /// assert!(arr.slice(s![0, ..]).iter().eq(&[0.1, 0.5, 1.1, 1.5]));
    /// ```
    impl<N: Scalar> ToNdarray2 for Matrix<N, Dyn, Dyn, VecStorage<N, Dyn, Dyn>>
    where
        DefaultAllocator: Allocator<Dyn, Dyn, Buffer<N> = VecStorage<N, Dyn, Dyn>>,
    {
        type Out = Array2<N>;

        fn into_ndarray2(self) -> Self::Out {
            Array2::from_shape_vec(self.shape().strides(self.strides()), self.data.into()).unwrap()
        }
    }
}
