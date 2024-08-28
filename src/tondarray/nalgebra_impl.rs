//! Implementations for nalgebra types being converted to ndarray types.

use super::*;
use nalgebra::{dimension::U1, Dim, Dyn, Scalar};
use ndarray::ShapeBuilder;

/// ```
/// use nshare::AsNdarray1;
/// use nalgebra::Vector4;
/// use ndarray::s;
///
/// let m = Vector4::new(
///     0.1, 0.2, 0.3, 0.4f32,
/// );
/// let arr = m.as_ndarray1();
/// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(arr.dim(), 4);
/// ```
impl<N: Scalar, R: Dim, S> AsNdarray1 for nalgebra::Vector<N, R, S>
where
    S: nalgebra::Storage<N, R, U1>,
{
    type Out<'a> = ndarray::ArrayView1<'a, N>
    where
        S: 'a;

    fn as_ndarray1(&self) -> Self::Out<'_> {
        unsafe {
            ndarray::ArrayView1::from_shape_ptr(
                (self.shape().0,).strides((self.strides().0,)),
                self.as_ptr(),
            )
        }
    }
}

/// ```
/// use nshare::AsNdarray1Mut;
/// use nalgebra::Vector4;
/// use ndarray::s;
///
/// let mut m = Vector4::new(
///     0.1, 0.2, 0.3, 0.4f32,
/// );
/// // Set everything to 0.
/// m.as_ndarray1_mut().fill(0.0);
/// assert!(m.iter().eq(&[0.0; 4]));
/// ```
impl<N: Scalar, R: Dim, S> AsNdarray1Mut for nalgebra::Vector<N, R, S>
where
    S: nalgebra::StorageMut<N, R, U1>,
{
    type Out<'a> = ndarray::ArrayViewMut1<'a, N>
    where
        S: 'a;

    fn as_ndarray1_mut(&mut self) -> Self::Out<'_> {
        unsafe {
            ndarray::ArrayViewMut1::from_shape_ptr(
                (self.shape().0,).strides((self.strides().0,)),
                self.as_ptr() as *mut N,
            )
        }
    }
}

/// ```
/// use nshare::IntoNdarray1;
/// use nalgebra::Vector4;
///
/// let m = Vector4::new(
///     0.1, 0.2, 0.3, 0.4f32,
/// );
/// let arr = m.rows(0, 4).into_ndarray1();
/// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(arr.dim(), 4);
/// ```
impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> IntoNdarray1
    for nalgebra::Vector<N, R, nalgebra::ViewStorage<'a, N, R, U1, RStride, CStride>>
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
/// use nshare::IntoNdarray1;
/// use nalgebra::{Vector4, dimension::U2, Const};
///
/// let mut m = Vector4::new(
///     0.1, 0.2, 0.3, 0.4,
/// );
/// let arr = m.rows_generic_with_step_mut::<Const<2>>(0, Const::<2>, 1).into_ndarray1().fill(0.0);
/// assert!(m.iter().eq(&[0.0, 0.2, 0.0, 0.4]));
/// ```
impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> IntoNdarray1
    for nalgebra::Matrix<N, R, U1, nalgebra::ViewStorageMut<'a, N, R, U1, RStride, CStride>>
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
/// use nshare::AsNdarray2;
/// use nalgebra::Matrix4;
/// use ndarray::s;
///
/// let m = Matrix4::new(
///     0.1, 0.2, 0.3, 0.4,
///     0.5, 0.6, 0.7, 0.8,
///     1.1, 1.2, 1.3, 1.4,
///     1.5, 1.6, 1.7, 1.8,
/// );
/// let arr = m.as_ndarray2();
/// assert!(arr.slice(s![1, ..]).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(arr.dim(), (4, 4));
/// ```
impl<N: Scalar, R: Dim, C: Dim, S> AsNdarray2 for nalgebra::Matrix<N, R, C, S>
where
    S: nalgebra::Storage<N, R, C>,
{
    type Out<'a> = ndarray::ArrayView2<'a, N>
    where
        S: 'a;

    fn as_ndarray2(&self) -> Self::Out<'_> {
        unsafe {
            ndarray::ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
        }
    }
}

/// ```
/// use nshare::AsNdarray2Mut;
/// use nalgebra::Matrix4;
/// use ndarray::s;
///
/// let mut m = Matrix4::new(
///     0.1, 0.2, 0.3, 0.4,
///     0.5, 0.6, 0.7, 0.8,
///     1.1, 1.2, 1.3, 1.4,
///     1.5, 1.6, 1.7, 1.8,
/// );
/// let arr = m.as_ndarray2_mut().slice_mut(s![1, ..]).fill(0.0);
/// assert!(m.row(1).iter().eq(&[0.0; 4]));
/// ```
impl<N: Scalar, R: Dim, C: Dim, S> AsNdarray2Mut for nalgebra::Matrix<N, R, C, S>
where
    S: nalgebra::StorageMut<N, R, C>,
{
    type Out<'a> = ndarray::ArrayViewMut2<'a, N>
    where
        S: 'a;

    fn as_ndarray2_mut(&mut self) -> Self::Out<'_> {
        unsafe {
            ndarray::ArrayViewMut2::from_shape_ptr(
                self.shape().strides(self.strides()),
                self.as_ptr() as *mut N,
            )
        }
    }
}

/// ```
/// use nshare::IntoNdarray2;
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
impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarray2
    for nalgebra::Matrix<N, R, C, nalgebra::ViewStorage<'a, N, R, C, RStride, CStride>>
{
    type Out = ndarray::ArrayView2<'a, N>;

    fn into_ndarray2(self) -> Self::Out {
        unsafe {
            ndarray::ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr())
        }
    }
}

/// ```
/// use nshare::IntoNdarray2;
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
impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> IntoNdarray2
    for nalgebra::Matrix<N, R, C, nalgebra::ViewStorageMut<'a, N, R, C, RStride, CStride>>
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

/// ```
/// use nshare::IntoNdarray1;
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
impl<N: Scalar> IntoNdarray1 for nalgebra::DVector<N> {
    type Out = ndarray::Array1<N>;

    fn into_ndarray1(self) -> Self::Out {
        ndarray::Array1::from_shape_vec((self.shape().0,), self.data.into()).unwrap()
    }
}

/// ```
/// use nshare::IntoNdarray2;
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
impl<N: Scalar> IntoNdarray2 for nalgebra::Matrix<N, Dyn, Dyn, nalgebra::VecStorage<N, Dyn, Dyn>>
where
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<Dyn, Dyn, Buffer<N> = nalgebra::VecStorage<N, Dyn, Dyn>>,
{
    type Out = ndarray::Array2<N>;

    fn into_ndarray2(self) -> Self::Out {
        ndarray::Array2::from_shape_vec(self.shape().strides(self.strides()), self.data.into())
            .unwrap()
    }
}
