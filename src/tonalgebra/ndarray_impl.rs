//! Implementations for ndarray types being converted to nalgebra types.

use super::*;

use nalgebra::Dynamic as Dy;
use std::convert::TryFrom;

/// ```
/// use nshare::ToNalgebra;
///
/// let arr = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
/// let m = arr.view().into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
impl<'a, T> ToNalgebra for ndarray::ArrayView1<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVectorSlice<'a, T>;
    fn into_nalgebra(self) -> Self::Out {
        let len = Dy::new(self.len());
        let ptr = self.as_ptr();
        let stride: usize = TryFrom::try_from(self.strides()[0]).expect("Negative stride");
        let storage = unsafe {
            nalgebra::SliceStorage::from_raw_parts(
                ptr,
                (len, nalgebra::Const::<1>),
                (nalgebra::Const::<1>, Dy::new(stride)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}
/// ```
/// use nshare::ToNalgebra;
///
/// let mut arr = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
/// let m = arr.view_mut().into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
impl<'a, T> ToNalgebra for ndarray::ArrayViewMut1<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVectorSliceMut<'a, T>;
    fn into_nalgebra(mut self) -> Self::Out {
        let len = Dy::new(self.len());
        let stride: usize = TryFrom::try_from(self.strides()[0]).expect("Negative stride");
        let ptr = self.as_mut_ptr();
        let storage = unsafe {
            // Drop to not have simultaneously the ndarray and nalgebra valid.
            drop(self);
            nalgebra::SliceStorageMut::from_raw_parts(
                ptr,
                (len, nalgebra::Const::<1>),
                (nalgebra::Const::<1>, Dy::new(stride)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::ToNalgebra;
///
/// let arr = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
/// let m = arr.into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
impl<T> ToNalgebra for ndarray::Array1<T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVector<T>;
    fn into_nalgebra(self) -> Self::Out {
        let len = Dy::new(self.len());
        Self::Out::from_vec_generic(len, nalgebra::Const::<1>, self.into_raw_vec())
    }
}

/// ```
/// use nshare::ToNalgebra;
///
/// let arr = ndarray::arr2(&[
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ]);
/// let m = arr.view().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr.t().into_nalgebra().column(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
impl<'a, T> ToNalgebra for ndarray::ArrayView2<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrixSlice<'a, T, Dy, Dy>;
    fn into_nalgebra(self) -> Self::Out {
        let nrows = Dy::new(self.nrows());
        let ncols = Dy::new(self.ncols());
        let ptr = self.as_ptr();
        let stride_row: usize = TryFrom::try_from(self.strides()[0]).expect("Negative row stride");
        let stride_col: usize =
            TryFrom::try_from(self.strides()[1]).expect("Negative column stride");
        let storage = unsafe {
            nalgebra::SliceStorage::from_raw_parts(
                ptr,
                (nrows, ncols),
                (Dy::new(stride_row), Dy::new(stride_col)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::ToNalgebra;
///
/// let mut arr = ndarray::arr2(&[
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ]);
/// let m = arr.view_mut().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr.view_mut().reversed_axes().into_nalgebra().column(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
impl<'a, T> ToNalgebra for ndarray::ArrayViewMut2<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrixSliceMut<'a, T, Dy, Dy>;
    fn into_nalgebra(mut self) -> Self::Out {
        let nrows = Dy::new(self.nrows());
        let ncols = Dy::new(self.ncols());
        let stride_row: usize = TryFrom::try_from(self.strides()[0]).expect("Negative row stride");
        let stride_col: usize =
            TryFrom::try_from(self.strides()[1]).expect("Negative column stride");
        let ptr = self.as_mut_ptr();
        let storage = unsafe {
            // Drop to not have simultaneously the ndarray and nalgebra valid.
            drop(self);
            nalgebra::SliceStorageMut::from_raw_parts(
                ptr,
                (nrows, ncols),
                (Dy::new(stride_row), Dy::new(stride_col)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::ToNalgebra;
///
/// let mut arr = ndarray::arr2(&[
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ]);
/// let m = arr.clone().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr.reversed_axes().into_nalgebra().column(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
impl<T> ToNalgebra for ndarray::Array2<T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrix<T>;
    fn into_nalgebra(self) -> Self::Out {
        let std_layout = self.is_standard_layout();
        let nrows = Dy::new(self.nrows());
        let ncols = Dy::new(self.ncols());
        let mut res = Self::Out::from_vec_generic(nrows, ncols, self.into_raw_vec());
        if std_layout {
            // This can be expensive, but we have no choice since nalgebra VecStorage is always
            // column-based.
            res.transpose_mut();
        }
        res
    }
}
