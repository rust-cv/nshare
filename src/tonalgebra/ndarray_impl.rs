//! Implementations for ndarray types being converted to nalgebra types.

use super::*;

use core::convert::TryFrom;
use nalgebra::Dyn;

/// ```
/// use nshare::IntoNalgebra;
///
/// let arr = ndarray::array![0.1, 0.2, 0.3, 0.4];
/// let m = arr.view().into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
impl<'a, T> IntoNalgebra for ndarray::ArrayView1<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVectorView<'a, T>;
    fn into_nalgebra(self) -> Self::Out {
        let len = Dyn(self.len());
        let ptr = self.as_ptr();
        let stride: usize = TryFrom::try_from(self.strides()[0]).expect("Negative stride");
        let storage = unsafe {
            nalgebra::ViewStorage::from_raw_parts(
                ptr,
                (len, nalgebra::Const::<1>),
                (nalgebra::Const::<1>, Dyn(stride)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}
/// ```
/// use nshare::IntoNalgebra;
///
/// let mut arr = ndarray::array![0.1, 0.2, 0.3, 0.4];
/// let m = arr.view_mut().into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
impl<'a, T> IntoNalgebra for ndarray::ArrayViewMut1<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVectorViewMut<'a, T>;
    fn into_nalgebra(mut self) -> Self::Out {
        let len = Dyn(self.len());
        let stride: usize = TryFrom::try_from(self.strides()[0]).expect("Negative stride");
        let ptr = self.as_mut_ptr();
        let storage = unsafe {
            nalgebra::ViewStorageMut::from_raw_parts(
                ptr,
                (len, nalgebra::Const::<1>),
                (nalgebra::Const::<1>, Dyn(stride)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::IntoNalgebra;
///
/// let arr = ndarray::array![0.1, 0.2, 0.3, 0.4];
/// let m = arr.into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
impl<T> IntoNalgebra for ndarray::Array1<T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVector<T>;
    fn into_nalgebra(self) -> Self::Out {
        let len = Dyn(self.len());
        // There is no method to give nalgebra the vector directly where it isn't allocated. If you call
        // from_vec_generic, it simply calls from_iterator_generic which uses Iterator::collect(). Due to this,
        // the simplest solution is to just pass an iterator over the values. If you come across this because you
        // have a performance issue, I would recommend creating the owned data using naglebra and borrowing it with
        // ndarray to perform operations on it instead of the other way around.
        Self::Out::from_iterator_generic(len, nalgebra::Const::<1>, self.iter().cloned())
    }
}

/// ```
/// use nshare::IntoNalgebra;
///
/// let arr = ndarray::array![
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ];
/// let m = arr.view().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr.t().into_nalgebra().column(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
impl<'a, T> IntoNalgebra for ndarray::ArrayView2<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrixView<'a, T, Dyn, Dyn>;
    fn into_nalgebra(self) -> Self::Out {
        let nrows = Dyn(self.nrows());
        let ncols = Dyn(self.ncols());
        let ptr = self.as_ptr();
        let stride_row: usize = TryFrom::try_from(self.strides()[0])
            .expect("can only convert positive row stride to nalgebra");
        let stride_col: usize = TryFrom::try_from(self.strides()[1])
            .expect("can only convert positive col stride to nalgebra");
        let storage = unsafe {
            nalgebra::ViewStorage::from_raw_parts(
                ptr,
                (nrows, ncols),
                (Dyn(stride_row), Dyn(stride_col)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::IntoNalgebra;
///
/// let mut arr = ndarray::array![
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ];
/// let m = arr.view_mut().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr.view_mut().reversed_axes().into_nalgebra().column(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
impl<'a, T> IntoNalgebra for ndarray::ArrayViewMut2<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrixViewMut<'a, T, Dyn, Dyn>;
    fn into_nalgebra(mut self) -> Self::Out {
        let nrows = Dyn(self.nrows());
        let ncols = Dyn(self.ncols());
        let stride_row: usize = TryFrom::try_from(self.strides()[0])
            .expect("can only convert positive row stride to nalgebra");
        let stride_col: usize = TryFrom::try_from(self.strides()[1])
            .expect("can only convert positive col stride to nalgebra");
        let ptr = self.as_mut_ptr();
        let storage = unsafe {
            nalgebra::ViewStorageMut::from_raw_parts(
                ptr,
                (nrows, ncols),
                (Dyn(stride_row), Dyn(stride_col)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::IntoNalgebra;
///
/// let mut arr = ndarray::array![
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ];
/// let m = arr.clone().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr.reversed_axes().into_nalgebra().column(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
impl<T> IntoNalgebra for ndarray::Array2<T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrix<T>;
    fn into_nalgebra(self) -> Self::Out {
        let nrows = Dyn(self.nrows());
        let ncols = Dyn(self.ncols());
        Self::Out::from_iterator_generic(nrows, ncols, self.t().iter().cloned())
    }
}
