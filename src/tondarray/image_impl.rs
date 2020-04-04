//! Implementations for conversions from image types to ndarray types.

use super::*;
use image::{ImageBuffer, Luma, Primitive};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, ShapeBuilder};
use std::ops::{Deref, DerefMut};

/// ```
/// use image::GrayImage;
/// use nshare::ToNdarray2;
/// use ndarray::s;
///
/// let zeros = GrayImage::new(2, 4);
/// let mut nd = zeros.to_ndarray2();
/// // Fill x = 1 to all 255.
/// nd.fill(255);
/// // ndarray uses (row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (4, 2));
/// ```
impl<A> ToNdarray2 for ImageBuffer<Luma<A>, Vec<A>>
where
    A: Primitive + 'static,
{
    type Out = Array2<A>;

    fn to_ndarray2(self) -> Self::Out {
        let (width, height) = self.dimensions();
        let (width, height) = (width as usize, height as usize);
        let container = self.into_raw();
        Array2::from_shape_vec((height, width), container).unwrap()
    }
}

/// ```
/// use image::{GrayImage, Luma};
/// use nshare::RefNdarray2;
/// use ndarray::s;
///
/// let mut vals = GrayImage::new(2, 4);
/// vals[(1, 0)] = Luma([255]);
/// let nd = vals.ref_ndarray2();
/// // ndarray uses (row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (4, 2));
/// // The first row should sum to 255.
/// assert_eq!(nd.slice(s![0, ..]).sum(), 255);
/// ```
impl<'a, A, Container> RefNdarray2 for &'a ImageBuffer<Luma<A>, Container>
where
    A: Primitive + 'static,
    Container: Deref<Target = [A]>,
{
    type Out = ArrayView2<'a, A>;

    fn ref_ndarray2(self) -> Self::Out {
        let (width, height) = self.dimensions();
        let (width, height) = (width as usize, height as usize);
        ArrayView2::from_shape((height, width).strides((width, 1)), &**self).unwrap()
    }
}

/// ```
/// use image::{GrayImage, Luma};
/// use nshare::MutNdarray2;
/// use ndarray::s;
///
/// let mut vals = GrayImage::new(2, 4);
/// let mut nd = vals.mut_ndarray2();
/// assert_eq!(nd.dim(), (4, 2));
/// nd.slice_mut(s![0, ..]).fill(255);
/// assert_eq!(vals[(1, 0)], Luma([255]));
/// ```
impl<'a, A, Container> MutNdarray2 for &'a mut ImageBuffer<Luma<A>, Container>
where
    A: Primitive + 'static,
    Container: DerefMut<Target = [A]>,
{
    type Out = ArrayViewMut2<'a, A>;

    fn mut_ndarray2(self) -> Self::Out {
        let (width, height) = self.dimensions();
        let (width, height) = (width as usize, height as usize);
        ArrayViewMut2::from_shape((height, width).strides((width, 1)), &mut **self).unwrap()
    }
}
