//! Implementations for conversions from image types to ndarray types.

use super::*;
use core::ops::{Deref, DerefMut};
use image::{flat::SampleLayout, ImageBuffer, Luma, Pixel, Primitive};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, ShapeBuilder};

extern crate alloc;

use alloc::vec::Vec;

/// ```
/// use image::GrayImage;
/// use nshare::ToNdarray2;
/// use ndarray::s;
///
/// let zeros = GrayImage::new(2, 4);
/// let mut nd = zeros.into_ndarray2();
/// nd.fill(255);
/// // ndarray uses (row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (4, 2));
/// ```
impl<A> ToNdarray2 for ImageBuffer<Luma<A>, Vec<A>>
where
    A: Primitive + 'static,
{
    type Out = Array2<A>;

    fn into_ndarray2(self) -> Self::Out {
        let SampleLayout {
            height,
            height_stride,
            width,
            width_stride,
            ..
        } = self.sample_layout();
        let shape = (height as usize, width as usize);
        let strides = (height_stride, width_stride);
        Array2::from_shape_vec(shape.strides(strides), self.into_raw()).unwrap()
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
        let SampleLayout {
            height,
            height_stride,
            width,
            width_stride,
            ..
        } = self.sample_layout();
        let shape = (height as usize, width as usize);
        let strides = (height_stride, width_stride);
        ArrayView2::from_shape(shape.strides(strides), self).unwrap()
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
        let SampleLayout {
            height,
            height_stride,
            width,
            width_stride,
            ..
        } = self.sample_layout();
        let shape = (height as usize, width as usize);
        let strides = (height_stride, width_stride);
        ArrayViewMut2::from_shape(shape.strides(strides), self).unwrap()
    }
}

/// ```
/// use image::RgbImage;
/// use nshare::ToNdarray3;
/// use ndarray::s;
///
/// let zeros = RgbImage::new(2, 4);
/// let mut nd = zeros.into_ndarray3();
/// nd.fill(255);
/// // ndarray uses (channel, row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (3, 4, 2));
/// ```
impl<P> ToNdarray3 for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    type Out = Array3<P::Subpixel>;

    fn into_ndarray3(self) -> Self::Out {
        let SampleLayout {
            channels,
            channel_stride,
            height,
            height_stride,
            width,
            width_stride,
        } = self.sample_layout();
        let shape = (channels as usize, height as usize, width as usize);
        let strides = (channel_stride, height_stride, width_stride);
        Array3::from_shape_vec(shape.strides(strides), self.into_raw()).unwrap()
    }
}

/// ```
/// use image::{RgbImage, Rgb};
/// use nshare::RefNdarray3;
/// use ndarray::s;
///
/// let mut vals = RgbImage::new(2, 4);
/// vals[(1, 0)] = Rgb([0, 255, 0]);
/// let nd = vals.ref_ndarray3();
/// // ndarray uses (channel, row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (3, 4, 2));
/// // The first row green should sum to 255.
/// assert_eq!(nd.slice(s![1, 0, ..]).sum(), 255);
/// // The first row red should sum to 0.
/// assert_eq!(nd.slice(s![0, 0, ..]).sum(), 0);
/// ```
impl<'a, P> RefNdarray3 for &'a ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    type Out = ArrayView3<'a, P::Subpixel>;

    fn ref_ndarray3(self) -> Self::Out {
        let SampleLayout {
            channels,
            channel_stride,
            height,
            height_stride,
            width,
            width_stride,
        } = self.sample_layout();
        let shape = (channels as usize, height as usize, width as usize);
        let strides = (channel_stride, height_stride, width_stride);
        ArrayView3::from_shape(shape.strides(strides), self).unwrap()
    }
}

/// ```
/// use image::{RgbImage, Rgb};
/// use nshare::MutNdarray3;
/// use ndarray::s;
///
/// let mut vals = RgbImage::new(2, 4);
/// // Set all the blue channel to 255.
/// vals.mut_ndarray3().slice_mut(s![2, .., ..]).fill(255);
/// assert_eq!(vals[(0, 0)], Rgb([0, 0, 255]));
/// ```
impl<'a, P> MutNdarray3 for &'a mut ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    type Out = ArrayViewMut3<'a, P::Subpixel>;

    fn mut_ndarray3(self) -> Self::Out {
        let SampleLayout {
            channels,
            channel_stride,
            height,
            height_stride,
            width,
            width_stride,
        } = self.sample_layout();
        let shape = (channels as usize, height as usize, width as usize);
        let strides = (channel_stride, height_stride, width_stride);
        ArrayViewMut3::from_shape(shape.strides(strides), self).unwrap()
    }
}
