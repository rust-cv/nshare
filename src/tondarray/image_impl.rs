//! Implementations for conversions from image types to ndarray types.

use super::*;
use core::ops::{Deref, DerefMut};
use image::{flat::SampleLayout, ImageBuffer, Luma, Pixel, Primitive};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, ShapeBuilder};

extern crate alloc;

use alloc::vec::Vec;

/// ```
/// use image::GrayImage;
/// use nshare::IntoNdarray2;
/// use ndarray::s;
///
/// let zeros = GrayImage::new(2, 4);
/// let mut nd = zeros.into_ndarray2();
/// nd.fill(255);
/// // ndarray uses (row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (4, 2));
/// ```
impl<A> IntoNdarray2 for ImageBuffer<Luma<A>, Vec<A>>
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
/// use nshare::AsNdarray2;
/// use ndarray::s;
///
/// let mut vals = GrayImage::new(2, 4);
/// vals[(1, 0)] = Luma([255]);
/// let nd = vals.as_ndarray2();
/// // ndarray uses (row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (4, 2));
/// // The first row should sum to 255.
/// assert_eq!(nd.slice(s![0, ..]).sum(), 255);
/// ```
impl<A, Container> AsNdarray2 for ImageBuffer<Luma<A>, Container>
where
    A: Primitive + 'static,
    Container: Deref<Target = [A]>,
{
    type Out<'a> = ArrayView2<'a, A>
    where
        Container: 'a;

    fn as_ndarray2(&self) -> Self::Out<'_> {
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
/// use nshare::AsNdarray2Mut;
/// use ndarray::s;
///
/// let mut vals = GrayImage::new(2, 4);
/// let mut nd = vals.as_ndarray2_mut();
/// assert_eq!(nd.dim(), (4, 2));
/// nd.slice_mut(s![0, ..]).fill(255);
/// assert_eq!(vals[(1, 0)], Luma([255]));
/// ```
impl<A, Container> AsNdarray2Mut for ImageBuffer<Luma<A>, Container>
where
    A: Primitive + 'static,
    Container: DerefMut<Target = [A]>,
{
    type Out<'a> = ArrayViewMut2<'a, A>
    where
        Container: 'a;

    fn as_ndarray2_mut(&mut self) -> Self::Out<'_> {
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
/// use nshare::IntoNdarray3;
/// use ndarray::s;
///
/// let zeros = RgbImage::new(2, 4);
/// let mut nd = zeros.into_ndarray3();
/// nd.fill(255);
/// // ndarray uses (channel, row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (3, 4, 2));
/// ```
impl<P> IntoNdarray3 for ImageBuffer<P, Vec<P::Subpixel>>
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
/// use nshare::AsNdarray3;
/// use ndarray::s;
///
/// let mut vals = RgbImage::new(2, 4);
/// vals[(1, 0)] = Rgb([0, 255, 0]);
/// let nd = vals.as_ndarray3();
/// // ndarray uses (channel, row, col), so the dims get flipped.
/// assert_eq!(nd.dim(), (3, 4, 2));
/// // The first row green should sum to 255.
/// assert_eq!(nd.slice(s![1, 0, ..]).sum(), 255);
/// // The first row red should sum to 0.
/// assert_eq!(nd.slice(s![0, 0, ..]).sum(), 0);
/// ```
impl<P> AsNdarray3 for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    type Out<'a> = ArrayView3<'a, P::Subpixel>;

    fn as_ndarray3(&self) -> Self::Out<'_> {
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
/// use nshare::AsNdarray3Mut;
/// use ndarray::s;
///
/// let mut vals = RgbImage::new(2, 4);
/// // Set all the blue channel to 255.
/// vals.as_ndarray3_mut().slice_mut(s![2, .., ..]).fill(255);
/// assert_eq!(vals[(0, 0)], Rgb([0, 0, 255]));
/// ```
impl<P> AsNdarray3Mut for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    type Out<'a> = ArrayViewMut3<'a, P::Subpixel>;

    fn as_ndarray3_mut(&mut self) -> Self::Out<'_> {
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
