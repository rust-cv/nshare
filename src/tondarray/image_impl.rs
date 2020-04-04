//! Implementations for conversions from image types to ndarray types.

use super::*;
use image::{ImageBuffer, Luma, Primitive};
use ndarray::Array2;

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