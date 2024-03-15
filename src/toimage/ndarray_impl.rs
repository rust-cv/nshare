use super::*;
use image::{GrayImage, ImageBuffer, RgbImage, RgbaImage, };
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3};

impl ToImageRgba for Array3<u8> {
    type Out = RgbaImage;

    fn into_image_rgba(self) -> Self::Out {
        let (c, h, w) = self.dim();
        assert_eq!(c, 4);
        let img: RgbaImage =
            ImageBuffer::from_raw(w as u32, h as u32, self.into_raw_vec()).unwrap();
        img
    }
}

impl ToImageRgb for Array3<u8> {
    type Out = RgbImage;

    fn into_image_rgb(self) -> Self::Out {
        let (c, h, w) = self.dim();
        assert_eq!(c, 3);
        let img: RgbImage = ImageBuffer::from_raw(w as u32, h as u32, self.into_raw_vec()).unwrap();
        img
    }
}

impl ToImageGray for Array2<u8> {
    type Out = GrayImage;

    fn into_image_gray(self) -> Self::Out {
        let (h, w) = self.dim();
        let img: GrayImage =
            ImageBuffer::from_raw(w as u32, h as u32, self.into_raw_vec()).unwrap();
        img
    }
}

mod test {
    use image::Rgba;
    use crate::ToNdarray3;
    use super::*;
    use super::ImageBuffer;

    #[test]
    fn test_img_alpha() {
        let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_fn(3, 3, |x, y| {
            let value = (y * 3 + x) as u8;
            Rgba([value, value, value, 255 - value])
        });
        let arr = img.into_ndarray3();
        assert_eq!(arr.dim(), (4, 3, 3));
        let arr_clone = arr.clone();
        let img = arr.into_image_rgba();
        let (width, height) = img.dimensions();

        for y in 0..height {
            for x in 0..width {
                let arr_pixel = [
                    arr_clone[[0, y as usize, x as usize]],
                    arr_clone[[1, y as usize, x as usize]],
                    arr_clone[[2, y as usize, x as usize]],
                    arr_clone[[3, y as usize, x as usize]],
                ];
                let img_pixel = img.get_pixel(x, y).0;
                assert_eq!(arr_pixel, img_pixel);
            }
        }
    }
}
