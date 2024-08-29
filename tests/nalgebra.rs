use nshare::IntoNalgebra;

#[test]
fn single_row_ndarray_to_nalgebra() {
    let mut arr = ndarray::array![[0.1, 0.2, 0.3, 0.4]];
    let m = arr.view_mut().into_nalgebra();
    assert!(m.row(0).iter().eq(&[0.1, 0.2, 0.3, 0.4]));
    assert_eq!(m.shape(), (1, 4));
    assert!(arr
        .view_mut()
        .reversed_axes()
        .into_nalgebra()
        .column(0)
        .iter()
        .eq(&[0.1, 0.2, 0.3, 0.4]));
}
