# nshare

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo] ![ci][bci]

[ci]: https://img.shields.io/crates/v/nshare.svg
[cl]: https://crates.io/crates/nshare/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/nshare/badge.svg
[dl]: https://docs.rs/nshare/

[lo]: https://tokei.rs/b1/github/rust-cv/nshare?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[bci]: https://github.com/rust-cv/nshare/workflows/ci/badge.svg

Provides traits that allow conversion between n-dimensional types in different Rust crates

**NOTE**: By default, this crate includes no conversions. You must choose which crates you want to use using the features:

* `ndarray`
* `nalgebra`
* `image`

When crates are included, any available conversions between the enabled crates are turned on.

Right now this crate really only provides conversions to owned and borrowed ndarray types. Some limitations exist with `nalgebra`, as it only utilizes positive strides, while `ndarray` supports negative strides as well. The `image` crate has no concept of strides. Due to this, the `ndarray` crate is the most flexible, and is ideal for interoperability between these various crates.

## Supported Crates
* `image`
* `ndarray`
* `nalgebra`
