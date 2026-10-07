# fpng 1.0.6

`fpng.cpp` and `fpng.h` are unmodified copies from the dependency selected by
`C:/Users/PC/Desktop/Repository/Imagetools/resize/build.ps1`:
`C:/Users/PC/Desktop/Repository/fpng/src`.

SHA-256:

- fpng.cpp: `c7fc10f3e9b67023a3551ed1919e23e54c7c9786b85838ffa55076d9c37d54dc`
- fpng.h: `96c1f60b276ec54bb0394c0865c6d8b15457694232c81199f4e5c97ce829b4df`

The full Unlicense notice is preserved at the end of `fpng.cpp` and reproduced
in `LICENSE`. The original source's public-domain attributions are also intact.
Upstream project: https://github.com/richgel999/fpng

chaiNNer initializes fpng once before use, encodes to memory with default flags,
and retains its existing Unicode-capable file-write boundary. Unsupported image
types, oversized vendor buffers, and explicit OpenCV codec options use the
existing encoder by capability selection. An fpng encoding failure is an error.

The integration caps each dimension at 65,535: this version writes only the low
two bytes of width/height to IHDR even though its argument check accepts larger
dimensions. It also keeps the filtered stream below `INT32_MAX - 1 MiB` so the
vendor's signed row sizes and 32-bit buffer offsets cannot overflow. Images
beyond either limit retain the existing OpenCV encoder.
