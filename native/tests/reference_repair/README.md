These two Python node modules are unmodified snapshots from the original chaiNNer
checkout. Differential tests invoke the pinned OpenCV 4.8 runtime independently
of the new kernels and require identical byte images.

The inpainting reference is well-defined for tested images with both dimensions
at least two. The original source accesses an invalid adjacent row/column on
one-dimensional images. `native/reports/inpaint-idempotency.json` preserves the
fixture and observed repeated-output differences. The C++ port clamps only those
invalid output-pixel reads, so tiny-image tests assert stable output and untouched
unmasked pixels rather than equating arbitrary original heap contents.

The port preserves the source notices from OpenCV 4.8.0 `canny.cpp` and
`inpaint.cpp`. The original chaiNNer node snapshots retain the repository license.
