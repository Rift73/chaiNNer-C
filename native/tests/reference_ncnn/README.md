# Installed NCNN oracle

These three source files are unmodified snapshots from chaiNNer
`app-0.25.1-nightly2025-10-21/resources/src/nodes/impl/ncnn`. They inherit the
repository's GPL-3.0 license and are independent of the converted implementation.

Original byte SHA-256:

- `model.py`: `ca402b2df706b879bde5bb80d33d779a0cfd0e2e9942a8bf8e2e5be09b425139`
- `optimizer.py`: `3b79fe76f4f1b8d5317b838e1471b41a27a281bab21ead47b39427143e4bd4ca`
- `param_schema.json`: `54f6026734751c4ae32296d1017609aef6dddee147171fdc4e580df3762ddebb`

`generate_optimizer_cpp.py` translates the optimizer's fixed control flow using
`native/tools/graph_codegen_common.py`; it is a provenance tool, not a runtime
dependency. The emitted C++ calls native weight operations and CPython object
primitives. The original Python passes are never called by the port.

The generator first applies its `CORRECTIONS`, single-line repairs of upstream
defects that keep every line number (`native/ARCHITECTURE.md` section 7).
`optimizer.py` itself stays unmodified; the differential tests run the corrected
source as their oracle, and a test regenerates the header and compares it with
the committed one.

Differential tests deliberately preserve the other observed upstream quirks:
Scale's decoded `weight` versus fusion's `scale` key, invalid fixed-axis
BatchNorm transposes, and original partial mutations before exceptions. They are
not silently repaired by the conversion; a repair is a recorded correction.

Inference fixtures are synthetic tiny networks run through the same installed
NCNN public binding with Vulkan explicitly disabled. No downloaded models,
GPU inference, or performance measurements are involved.

## Additional numeric primitive provenance

Integer weight arithmetic is implemented independently with explicit unsigned
modulo-width operations. Complex multiply/divide and square root preserve NumPy
1.24.4 operation order rather than substituting C++ `std::complex` semantics.
The square-root helper retains the upstream FreeBSD notice in
`native/include/ncnn_complex_math.hpp`. Source files were fetched from the pinned
NumPy tag and retained under ignored `native/reports/reference-sources-ncnn`:

- [`loops.c.src`](https://github.com/numpy/numpy/blob/v1.24.4/numpy/core/src/umath/loops.c.src): SHA-256 `4fefb71adc3c079fcc19bd820baef6764375971c7b1fe1b973d6db7fc7b50724`.
- [`npy_math_complex.c.src`](https://github.com/numpy/numpy/blob/v1.24.4/numpy/core/src/npymath/npy_math_complex.c.src): SHA-256 `895f53a743573f94c94b2fc5412942791159a03bf7d61efa36e316a176baefa4`.
- [`npy_math_internal.h.src`](https://github.com/numpy/numpy/blob/v1.24.4/numpy/core/src/npymath/npy_math_internal.h.src): SHA-256 `dbe1c4a7da4aec96cf6ac1a72a7ced18df8f961b7d5115c9fd41cfa9eadd8599`.
