# Matting reference

The reference is the live PyMatting 1.1.16 of the tested stack; no PyMatting
source is copied here, so no license copy is kept here either. PyMatting's MIT
notice for the C adaptation is in `backend/src/nodes/impl/chainner_native.LICENSE.txt`.

`native/src/matting_complete_ops.c` adapts PyMatting 1.1.16's `cf_laplacian`,
`estimate_alpha_cf`, `ichol`, `cg` and `estimate_foreground_ml`. 1.1.16's
foreground starts from the mean colours of the alpha > 0.9 and alpha < 0.1 pixels,
so it is deterministic. 1.1.10 read two uninitialized `np.empty` buffers instead
(`native/reports/matting-idempotency.json`), and its frozen copies with those
buffers zeroed were retired (Consult 11 D-17.3). `test_matting_complete.py` and
`test_transparency.py` compare alpha, foreground and background with the live
`pymatting.estimate_alpha_cf` and `pymatting.estimate_foreground_ml` bit for bit.

Solver defaults match the actual nodes: closed-form radius 1 and epsilon 1e-7,
incomplete-Cholesky discard threshold 1e-4 and its original shift schedule, CG
tolerance 1e-7 / 10000 iterations, foreground regularization 1e-5 with 10 small /
2 large iterations at size 32. No test relaxes numerical tolerances. The C solver
retains only NumPy's public ILP64 OpenBLAS `cblas_ddot64_` for identical dot/norm
reduction ordering.

The C solver's sparse factor grows as needed while preserving the original
250,000,000-entry Cholesky limit, instead of allocating the full original two
250,000,000-entry arrays for every solve. Ordered Gauss-Seidel updates remain
serial within each call; independent calls have independent scratch buffers.
