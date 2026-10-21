/* The NumPy pool's (src/numpy_pool.cpp) allocator for the run engine's image storage (src/run): the same pooled
 * blocks NumPy's handler serves, without the GIL, from any thread. A request at or above the pool's threshold
 * (512 KiB) is served like NumPy's (an idle block in [size, 2 size], else a fresh one) once the handler has been
 * installed; any other request, or every request while the pool is not installed, is a 64-byte-aligned heap block.
 * numpy_pool_release() releases idle engine blocks too. */
#pragma once

#include <cstddef>

/* size bytes of uninitialised memory, 64-byte aligned; null when memory runs out. */
void *cn_numpy_pool_allocate(size_t size) noexcept;
/* Frees a block cn_numpy_pool_allocate returned (null is ignored). */
void cn_numpy_pool_free(void *data) noexcept;
