#ifndef CHAINNER_NATIVE_H
#define CHAINNER_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CN_EXPORT __declspec(dllexport)
#else
#define CN_EXPORT __attribute__((visibility("default")))
#endif

/* Forced and suppressed inlining: MSVC's keywords under cl, the GNU attributes under
 * gcc and clang (clang-cl included; clang's GNU driver rejects __forceinline under
 * -Wpedantic), plain inlining elsewhere. */
#if defined(_MSC_VER) && !defined(__clang__)
#define CN_FORCE_INLINE static __forceinline
#define CN_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define CN_FORCE_INLINE static inline __attribute__((always_inline))
#define CN_NOINLINE __attribute__((noinline))
#else
#define CN_FORCE_INLINE static inline
#define CN_NOINLINE
#endif

typedef enum cn_status {
    CN_OK = 0,
    CN_INVALID_ARGUMENT = 1,
    CN_SIZE_OVERFLOW = 2,
    CN_ALLOCATION_FAILED = 3
} cn_status;

/* ABI version changes whenever an exported signature or its meaning changes. */
#ifdef __cplusplus
extern "C" {
#endif
CN_EXPORT int cn_abi_version(void);
#ifdef __cplusplus
}
#endif

#endif
