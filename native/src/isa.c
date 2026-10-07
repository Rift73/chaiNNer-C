#include "isa.h"

/* CPUID leaf 1, ECX */
#define CN_CPUID_FMA (1u << 12)
#define CN_CPUID_OSXSAVE (1u << 27)
#define CN_CPUID_AVX (1u << 28)
/* CPUID leaf 7 subleaf 0, EBX */
#define CN_CPUID_BMI1 (1u << 3)
#define CN_CPUID_AVX2 (1u << 5)
#define CN_CPUID_BMI2 (1u << 8)
#define CN_CPUID_AVX512F (1u << 16)
#define CN_CPUID_AVX512DQ (1u << 17)
#define CN_CPUID_AVX512CD (1u << 28)
#define CN_CPUID_AVX512BW (1u << 30)
#define CN_CPUID_AVX512VL (1u << 31)
/* XCR0: SSE and YMM state; plus opmask, ZMM_Hi256 and Hi16_ZMM state */
#define CN_XCR0_AVX 0x6u
#define CN_XCR0_AVX512 0xE6u

int cn_isa_classify(uint32_t max_leaf, uint32_t leaf1_ecx, uint32_t leaf7_ebx, uint64_t xcr0)
{
    const uint32_t avx2_ecx = CN_CPUID_FMA | CN_CPUID_OSXSAVE | CN_CPUID_AVX;
    const uint32_t avx2_ebx = CN_CPUID_BMI1 | CN_CPUID_AVX2 | CN_CPUID_BMI2;
    const uint32_t avx512_ebx = CN_CPUID_AVX512F | CN_CPUID_AVX512DQ | CN_CPUID_AVX512CD
        | CN_CPUID_AVX512BW | CN_CPUID_AVX512VL;
    if (max_leaf < 7 || (leaf1_ecx & avx2_ecx) != avx2_ecx || (leaf7_ebx & avx2_ebx) != avx2_ebx
        || (xcr0 & CN_XCR0_AVX) != CN_XCR0_AVX) return CN_ISA_SCALAR;
    if ((leaf7_ebx & avx512_ebx) != avx512_ebx || (xcr0 & CN_XCR0_AVX512) != CN_XCR0_AVX512) return CN_ISA_AVX2;
    return CN_ISA_AVX512;
}

#if defined(_MSC_VER) && defined(_M_X64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#include <wchar.h>

static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
/* Set by the probe, afterwards only by cn_isa_set. */
static volatile LONG effective = CN_ISA_SCALAR;
/* Set once by the probe; InitOnceExecuteOnce orders them before every reader. */
static int requested = -1;
static int maximum = CN_ISA_SCALAR;

/* CHAINNER_C_ISA as a level: -1 (auto) unless it is exactly a level's name. */
static int read_override(void)
{
    static const wchar_t *const names[] = {L"scalar", L"avx2", L"avx512"};
    wchar_t value[8];
    /* Unset or empty returns 0; a value that does not fit returns the size it
     * needs, which is at least the buffer's. */
    DWORD length = GetEnvironmentVariableW(L"CHAINNER_C_ISA", value, ARRAYSIZE(value));
    if (length == 0 || length >= ARRAYSIZE(value)) return -1;
    for (int level = CN_ISA_SCALAR; level <= CN_ISA_AVX512; ++level) {
        if (wcscmp(value, names[level]) == 0) return level;
    }
    return -1;
}

static BOOL CALLBACK probe(PINIT_ONCE init, PVOID parameter, PVOID *context)
{
    (void)init; (void)parameter; (void)context;
    int words[4];
    uint32_t leaf1_ecx = 0, leaf7_ebx = 0;
    uint64_t xcr0 = 0;
    __cpuid(words, 0);
    uint32_t max_leaf = (uint32_t)words[0];
    if (max_leaf >= 1) {
        __cpuid(words, 1);
        leaf1_ecx = (uint32_t)words[2];
    }
    if (max_leaf >= 7) {
        __cpuidex(words, 7, 0);
        leaf7_ebx = (uint32_t)words[1];
    }
    /* XGETBV raises #UD unless the OS enabled XSAVE. */
    if (leaf1_ecx & CN_CPUID_OSXSAVE) xcr0 = _xgetbv(0);
    maximum = cn_isa_classify(max_leaf, leaf1_ecx, leaf7_ebx, xcr0);
    requested = read_override();
    effective = requested >= 0 && requested < maximum ? requested : maximum;
    return TRUE;
}

cn_isa_level cn_isa_current(void)
{
    /* The probe always succeeds; scalar is the level every CPU runs. */
    if (!InitOnceExecuteOnce(&once, probe, NULL, NULL)) return CN_ISA_SCALAR;
    return (cn_isa_level)ReadAcquire(&effective);
}

int cn_isa_set(int level)
{
    (void)cn_isa_current();
    if (level < CN_ISA_SCALAR || level > CN_ISA_AVX512) return -1;
    LONG capped = level < maximum ? level : maximum;
    InterlockedExchange(&effective, capped);
    return (int)capped;
}
#else
static const int requested = -1;
static const int maximum = CN_ISA_SCALAR;

cn_isa_level cn_isa_current(void) { return CN_ISA_SCALAR; }

int cn_isa_set(int level) { return level >= CN_ISA_SCALAR && level <= CN_ISA_AVX512 ? CN_ISA_SCALAR : -1; }
#endif

cn_status cn_isa_get(int *out)
{
    if (!out || (uintptr_t)out % _Alignof(int)) return CN_INVALID_ARGUMENT;
    out[0] = (int)cn_isa_current(); /* runs the probe before requested and maximum are read */
    out[1] = requested;
    out[2] = maximum;
    return CN_OK;
}
