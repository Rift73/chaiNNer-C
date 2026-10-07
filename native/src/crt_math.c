/* Fills cn_crt (cn_crt_math.h) from the process's ucrtbase.dll as chainner_native.dll
 * loads, and cn_crt_probe, which compares each entry with this module's static CRT.
 */
#include "cn_crt_math.h"
#include <math.h>
#include <stddef.h>
#include <string.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

/* The static CRT's functions behind the two entries whose export is not their name. */
static double static_cabs(cn_crt_complex z)
{
#if defined(_MSC_VER)
    struct _complex value = {z.real, z.imag};
    return _cabs(value);
#else
    return hypot(z.real, z.imag);
#endif
}
#if defined(_MSC_VER)
#define CN_STATIC_HYPOTF _hypotf
#else
#define CN_STATIC_HYPOTF hypotf
#endif

typedef enum crt_shape { CRT_UNARY, CRT_BINARY, CRT_UNARY_F, CRT_BINARY_F, CRT_ABSOLUTE } crt_shape;
typedef union crt_function {
    double (*unary)(double);
    double (*binary)(double, double);
    float (*unary_f)(float);
    float (*binary_f)(float, float);
    double (*absolute)(cn_crt_complex);
} crt_function;
_Static_assert(sizeof(crt_function) == sizeof(double (*)(double)), "one function pointer size");

typedef struct crt_entry {
    const char *member, *export_name;
    crt_shape shape;
    size_t offset;      /* of the member in cn_crt_table */
    crt_function fixed; /* the static CRT's function */
} crt_entry;

#define CN_CRT_ENTRY(member, export_name, shape, kind, function) \
    {#member, export_name, shape, offsetof(cn_crt_table, member), {.kind = function}}
static const crt_entry entries[] = {
    CN_CRT_ENTRY(cos, "cos", CRT_UNARY, unary, cos),
    CN_CRT_ENTRY(sin, "sin", CRT_UNARY, unary, sin),
    CN_CRT_ENTRY(exp, "exp", CRT_UNARY, unary, exp),
    CN_CRT_ENTRY(log, "log", CRT_UNARY, unary, log),
    CN_CRT_ENTRY(log1p, "log1p", CRT_UNARY, unary, log1p),
    CN_CRT_ENTRY(log2, "log2", CRT_UNARY, unary, log2),
    CN_CRT_ENTRY(atan2, "atan2", CRT_BINARY, binary, atan2),
    CN_CRT_ENTRY(hypot, "hypot", CRT_BINARY, binary, hypot),
    CN_CRT_ENTRY(pow, "pow", CRT_BINARY, binary, pow),
    CN_CRT_ENTRY(cabs, "_cabs", CRT_ABSOLUTE, absolute, static_cabs),
    CN_CRT_ENTRY(expf, "expf", CRT_UNARY_F, unary_f, expf),
    CN_CRT_ENTRY(logf, "logf", CRT_UNARY_F, unary_f, logf),
    CN_CRT_ENTRY(powf, "powf", CRT_BINARY_F, binary_f, powf),
    CN_CRT_ENTRY(log10f, "log10f", CRT_UNARY_F, unary_f, log10f),
    CN_CRT_ENTRY(atan2f, "atan2f", CRT_BINARY_F, binary_f, atan2f),
    CN_CRT_ENTRY(hypotf, "_hypotf", CRT_BINARY_F, binary_f, CN_STATIC_HYPOTF),
    CN_CRT_ENTRY(asinf, "asinf", CRT_UNARY_F, unary_f, asinf),
    CN_CRT_ENTRY(cosf, "cosf", CRT_UNARY_F, unary_f, cosf),
    CN_CRT_ENTRY(sinf, "sinf", CRT_UNARY_F, unary_f, sinf),
};
#define CN_CRT_ENTRIES (sizeof entries / sizeof entries[0])
_Static_assert(CN_CRT_ENTRIES * sizeof(crt_function) == sizeof(cn_crt_table), "every member has an entry");

#if defined(_WIN32)
cn_crt_table cn_crt;

/* Writes "chainner_native.dll: <reason><name>" to stderr and refuses the load. */
static BOOL refuse(const char *reason, const char *name)
{
    const char *parts[] = {"chainner_native.dll: ", reason, name, "\n"};
    char message[160];
    size_t used = 0;
    for (size_t p = 0; p < sizeof parts / sizeof parts[0]; ++p) {
        size_t length = strlen(parts[p]);
        if (length > sizeof message - used) length = sizeof message - used;
        memcpy(message + used, parts[p], length);
        used += length;
    }
    HANDLE error = GetStdHandle(STD_ERROR_HANDLE);
    DWORD written = 0;
    if (error && error != INVALID_HANDLE_VALUE) WriteFile(error, message, (DWORD)used, &written, NULL);
    return FALSE;
}

/* Under the loader lock: GetModuleHandleW and GetProcAddress of exports that ucrtbase.dll
 * implements itself, so nothing is loaded. python3xx.dll has loaded ucrtbase.dll before
 * any extension or ctypes load reaches this DLL. */
BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    (void)instance; (void)reserved;
    if (reason != DLL_PROCESS_ATTACH) return TRUE;
    HMODULE module = GetModuleHandleW(L"ucrtbase.dll");
    if (!module) return refuse("ucrtbase.dll is not loaded in this process", "");
    _Static_assert(sizeof(FARPROC) == sizeof(crt_function), "function pointer size");
    for (size_t i = 0; i < CN_CRT_ENTRIES; ++i) {
        FARPROC address = GetProcAddress(module, entries[i].export_name);
        if (!address) return refuse("ucrtbase.dll has no export ", entries[i].export_name);
        memcpy((char *)&cn_crt + entries[i].offset, &address, sizeof address);
    }
    return TRUE;
}
#else
cn_crt_table cn_crt = {cos, sin, exp, log, log1p, log2, atan2, hypot, pow, static_cabs,
    expf, logf, powf, log10f, atan2f, hypotf, asinf, cosf, sinf};
#endif

static double evaluate(crt_shape shape, crt_function f, double x, double y)
{
    switch (shape) {
    case CRT_UNARY: return f.unary(x);
    case CRT_BINARY: return f.binary(x, y);
    case CRT_UNARY_F: return f.unary_f((float)x);
    case CRT_BINARY_F: return f.binary_f((float)x, (float)y);
    default: {
        cn_crt_complex z = {x, y};
        return f.absolute(z);
    }
    }
}

/* For the tests: evaluates the member `name` on count inputs through this module's
 * static CRT (from_static) and through cn_crt (from_table); a float entry rounds x and
 * y to float, and y is read only by two-argument entries. fma3 0 or 1 sets both CRTs'
 * FMA3 use for the call and restores it afterwards (process state: no concurrent
 * callers); -1 leaves it. used[0] and used[1] receive the static CRT's and cn_crt's FMA3
 * states during the call (-1 where this build cannot read them). */
CN_EXPORT cn_status cn_crt_probe(const char *name, int fma3, const double *x,
    const double *y, size_t count, double *from_static, double *from_table, int *used)
{
    if (!name || !used || fma3 < -1 || fma3 > 1) return CN_INVALID_ARGUMENT;
    const crt_entry *entry = NULL;
    for (size_t i = 0; i < CN_CRT_ENTRIES && !entry; ++i)
        if (!strcmp(entries[i].member, name)) entry = &entries[i];
    if (!entry) return CN_INVALID_ARGUMENT;
    int two_inputs = entry->shape == CRT_BINARY || entry->shape == CRT_BINARY_F
        || entry->shape == CRT_ABSOLUTE;
    if (count && (!x || (two_inputs && !y) || !from_static || !from_table)) return CN_INVALID_ARGUMENT;
    crt_function table;
    memcpy(&table, (const char *)&cn_crt + entry->offset, sizeof table);
#if defined(_MSC_VER) && defined(_M_X64)
    typedef int (*fma3_control)(int);
    typedef int (*fma3_state)(void);
    fma3_control set_table = NULL;
    fma3_state get_table = NULL;
    HMODULE module = GetModuleHandleW(L"ucrtbase.dll");
    FARPROC set_address = module ? GetProcAddress(module, "_set_FMA3_enable") : NULL;
    FARPROC get_address = module ? GetProcAddress(module, "_get_FMA3_enable") : NULL;
    if (!set_address || !get_address) return CN_INVALID_ARGUMENT;
    memcpy(&set_table, &set_address, sizeof set_table);
    memcpy(&get_table, &get_address, sizeof get_table);
    int static_before = _get_FMA3_enable(), table_before = get_table();
    used[0] = fma3 < 0 ? static_before : _set_FMA3_enable(fma3);
    used[1] = fma3 < 0 ? table_before : set_table(fma3);
#else
    if (fma3 >= 0) return CN_INVALID_ARGUMENT;
    used[0] = used[1] = -1;
#endif
    for (size_t i = 0; i < count; ++i) {
        double second = two_inputs ? y[i] : 0.0;
        from_static[i] = evaluate(entry->shape, entry->fixed, x[i], second);
        from_table[i] = evaluate(entry->shape, table, x[i], second);
    }
#if defined(_MSC_VER) && defined(_M_X64)
    if (fma3 >= 0) {
        _set_FMA3_enable(static_before);
        set_table(table_before);
    }
#endif
    return CN_OK;
}
