/* Altered C adaptation of resize 0.8.3 src/lib.rs and chaiNNer-rs
 * crates/image-ops/src/scale/filter.rs at commit
 * 6f6ead6064f81b4049d3deb803c9279c7950736b. Coefficient recycling, evaluation
 * order and float32 accumulators intentionally preserve their numerical ABI.
 *
 * The MIT License (MIT)
 * Copyright (c) 2015 PistonDevelopers
 * Copyright (c) 2023 Michael Schmidt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "chainner.h"
#include "cn_crt_math.h"
#include "parallel.h"
#include "resample_shared.h"
#include "scratch.h"
#include <fenv.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <stdatomic.h>
#endif
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

/* Shared C gamma implementation preserves the installed AVX/scalar split. */
cn_status cn_alpha_gamma(const float *src, float *out,
    size_t pixels, size_t channels, float gamma);

/* coefficient_line, resample_context and vertical_one live in resample_shared.h (SP4b
 * Task 7), shared with the AVX2 unit resample_avx2.c. */
typedef struct coefficient_entry {
    size_t count;
    uint32_t scale, offset;
    float *weights;
} coefficient_entry;

/* Each slot is exclusively leased until both parallel passes and gamma finish.
 * An idle slot retains at most 16 MiB of coefficients, so all four together retain
 * <= 64 MiB. Busy callers never wait for another resize: they use ordinary
 * temporary storage. No source/output pointer is kept between calls. The cache is
 * local to this process/DLL; only slot selection and return take the short lock.
 * The intermediate and linear images are one lease of the shared scratch pool
 * (scratch.h), whose slots are not these: no other kernel's lease evicts a plan. */
#define RESAMPLE_CACHE_SLOTS 4
#define RESAMPLE_RETAIN_BYTES ((size_t)16 * 1024 * 1024)
typedef struct resample_workspace {
    coefficient_line *lines;
    coefficient_entry *entries;
    size_t cache_size, coefficient_bytes;
    size_t height, width, target_height, target_width;
    int filter, rounding, ready;
    unsigned int float_controls;
} resample_workspace;

typedef struct workspace_slot {
    resample_workspace workspace;
    int leased;
} workspace_slot;

static workspace_slot workspace_slots[RESAMPLE_CACHE_SLOTS];
static size_t next_workspace;
/* Read-only diagnostics make reuse/bounded retention testable without timings.
 * Counters are updated only when a complete lease returns; scratch_allocations
 * counts the pool leases that allocated for a resize. */
static uint64_t coefficient_builds, coefficient_hits, scratch_allocations;
static uint64_t temporary_calls;
#ifdef _WIN32
static SRWLOCK workspace_lock = SRWLOCK_INIT;
static void lock_workspaces(void) { AcquireSRWLockExclusive(&workspace_lock); }
static void unlock_workspaces(void) { ReleaseSRWLockExclusive(&workspace_lock); }
#else
static atomic_flag workspace_lock = ATOMIC_FLAG_INIT;
static void lock_workspaces(void)
{
    while (atomic_flag_test_and_set_explicit(&workspace_lock, memory_order_acquire)) {}
}
static void unlock_workspaces(void)
{
    atomic_flag_clear_explicit(&workspace_lock, memory_order_release);
}
#endif

static unsigned int float_controls(void)
{
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    /* Coefficients depend on rounding and denormal controls, not sticky flags. */
    return _mm_getcsr() & 0xe040u;
#else
    return 0;
#endif
}

static int matching_workspace(const resample_workspace *ws, size_t height,
    size_t width, size_t target_height, size_t target_width, int filter,
    int rounding, unsigned int controls)
{
    return ws->ready && ws->height == height && ws->width == width &&
        ws->target_height == target_height && ws->target_width == target_width &&
        ws->filter == filter && ws->rounding == rounding && ws->float_controls == controls;
}

static workspace_slot *acquire_workspace(size_t height, size_t width,
    size_t target_height, size_t target_width, int filter, int rounding,
    unsigned int controls)
{
    workspace_slot *selected = NULL;
    lock_workspaces();
    for (size_t i = 0; i < RESAMPLE_CACHE_SLOTS; ++i) {
        workspace_slot *slot = &workspace_slots[i];
        if (!slot->leased && matching_workspace(&slot->workspace, height, width,
                target_height, target_width, filter, rounding, controls)) {
            selected = slot;
            break;
        }
    }
    if (!selected) {
        for (size_t i = 0; i < RESAMPLE_CACHE_SLOTS; ++i) {
            size_t index = (next_workspace + i) % RESAMPLE_CACHE_SLOTS;
            if (!workspace_slots[index].leased) {
                selected = &workspace_slots[index];
                next_workspace = (index + 1) % RESAMPLE_CACHE_SLOTS;
                break;
            }
        }
    }
    if (selected) selected->leased = 1;
    unlock_workspaces();
    return selected;
}

static void free_coefficients(resample_workspace *ws)
{
    if (ws->entries) {
        for (size_t i = 0; i < ws->cache_size; ++i) free(ws->entries[i].weights);
    }
    free(ws->entries);
    free(ws->lines);
    ws->entries = NULL;
    ws->lines = NULL;
    ws->cache_size = ws->coefficient_bytes = 0;
    ws->ready = 0;
}

static void release_workspace(workspace_slot *slot, resample_workspace *ws,
    uint64_t builds, uint64_t hits, uint64_t allocations)
{
    if (!slot || ws->coefficient_bytes > RESAMPLE_RETAIN_BYTES) free_coefficients(ws);
    lock_workspaces();
    coefficient_builds += builds;
    coefficient_hits += hits;
    scratch_allocations += allocations;
    if (slot) slot->leased = 0;
    else ++temporary_calls;
    unlock_workspaces();
}

/* Values: completed plan builds/hits, scratch allocations (the pool leases that
 * allocated for a resize), temporary calls, idle retained coefficient bytes, leased
 * slot count. The count argument guards the ABI. */
CN_EXPORT cn_status cn_resample_cache_info(uint64_t *values, size_t count)
{
    if (!values || count != 6) return CN_INVALID_ARGUMENT;
    lock_workspaces();
    values[0] = coefficient_builds;
    values[1] = coefficient_hits;
    values[2] = scratch_allocations;
    values[3] = temporary_calls;
    values[4] = values[5] = 0;
    for (size_t i = 0; i < RESAMPLE_CACHE_SLOTS; ++i) {
        const workspace_slot *slot = &workspace_slots[i];
        if (slot->leased) ++values[5];
        else values[4] += slot->workspace.coefficient_bytes;
    }
    unlock_workspaces();
    return CN_OK;
}

static uint32_t bits_of(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* cubic_bc's polynomial coefficients for one (B, C). The reference inlines cubic_bc with
 * literal B and C, so its compiler folds these expressions (round to nearest, in this
 * order); here they are static initializers, evaluated at translation time the same way.
 * Only the per-tap arithmetic follows the caller's rounding mode, as upstream's does. */
typedef struct cubic_coefficients { float a3, a2, a0, b3, b2, b1, b0; } cubic_coefficients;
#define CUBIC_BC(b, c) {12.0f - 9.0f * (b) - 6.0f * (c), -18.0f + 12.0f * (b) + 6.0f * (c), \
    6.0f - 2.0f * (b), -(b) - 6.0f * (c), 6.0f * (b) + 30.0f * (c), -12.0f * (b) - 48.0f * (c), \
    8.0f * (b) + 24.0f * (c)}
static const cubic_coefficients catrom = CUBIC_BC(0.0f, 0.5f), hermite = CUBIC_BC(0.0f, 0.0f),
    mitchell = CUBIC_BC(1.0f / 3.0f, 1.0f / 3.0f), bspline = CUBIC_BC(1.0f, 0.0f);

static float cubic(const cubic_coefficients *coefficients, float x)
{
    float a = fabsf(x), square = a * a, cube = square * a, k;
    if (a < 1.0f) {
        k = coefficients->a3 * cube;
        k += coefficients->a2 * square;
        k += coefficients->a0;
    } else if (a < 2.0f) {
        k = coefficients->b3 * cube;
        k += coefficients->b2 * square;
        k += coefficients->b1 * a;
        k += coefficients->b0;
    } else k = 0.0f;
    return k / 6.0f;
}

/* sinf and cosf, as the Gaussian's expf, are the process's ucrtbase.dll exports that the
 * real chainner_ext imports (cn_crt_math.h, group 1). */
static float sinc(float x)
{
    return x == 0.0f ? 1.0f : cn_crt.sinf(x) / x;
}

/* Filter IDs are the existing Python ResizeFilter values (not Rust IDs). */
static float kernel(int filter, float x)
{
    const float pi = 3.14159265358979323846f;
    switch (filter) {
    case 1:
        return fabsf(x) < 3.0f ? sinc(x * pi) * sinc((x / 3.0f) * pi) : 0.0f;
    case 2:
        return fmaxf(1.0f - fabsf(x), 0.0f);
    case 3:
        return cubic(&catrom, x);
    case 4:
        return fabsf(x) <= 0.5f ? 1.0f : 0.0f;
    case 5:
        return cubic(&hermite, x);
    case 6:
        return cubic(&mitchell, x);
    case 7:
        return cubic(&bspline, x);
    case 8:
        x = fabsf(x) * pi;
        return sinc(x) * (0.54f + 0.46f * cn_crt.cosf(x));
    case 9:
        x = fabsf(x) * pi;
        return sinc(x) * (0.5f + 0.5f * cn_crt.cosf(x));
    case 10: {
        x = fabsf(x);
        if (x > 2.0f) return 0.0f;
        int n = (int)(2.0f + x);
        float value = 1.0f;
        for (int i = 0; i < 4; ++i) {
            float d = (float)(n - i);
            if (d != 0.0f) value *= (d - x) / d;
        }
        return value;
    }
    default:
        /* gaussian(x, 0.5): the reference's compiler folds 1 / (sqrt(2 pi) * 0.5), each
         * step rounded to nearest, to this constant, and computes -x.powi(2) as (-x) * x
         * (LLVM moves a product's negation onto an operand: equal under round to nearest,
         * not under the directed modes). Its exp is the process's ucrtbase.dll expf that
         * the reference calls (cn_crt_math.h). */
        return 0x1.988454p-1f * cn_crt.expf(-x * x / 0.5f);
    }
}

static size_t cache_slot(size_t count, uint32_t scale, uint32_t offset, size_t mask)
{
    uint64_t hash = (uint64_t)count ^ ((uint64_t)scale << 32) ^ offset;
    hash ^= hash >> 30;
    hash *= UINT64_C(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash *= UINT64_C(0x94d049bb133111eb);
    hash ^= hash >> 31;
    return (size_t)hash & mask;
}

static cn_status coefficients(size_t source, size_t target, int filter,
    coefficient_line *lines, coefficient_entry *cache, size_t cache_size,
    size_t *allocated_bytes)
{
    float support = filter == 1 || filter == 11 ? 3.0f :
        (filter == 3 || filter == 6 || filter == 7 || filter == 10 ? 2.0f : 1.0f);
    double ratio = (double)source / (double)target;
    double scale = ratio > 1.0 ? ratio : 1.0;
    double radius = ceil((double)support * scale);
    uint32_t scale_bits = bits_of((float)scale);
    for (size_t x = 0; x < target; ++x) {
        double center = ((double)x + 0.5) * ratio - 0.5;
        double first = ceil(center - radius), last = floor(center + radius);
        size_t start = first <= 0.0 ? 0 : first >= (double)(source - 1) ? source - 1 : (size_t)first;
        size_t end = last <= 0.0 ? 0 : last >= (double)(source - 1) ? source - 1 : (size_t)last;
        if (end < start) end = start;
        size_t count = end - start + 1;
        uint32_t offset = bits_of((float)start - (float)center);
        size_t slot = cache_slot(count, scale_bits, offset, cache_size - 1);
        while (cache[slot].weights && (cache[slot].count != count ||
            cache[slot].scale != scale_bits || cache[slot].offset != offset))
            slot = (slot + 1) & (cache_size - 1);
        coefficient_entry *entry = &cache[slot];
        if (!entry->weights) {
            if (count > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
            if (count * sizeof(float) > SIZE_MAX - *allocated_bytes) return CN_SIZE_OVERFLOW;
            float *weights = malloc(count * sizeof(float));
            if (!weights) return CN_ALLOCATION_FAILED;
            double sum = 0.0;
            for (size_t i = start; i <= end; ++i)
                sum += (double)kernel(filter, (float)(((double)i - center) / scale));
            for (size_t i = 0; i < count; ++i) {
                float n = (float)(((double)(start + i) - center) / scale);
                if (n > support) n = support;
                if (n < -support) n = -support;
                weights[i] = (float)((double)kernel(filter, n) / sum);
            }
            entry->count = count;
            entry->scale = scale_bits;
            entry->offset = offset;
            entry->weights = weights;
            *allocated_bytes += count * sizeof(float);
        }
        lines[x].start = start;
        lines[x].count = count;
        lines[x].weights = entry->weights;
    }
    return CN_OK;
}

/* Horizontal outputs of one source row into out, one function per channel count
 * (SP4b D14). Output x of channel k keeps B3's sequence: acc = 0, then
 * acc += row[(start + i) * channels + k] * weights[i] for each tap i in order. The
 * tap's pixel pointer is hoisted and the channels of a pixel accumulate side by side,
 * each in its own sum. */
static void horizontal_row1(const float *row, const coefficient_line *lines,
    size_t width, float *out)
{
    for (size_t x = 0; x < width; ++x) {
        const float *pixel = row + lines[x].start, *weights = lines[x].weights;
        size_t count = lines[x].count;
        float a0 = 0.0f;
        for (size_t i = 0; i < count; ++i)
            a0 += pixel[i] * weights[i];
        out[x] = a0;
    }
}

static void horizontal_row2(const float *row, const coefficient_line *lines,
    size_t width, float *out)
{
    for (size_t x = 0; x < width; ++x, out += 2) {
        const float *pixel = row + lines[x].start * 2, *weights = lines[x].weights;
        size_t count = lines[x].count;
        float a0 = 0.0f, a1 = 0.0f;
        for (size_t i = 0; i < count; ++i, pixel += 2) {
            float weight = weights[i];
            a0 += pixel[0] * weight;
            a1 += pixel[1] * weight;
        }
        out[0] = a0;
        out[1] = a1;
    }
}

static void horizontal_row3(const float *row, const coefficient_line *lines,
    size_t width, float *out)
{
    for (size_t x = 0; x < width; ++x, out += 3) {
        const float *pixel = row + lines[x].start * 3, *weights = lines[x].weights;
        size_t count = lines[x].count;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        for (size_t i = 0; i < count; ++i, pixel += 3) {
            float weight = weights[i];
            a0 += pixel[0] * weight;
            a1 += pixel[1] * weight;
            a2 += pixel[2] * weight;
        }
        out[0] = a0;
        out[1] = a1;
        out[2] = a2;
    }
}

static void horizontal_row4(const float *row, const coefficient_line *lines,
    size_t width, float *out)
{
    for (size_t x = 0; x < width; ++x, out += 4) {
        const float *pixel = row + lines[x].start * 4, *weights = lines[x].weights;
        size_t count = lines[x].count;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (size_t i = 0; i < count; ++i, pixel += 4) {
            float weight = weights[i];
            a0 += pixel[0] * weight;
            a1 += pixel[1] * weight;
            a2 += pixel[2] * weight;
            a3 += pixel[3] * weight;
        }
        out[0] = a0;
        out[1] = a1;
        out[2] = a2;
        out[3] = a3;
    }
}

static void horizontal_range(void *opaque, size_t begin, size_t end)
{
    const resample_context *ctx = opaque;
    size_t channels = ctx->channels, width = ctx->target_width;
    size_t source_stride = ctx->source_width * channels, stride = width * channels;
    for (size_t y = begin; y < end; ++y) {
        const float *row = ctx->source + y * source_stride;
        float *out = ctx->intermediate + y * stride;
        /* cn_resample_filtered admits 1-4 channels, so default is 4. */
        switch (channels) {
        case 1: horizontal_row1(row, ctx->horizontal, width, out); break;
        case 2: horizontal_row2(row, ctx->horizontal, width, out); break;
        case 3: horizontal_row3(row, ctx->horizontal, width, out); break;
        default: horizontal_row4(row, ctx->horizontal, width, out); break;
        }
    }
}

/* Output rows [begin, end): at avx2 and above through the AVX2 unit, else each output
 * through vertical_one. */
static void vertical_range(void *opaque, size_t begin, size_t end)
{
    const resample_context *ctx = opaque;
    size_t stride = ctx->target_width * ctx->channels;
    for (size_t y = begin; y < end; ++y) {
#if defined(_MSC_VER) && defined(_M_X64)
        if (ctx->isa >= CN_ISA_AVX2) {
            cn_resample_vertical_avx2(ctx, y, 0, stride);
            continue;
        }
#endif
        const coefficient_line *line = &ctx->vertical[y];
        for (size_t x = 0; x < stride; ++x)
            ctx->out[y * stride + x] = vertical_one(ctx, line, x);
    }
}

CN_EXPORT cn_status cn_resample_filtered(const float *source, float *out,
    size_t height, size_t width, size_t channels, size_t target_height,
    size_t target_width, int filter, int gamma, int vector_clip)
{
    if (!source || !out || !height || !width || !target_height || !target_width ||
        channels < 1 || channels > 4 || filter < 1 || filter > 11 ||
        gamma < 0 || gamma > 1 || vector_clip < 0 || vector_clip > 1)
        return CN_INVALID_ARGUMENT;
    if (height > INT32_MAX || width > INT32_MAX ||
        target_height > UINT32_MAX || target_width > UINT32_MAX ||
        width > SIZE_MAX / sizeof(float) / channels / height ||
        target_width > SIZE_MAX / sizeof(float) / channels / height ||
        target_width > SIZE_MAX / sizeof(float) / channels / target_height ||
        target_width > SIZE_MAX - target_height)
        return CN_SIZE_OVERFLOW;
    size_t total_lines = target_width + target_height;
    if (total_lines > SIZE_MAX / sizeof(coefficient_line) || total_lines > SIZE_MAX / 2)
        return CN_SIZE_OVERFLOW;
    size_t cache_size = 1;
    while (cache_size < total_lines * 2) {
        if (cache_size > SIZE_MAX / 2) return CN_SIZE_OVERFLOW;
        cache_size *= 2;
    }
    if (cache_size > SIZE_MAX / sizeof(coefficient_entry)) return CN_SIZE_OVERFLOW;
    size_t line_bytes = total_lines * sizeof(coefficient_line);
    size_t entry_bytes = cache_size * sizeof(coefficient_entry);
    if (line_bytes > SIZE_MAX - entry_bytes) return CN_SIZE_OVERFLOW;
    int rounding = fegetround();
    unsigned int controls = float_controls();
    workspace_slot *slot = acquire_workspace(height, width, target_height,
        target_width, filter, rounding, controls);
    resample_workspace temporary = {0};
    resample_workspace *ws = slot ? &slot->workspace : &temporary;
    uint64_t builds = 0, hits = 0, allocations = 0;
    cn_scratch scratch = {NULL, 0, -1};
    cn_status status = CN_ALLOCATION_FAILED;
    if (matching_workspace(ws, height, width, target_height, target_width,
            filter, rounding, controls)) ++hits;
    else {
        free_coefficients(ws);
        ws->lines = malloc(line_bytes);
        ws->entries = calloc(cache_size, sizeof(coefficient_entry));
        ws->cache_size = cache_size;
        ws->coefficient_bytes = line_bytes + entry_bytes;
        if (!ws->lines || !ws->entries) goto cleanup;
        status = coefficients(width, target_width, filter, ws->lines,
            ws->entries, cache_size, &ws->coefficient_bytes);
        if (status != CN_OK) goto cleanup;
        status = coefficients(height, target_height, filter, ws->lines + target_width,
            ws->entries, cache_size, &ws->coefficient_bytes);
        if (status != CN_OK) goto cleanup;
        ws->height = height;
        ws->width = width;
        ws->target_height = target_height;
        ws->target_width = target_width;
        ws->filter = filter;
        ws->rounding = rounding;
        ws->float_controls = controls;
        ws->ready = 1;
        ++builds;
    }
    /* One pooled lease: the intermediate image, then with gamma the linear source,
     * each at a 64-byte offset. Both products fit size_t (the checks above); a
     * rounded sum that does not fails as the separate mallocs did. */
    size_t total = 0, intermediate_at = 0, linear_at = 0;
    if (!cn_scratch_carve(&total, height * target_width * channels * sizeof(float),
            &intermediate_at) ||
        (gamma && !cn_scratch_carve(&total, height * width * channels * sizeof(float),
            &linear_at))) {
        status = CN_ALLOCATION_FAILED;
        goto cleanup;
    }
    int allocated = 0;
    status = cn_scratch_lease(total, &scratch, &allocated);
    allocations += (uint64_t)allocated;
    if (status != CN_OK) goto cleanup;
    float *intermediate = (float *)((char *)scratch.data + intermediate_at);
    float *linear = gamma ? (float *)((char *)scratch.data + linear_at) : NULL;
    if (gamma) {
        status = cn_alpha_gamma(source, linear, height * width, channels, 2.2f);
        if (status != CN_OK) goto cleanup;
    }
    resample_context ctx = {gamma ? linear : source, intermediate, out, ws->lines,
        ws->lines + target_width, width, target_width, channels,
        filter == 2 ? 0 : !gamma && vector_clip ? 2 : 1, cn_isa_current()};
    size_t grain = 65536 / target_width / channels;
    if (grain == 0) grain = 1;
    status = cn_parallel_for(height, grain, horizontal_range, &ctx);
    if (status != CN_OK) goto cleanup;
    status = cn_parallel_for(target_height, grain, vertical_range, &ctx);
    if (status != CN_OK) goto cleanup;
    if (gamma)
        status = cn_alpha_gamma(out, out, target_height * target_width, channels, 1.0f / 2.2f);
cleanup:
    cn_scratch_release(&scratch);
    if (!ws->ready) free_coefficients(ws);
    release_workspace(slot, ws, builds, hits, allocations);
    return status;
}
