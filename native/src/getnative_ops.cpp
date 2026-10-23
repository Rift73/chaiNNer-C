/* Get Native Resolution: C entries over an analysis session of the vendored Vernier 0.2 core. A session
 * runs one ISA unit (src/getnative_unit.cpp, built per level with the core), picked at creation by
 * cn_isa_current() (CHAINNER_C_ISA caps it); both units give bit-identical results. Entries return
 * 0, or 1 for invalid arguments, 3 for an allocation failure and 4 for any other failure (a
 * cancelled analysis among them), with the message in `error`. cn_getnative_cancel may be called
 * from another thread while a frame is analyzed; the analysis then stops at Vernier's next check.
 */
#include "chainner.h"
#include "getnative_unit.hpp"
#include "isa.h"
#include "linear_light.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace {
struct Session {
    const GetnativeUnit *unit;
    void *state;
    std::atomic_bool cancel{false};
    std::string text;  // the last render, until it is copied out
    Session(const GetnativeUnit *u, void *s) : unit(u), state(s) {}
    ~Session() { unit->destroy(state); }
};

void error_text(char *dst, size_t capacity, const char *message) noexcept {
    if (!dst || !capacity) return;
    const size_t length = std::min(capacity - 1, std::strlen(message));
    std::memcpy(dst, message, length);
    dst[length] = 0;
}

template <class F> int guarded(char *error, size_t capacity, F &&operation) noexcept {
    error_text(error, capacity, "");
    try {
        operation();
        return CN_OK;
    } catch (const std::invalid_argument &e) {
        error_text(error, capacity, e.what());
        return CN_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        error_text(error, capacity, "Get Native Resolution ran out of memory");
        return CN_ALLOCATION_FAILED;
    } catch (const std::exception &e) {
        error_text(error, capacity, e.what());
        return 4;
    } catch (...) {
        error_text(error, capacity, "Unexpected Get Native Resolution failure");
        return 4;
    }
}

Session &session_of(void *session) {
    if (!session) throw std::invalid_argument("No Get Native Resolution session");
    return *static_cast<Session *>(session);
}

/* The linear-light models as Vernier numbers them (1 BT.709 camera, 2 sRGB, 3 gamma 2.2, 4 gamma 2.4
 * with a zero black), indexed by model - 1, as linear_light.h's transfers. That header's curves are
 * Vernier's media.cpp linearize, in its order and with pow. */
constexpr int light_transfer[] = {CN_LIGHT_BT709, CN_LIGHT_SRGB, CN_LIGHT_GAMMA22, CN_LIGHT_GAMMA24};
}

extern "C" {
CN_EXPORT int cn_getnative_create(const cn_getnative_options *options, void **session, char *error,
                                  size_t capacity) {
    return guarded(error, capacity, [&] {
        if (!options || !session) throw std::invalid_argument("Invalid Get Native Resolution options");
        const auto *unit = cn_isa_current() >= CN_ISA_AVX2 ? cn_getnative_unit_avx2() : cn_getnative_unit_base();
        void *state = unit->create(*options);
        *session = new Session(unit, state);
    });
}

CN_EXPORT void cn_getnative_free(void *session) { delete static_cast<Session *>(session); }

CN_EXPORT void cn_getnative_cancel(void *session) {
    if (session) static_cast<Session *>(session)->cancel = true;
}

/* plane: width x height doubles (row-major), analyzed as `frame` describes it. */
CN_EXPORT int cn_getnative_add_frame(void *session, const double *plane, int width, int height,
                                     const cn_getnative_frame *frame, char *error, size_t capacity) {
    return guarded(error, capacity, [&] {
        auto &s = session_of(session);
        if (!frame) throw std::invalid_argument("Invalid frame facts");
        s.unit->add_frame(s.state, plane, width, height, *frame, &s.cancel);
    });
}

/* Renders text (GetnativeText `what`) into the session; *length is its size in bytes, without the
 * terminator, for cn_getnative_copy. sampling is the sampling report's JSON, batch_error the run's
 * error text (both for the summary). */
CN_EXPORT int cn_getnative_render(void *session, int what, int frame, int candidate, int planned, int completed,
                                  int domains, const char *sampling, const char *batch_error, size_t *length,
                                  char *error, size_t capacity) {
    return guarded(error, capacity, [&] {
        auto &s = session_of(session);
        if (what < 0 || what > 3 || !sampling || !batch_error || !length)
            throw std::invalid_argument("Invalid Get Native Resolution render");
        GetnativeBatch batch;
        batch.planned = planned;
        batch.completed = completed;
        batch.domains = domains;
        batch.sampling = sampling;
        batch.error = batch_error;
        s.text = s.unit->render(s.state, GetnativeText(what), frame, candidate, batch);
        *length = s.text.size();
    });
}

/* Copies the last render (capacity at least its length + 1) and releases it. */
CN_EXPORT int cn_getnative_copy(void *session, char *out, size_t out_capacity, char *error, size_t capacity) {
    return guarded(error, capacity, [&] {
        auto &s = session_of(session);
        if (!out || out_capacity <= s.text.size()) throw std::invalid_argument("Text buffer too small");
        std::memcpy(out, s.text.c_str(), s.text.size() + 1);
        std::string().swap(s.text);
    });
}

/* The best score per native size of analysis `frame`'s width (horizontal) or height scan; *count is
 * the number of sizes, of which at most `slots` are written. */
CN_EXPORT int cn_getnative_curve(void *session, int frame, int horizontal, int *sizes, double *scores, size_t slots,
                                 size_t *count, char *error, size_t capacity) {
    return guarded(error, capacity, [&] {
        auto &s = session_of(session);
        if (!count || (slots && (!sizes || !scores))) throw std::invalid_argument("Invalid curve buffers");
        *count = s.unit->curve(s.state, frame, horizontal != 0, sizes, scores, slots);
    });
}

/* A linear-light analysis plane from an interleaved B, G, R float image (chaiNNer's channel order):
 * weights[0] R + weights[1] G + weights[2] B of the channels through `domain`'s inverse transfer
 * (cn_light_linearize: signed, no clipping), as Vernier's linearFrame. */
CN_EXPORT int cn_getnative_linear_plane(const float *image, size_t pixels, int domain, const double *weights,
                                        double *plane) {
    if (!image || !plane || !weights || domain < 1 || domain > 4) return CN_INVALID_ARGUMENT;
    const int transfer = light_transfer[domain - 1];
    for (size_t i = 0; i < pixels; ++i) {
        const float *bgr = image + i * 3;
        plane[i] = weights[0] * cn_light_linearize(bgr[2], transfer) +
                   weights[1] * cn_light_linearize(bgr[1], transfer) +
                   weights[2] * cn_light_linearize(bgr[0], transfer);
    }
    return CN_OK;
}
}
