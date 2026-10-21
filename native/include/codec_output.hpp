/* Scoped ownership of the codecs' growing output buffer (GPL-3.0). */
#pragma once

#include "image_codecs.h"

namespace cn::codecs {

struct Output {
    cn_codec_output value{};
    Output() = default;
    Output(const Output &) = delete;
    Output &operator=(const Output &) = delete;
    ~Output() { cn_codec_output_free(&value); }
};

} // namespace cn::codecs
