// OpenJPH, OpenEXR's HTJ2K codec, prints its errors, warnings and notes to stderr and stdout by default; OpenEXRCore
// already reports them through its own error codes, and a failed file declines quietly as every codec here does.
#include <cstdio>

#include "openjph/ojph_message.h"

extern "C" void cn_exr_quiet_openjph(void) {
    [[maybe_unused]] static const bool quiet = [] {
        ojph::set_info_stream(nullptr);
        ojph::set_warning_stream(nullptr);
        ojph::set_error_stream(nullptr);
        return true;
    }();
}
