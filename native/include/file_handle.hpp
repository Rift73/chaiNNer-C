/* Scoped Windows file handle ownership (GPL-3.0). */
#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace cn::files {

struct Handle {
    HANDLE handle;
    explicit Handle(HANDLE opened) : handle(opened) {}
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
    ~Handle() {
        if (valid()) CloseHandle(handle);
    }
    bool valid() const { return handle != INVALID_HANDLE_VALUE; }
};

} // namespace cn::files
