/* Native chaiNNer clipboard application path. Float32 quantization is in the
 * C kernel; this unit owns array/text validation, DIBV5 construction and Win32
 * publication. Adapted from chaiNNer-rs (MIT), arboard (MIT), and clipboard-win
 * (BSL-1.0), with notices in chainner_native.LICENSE.txt. No Rust clipboard call.
 * Failed HGLOBAL transfers use GlobalFree, correcting original leak branches. */
#define NOMINMAX
#include <windows.h>
#include <pybind11/numpy.h>
#include "clipboard_ops.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;
namespace {
using Bytes = std::vector<uint8_t>;
constexpr const char* inaccessible = "The native clipboard is not accessible due to being held by an other party.";
constexpr const char* prefix = "Unknown error while interacting with the clipboard: ";
constexpr const char* text_failure = "Could not place the specified text to the clipboard";

[[noreturn]] void fail(const std::string& message) { throw py::value_error(std::string(prefix) + message); }

std::string type_name(const py::object& value) { return py::str(py::type::of(value).attr("__name__")); }
const char* utf8(const py::object& value, const char* argument, Py_ssize_t& length) {
    if (!PyUnicode_Check(value.ptr()))
        throw py::type_error(std::string("argument '") + argument + "': '" + type_name(value) + "' object cannot be converted to 'PyString'");
    const char* result = PyUnicode_AsUTF8AndSize(value.ptr(), &length);
    if (!result) throw py::error_already_set();
    return result;
}

py::array image_array(const py::object& value) {
    const bool array = py::array::check_(value);
    py::array image;
    if (array) image = py::reinterpret_borrow<py::array>(value);
    if (array && (image.ndim() == 2 || image.ndim() == 3) && image.dtype().equal(py::dtype::of<float>())) return image;
    std::string message = "argument 'image': failed to extract enum PyImage ('D2 | D3')";
    for (int rank : {2, 3}) {
        const std::string dim = std::to_string(rank);
        message += "\n- variant D" + dim + " (D" + dim + "): TypeError: failed to extract field PyImage::D" + dim + ".0, caused by TypeError: ";
        if (!array) message += "'" + type_name(value) + "' object cannot be converted to 'PyArray<T, D>'";
        else if (image.ndim() != rank) message += "dimensionality mismatch:\n from=" + std::to_string(image.ndim()) + ", to=" + dim;
        else message += "type mismatch:\n from=" + std::string(py::str(image.dtype())) + ", to=float32";
    }
    throw py::type_error(message);
}

Bytes text_bytes(const py::object& text) {
    Py_ssize_t length;
    const char* value = utf8(text, "text", length);
    if (length > INT_MAX) fail(text_failure);
    const int count = length ? MultiByteToWideChar(CP_UTF8, 0, value, static_cast<int>(length), nullptr, 0) : 0;
    if (!count && length) fail(text_failure);
    Bytes bytes((static_cast<size_t>(count) + 1) * sizeof(wchar_t), 0);
    if (count && !MultiByteToWideChar(CP_UTF8, 0, value, static_cast<int>(length),
        reinterpret_cast<wchar_t*>(bytes.data()), count)) fail(text_failure);
    return bytes;
}

Bytes image_bytes(const py::object& value, const py::object& format) {
    // PyO3 extracts the image before the pixel-format string, then validates its
    // value before checking channels; keep that public failure priority.
    py::array image = image_array(value);
    Py_ssize_t format_length;
    const char* format_data = utf8(format, "pixel_format", format_length);
    const std::string pixel_format(format_data, static_cast<size_t>(format_length));
    if (pixel_format != "RGB" && pixel_format != "BGR")
        throw py::value_error("Invalid pixel format: " + pixel_format);
    const auto height = static_cast<size_t>(image.shape(0));
    const auto width = static_cast<size_t>(image.shape(1));
    const auto channels = image.ndim() == 3 ? static_cast<size_t>(image.shape(2)) : 1;
    if (channels != 1 && channels != 3 && channels != 4)
        throw py::value_error("Invalid number of channels: " + std::to_string(channels));
    if (height > LONG_MAX || width > LONG_MAX || (width && height > UINT32_MAX / 4 / width))
        throw py::value_error("Clipboard image dimensions exceed the Windows DIBV5 limits");
    const size_t bytes = width * height * 4;
    static_assert(sizeof(BITMAPV5HEADER) == 124);
    BITMAPV5HEADER header{};
    header.bV5Size = sizeof(header);
    header.bV5Width = static_cast<LONG>(width);
    header.bV5Height = static_cast<LONG>(height);
    header.bV5Planes = 1;
    header.bV5BitCount = 32;
    header.bV5Compression = BI_BITFIELDS;
    header.bV5SizeImage = static_cast<DWORD>(bytes);
    header.bV5RedMask = 0x00ff0000;
    header.bV5GreenMask = 0x0000ff00;
    header.bV5BlueMask = 0x000000ff;
    header.bV5AlphaMask = 0xff000000;
    header.bV5CSType = LCS_sRGB;
    header.bV5Intent = LCS_GM_IMAGES;
    Bytes output(sizeof(header) + bytes);
    std::memcpy(output.data(), &header, sizeof(header));
    int status;
    {
        py::gil_scoped_release release;
        status = cn_clipboard_pixels_f32(image.data(), height, width, channels,
            image.strides(0), image.strides(1), image.ndim() == 3 ? image.strides(2) : 0,
            pixel_format == "RGB", output.data() + sizeof(header), bytes);
    }
    if (status) throw py::value_error("Invalid or overflowing clipboard image buffer");
    return output;
}

class OpenClipboardScope {
public:
    OpenClipboardScope() {
        for (unsigned attempt = 0; attempt < 6; ++attempt) {
            if (OpenClipboard(nullptr)) return;
            if (attempt < 5) Sleep(5);
        }
        throw py::value_error(inaccessible);
    }
    ~OpenClipboardScope() { CloseClipboard(); }
    OpenClipboardScope(const OpenClipboardScope&) = delete;
    OpenClipboardScope& operator=(const OpenClipboardScope&) = delete;
};

class GlobalMemory {
    HGLOBAL handle_;
public:
    explicit GlobalMemory(size_t bytes) : handle_(GlobalAlloc(GHND, bytes)) {}
    ~GlobalMemory() { if (handle_) GlobalFree(handle_); }
    HGLOBAL get() const { return handle_; }
    void transfer() { handle_ = nullptr; }
    GlobalMemory(const GlobalMemory&) = delete;
    GlobalMemory& operator=(const GlobalMemory&) = delete;
};

std::string system_error(DWORD code) {
    wchar_t* buffer = nullptr;
    const DWORD count = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::string text;
    if (count && buffer) {
        const int size = WideCharToMultiByte(CP_UTF8, 0, buffer, static_cast<int>(count), nullptr, 0, nullptr, nullptr);
        text.resize(static_cast<size_t>(size));
        if (size) WideCharToMultiByte(CP_UTF8, 0, buffer, static_cast<int>(count), text.data(), size, nullptr, nullptr);
        LocalFree(buffer);
        while (!text.empty() && (text.back() == '\r' || text.back() == '\n' || text.back() == ' ')) text.pop_back();
    }
    return "OS error " + std::to_string(code) + (text.empty() ? "" : ": " + text);
}

void publish(const Bytes& bytes, bool image) {
    OpenClipboardScope clipboard;
    if (image && !EmptyClipboard()) {
        const DWORD error = GetLastError();
        fail("Failed to empty the clipboard. Got error code: " + system_error(error));
    }
    GlobalMemory memory(bytes.size());
    // Keep the installed binary's diagnostic text, including legacy source-line
    // numbers, while correcting its leaked/incorrectly-freed handle branches.
    if (!memory.get()) fail(image ? "Could not allocate global memory object. GlobalAlloc returned null at line 86." : text_failure);
    void* pointer = GlobalLock(memory.get());
    if (!pointer) fail(image ? "Could not lock the global memory object at line 94" : text_failure);
    std::memcpy(pointer, bytes.data(), bytes.size());
    GlobalUnlock(memory.get());
    if (!image) EmptyClipboard(); // Original text path ignores this result.
    if (!SetClipboardData(image ? CF_DIBV5 : CF_UNICODETEXT, memory.get()))
        fail(image ? "Call to `SetClipboardData` returned NULL at line 131" : text_failure);
    memory.transfer();
}

void write_text(const py::object& value) {
    Bytes bytes = text_bytes(value);
    py::gil_scoped_release release;
    publish(bytes, false);
}
void write_image(const py::object& value, const py::object& format) {
    Bytes bytes = image_bytes(value, format);
    py::gil_scoped_release release;
    publish(bytes, true);
}
void copy(const py::object& value, const py::object& array_type) {
    const int image = PyObject_IsInstance(value.ptr(), array_type.ptr());
    if (image < 0) throw py::error_already_set();
    if (image) write_image(value, py::str("BGR"));
    else write_text(value);
}
py::bytes as_bytes(const Bytes& value) { return py::bytes(reinterpret_cast<const char*>(value.data()), value.size()); }
}

void cn_bind_utility_clipboard(py::module_& module) {
    module.def("utility_copy_to_clipboard", &copy);
    module.def("utility_clipboard_text", &write_text);
    module.def("utility_clipboard_image", &write_image);
    module.def("utility_clipboard_text_bytes", [](const py::object& value) { return as_bytes(text_bytes(value)); });
    module.def("utility_clipboard_image_bytes", [](const py::object& value, const py::object& format) { return as_bytes(image_bytes(value, format)); });
}
