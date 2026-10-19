/* Atomic file writes (GPL-3.0); see native/include/atomic_file.hpp. */
#include "atomic_file.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace cn::files {
namespace {

constexpr size_t block = size_t(1) << 20;

std::atomic<uint64_t> sequence{0};

// A file opened for reading with the sharing Python's open(path, "rb") grants; invalid when it cannot be opened (a
// missing file or a folder).
class Reader {
public:
    explicit Reader(const std::filesystem::path &path)
        : handle_(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr)) {}
    Reader(const Reader &) = delete;
    Reader &operator=(const Reader &) = delete;
    ~Reader() {
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
    }
    // The file's size, or -1 when it cannot be read.
    int64_t size() const {
        LARGE_INTEGER size;
        return handle_ != INVALID_HANDLE_VALUE && GetFileSizeEx(handle_, &size) ? size.QuadPart : -1;
    }
    // The next n bytes; false at a read error or an early end.
    bool read(uint8_t *out, size_t n) const {
        while (n) {
            DWORD got = 0;
            if (!ReadFile(handle_, out, DWORD(std::min(n, block)), &got, nullptr) || !got) return false;
            out += got;
            n -= got;
        }
        return true;
    }

private:
    HANDLE handle_;
};

// Whether `final` holds exactly `data`: the size first, then 1 MiB blocks; false when it cannot be read.
bool holds(const std::filesystem::path &final, std::span<const uint8_t> data) {
    const Reader file(final);
    if (file.size() != int64_t(data.size())) return false;
    std::vector<uint8_t> buffer(std::min(data.size(), block));
    for (size_t done = 0; done < data.size();) {
        const size_t n = std::min(data.size() - done, block);
        if (!file.read(buffer.data(), n) || std::memcmp(buffer.data(), data.data() + done, n) != 0) return false;
        done += n;
    }
    return true;
}

// Whether two files hold the same bytes, compared as holds() compares; false when either cannot be read.
bool same_bytes(const std::filesystem::path &a, const std::filesystem::path &b) {
    const Reader first(a), second(b);
    const int64_t size = first.size();
    if (size < 0 || second.size() != size) return false;
    std::vector<uint8_t> x(size_t(std::min<int64_t>(size, int64_t(block)))), y(x.size());
    for (int64_t done = 0; done < size;) {
        const size_t n = size_t(std::min<int64_t>(size - done, int64_t(block)));
        if (!first.read(x.data(), n) || !second.read(y.data(), n) || std::memcmp(x.data(), y.data(), n) != 0)
            return false;
        done += int64_t(n);
    }
    return true;
}

// `<final>.chainner-c-<pid>-<seq>.tmp`, skipping a name some file already has (another process's leftover with this
// process's id): this process never overwrites or deletes a file it did not create.
std::filesystem::path temporary_name(const std::filesystem::path &final) {
    const std::wstring prefix = final.native() + L".chainner-c-" + std::to_wstring(GetCurrentProcessId()) + L"-";
    for (;;) {
        std::filesystem::path name(prefix + std::to_wstring(sequence.fetch_add(1)) + L".tmp");
        if (GetFileAttributesW(name.c_str()) == INVALID_FILE_ATTRIBUTES) return name;
    }
}

std::string describe(int error, bool windows) {
    return (windows ? "Windows error " : "errno ") + std::to_string(error) + " writing a file";
}

// The errno the CRT gives a Windows error (the UCRT's _dosmaperr table, errno.cpp), as open() reports a failed
// CreateFileW; 0 for one the table lacks, where the CRT falls back to EINVAL.
int errno_of(DWORD error) {
    struct Entry {
        DWORD error;
        int errno_value;
    };
    static constexpr Entry table[] = {
        {ERROR_INVALID_FUNCTION, EINVAL},  {ERROR_FILE_NOT_FOUND, ENOENT},      {ERROR_PATH_NOT_FOUND, ENOENT},
        {ERROR_TOO_MANY_OPEN_FILES, EMFILE}, {ERROR_ACCESS_DENIED, EACCES},     {ERROR_INVALID_HANDLE, EBADF},
        {ERROR_ARENA_TRASHED, ENOMEM},     {ERROR_NOT_ENOUGH_MEMORY, ENOMEM},   {ERROR_INVALID_BLOCK, ENOMEM},
        {ERROR_BAD_ENVIRONMENT, E2BIG},    {ERROR_BAD_FORMAT, ENOEXEC},         {ERROR_INVALID_ACCESS, EINVAL},
        {ERROR_INVALID_DATA, EINVAL},      {ERROR_INVALID_DRIVE, ENOENT},       {ERROR_CURRENT_DIRECTORY, EACCES},
        {ERROR_NOT_SAME_DEVICE, EXDEV},    {ERROR_NO_MORE_FILES, ENOENT},       {ERROR_LOCK_VIOLATION, EACCES},
        {ERROR_BAD_NETPATH, ENOENT},       {ERROR_NETWORK_ACCESS_DENIED, EACCES}, {ERROR_BAD_NET_NAME, ENOENT},
        {ERROR_FILE_EXISTS, EEXIST},       {ERROR_CANNOT_MAKE, EACCES},         {ERROR_FAIL_I24, EACCES},
        {ERROR_INVALID_PARAMETER, EINVAL}, {ERROR_NO_PROC_SLOTS, EAGAIN},       {ERROR_DRIVE_LOCKED, EACCES},
        {ERROR_BROKEN_PIPE, EPIPE},        {ERROR_DISK_FULL, ENOSPC},           {ERROR_INVALID_TARGET_HANDLE, EBADF},
        {ERROR_WAIT_NO_CHILDREN, ECHILD},  {ERROR_CHILD_NOT_COMPLETE, ECHILD},  {ERROR_DIRECT_ACCESS_HANDLE, EBADF},
        {ERROR_NEGATIVE_SEEK, EINVAL},     {ERROR_SEEK_ON_DEVICE, EACCES},      {ERROR_DIR_NOT_EMPTY, ENOTEMPTY},
        {ERROR_NOT_LOCKED, EACCES},        {ERROR_BAD_PATHNAME, ENOENT},        {ERROR_MAX_THRDS_REACHED, EAGAIN},
        {ERROR_LOCK_FAILED, EACCES},       {ERROR_ALREADY_EXISTS, EEXIST},      {ERROR_FILENAME_EXCED_RANGE, ENOENT},
        {ERROR_NESTING_NOT_ALLOWED, EAGAIN}, {ERROR_NOT_ENOUGH_QUOTA, ENOMEM},
    };
    for (const Entry &entry : table)
        if (entry.error == error) return entry.errno_value;
    if (error >= ERROR_WRITE_PROTECT && error <= ERROR_SHARING_BUFFER_EXCEEDED) return EACCES;
    if (error >= ERROR_INVALID_STARTING_CODESEG && error <= ERROR_INFLOOP_IN_RELOC_CHAIN) return ENOEXEC;
    return 0;
}

// A failed Windows call on the final path: open()'s errno where the CRT maps the error, else the Windows error.
FileError os_error(DWORD error, const std::filesystem::path &final) {
    const int mapped = errno_of(error);
    return mapped ? FileError(mapped, false, final) : FileError(int(error), true, final);
}

// open(final, "wb") refuses a folder with EACCES before it writes anything; so does this, before any temporary file.
void refuse_folder(const std::filesystem::path &final) {
    const DWORD attributes = GetFileAttributesW(final.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES && attributes & FILE_ATTRIBUTE_DIRECTORY)
        throw FileError(EACCES, false, final);
}

} // namespace

FileError::FileError(int error, bool windows, std::filesystem::path path)
    : std::runtime_error(describe(error, windows)), error_(error), windows_(windows), path_(std::move(path)) {}

AtomicTarget::AtomicTarget(std::filesystem::path final) : final_(std::move(final)), temp_(temporary_name(final_)) {}

AtomicTarget::~AtomicTarget() { abandon(); }

bool AtomicTarget::commit() {
    if (state_ != State::open) throw std::logic_error("The temporary file was already committed or abandoned");
    if (same_bytes(temp_, final_)) {
        DeleteFileW(temp_.c_str());
        state_ = State::committed;
        return false;
    }
    rename();
    return true;
}

void AtomicTarget::abandon() noexcept {
    if (state_ == State::committed) return;
    DeleteFileW(temp_.c_str());
    state_ = State::abandoned;
}

void AtomicTarget::rename() {
    if (!MoveFileExW(temp_.c_str(), final_.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        const DWORD error = GetLastError();
        abandon();
        throw os_error(error, final_);
    }
    state_ = State::committed;
}

bool write_atomic(const std::filesystem::path &final, std::span<const uint8_t> data) {
    if (holds(final, data)) return false;
    refuse_folder(final);
    AtomicTarget target(final);
    // The CRT's low-level I/O as Python's open(final, "wb") uses it, so its errno is the one Python reports; never
    // over an existing file.
    int fd = -1;
    if (const errno_t error = _wsopen_s(&fd, target.temp().c_str(),
                                        _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY | _O_NOINHERIT, _SH_DENYNO,
                                        _S_IREAD | _S_IWRITE))
        throw FileError(error, false, final);
    for (size_t done = 0; done < data.size();) {
        const int written = _write(fd, data.data() + done, unsigned(std::min(data.size() - done, size_t(1) << 30)));
        if (written <= 0) {
            const int error = errno;
            _close(fd);
            throw FileError(error, false, final);
        }
        done += size_t(written);
    }
    if (_close(fd) != 0) throw FileError(errno, false, final);
    target.rename();
    return true;
}

bool move_atomic(const std::filesystem::path &source, const std::filesystem::path &final) {
    if (same_bytes(source, final)) return false;
    refuse_folder(final);
    AtomicTarget target(final);
    // A rename on one volume; across volumes MoveFileExW copies the file and deletes the source.
    if (!MoveFileExW(source.c_str(), target.temp().c_str(), MOVEFILE_COPY_ALLOWED))
        throw os_error(GetLastError(), final);
    target.rename();
    return true;
}

} // namespace cn::files
