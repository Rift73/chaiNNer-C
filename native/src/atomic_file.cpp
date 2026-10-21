/* Atomic file writes (GPL-3.0); see native/include/atomic_file.hpp. */
#include "atomic_file.hpp"

#include "file_handle.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <chrono>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>

namespace cn::files {
namespace {

constexpr size_t block = size_t(1) << 20;

std::atomic<uint64_t> sequence{0};

// Whether Windows takes paths past MAX_PATH in this process (the LongPathsEnabled setting and a long-path-aware
// manifest), as the libraries handed a temporary file's name see it.
bool long_paths() {
    using Query = BOOLEAN(NTAPI *)();
    static const bool enabled = [] {
        const auto query =
            reinterpret_cast<Query>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlAreLongPathsEnabled"));
        return query && query();
    }();
    return enabled;
}

// path made absolute as Windows does (GetFullPathNameW); as given when that fails.
std::wstring full(const std::filesystem::path &path) {
    const DWORD size = GetFullPathNameW(path.c_str(), 0, nullptr, nullptr);
    std::wstring text(size, L'\0');
    const DWORD written = size ? GetFullPathNameW(path.c_str(), size, text.data(), nullptr) : 0;
    if (!written || written >= size) return path.native();
    text.resize(written);
    return text;
}

// path in the \\?\ form (\\?\UNC\ for a share) once its full path reaches MAX_PATH, which every Windows call takes
// whatever the long-path setting; as given when shorter or already in a device form.
std::filesystem::path extended(const std::filesystem::path &path) {
    const std::wstring &text = path.native();
    if (text.starts_with(L"\\\\?\\") || text.starts_with(L"\\\\.\\")) return path;
    const std::wstring absolute = full(path);
    if (absolute.size() < MAX_PATH) return path;
    if (absolute.starts_with(L"\\\\")) return std::filesystem::path(L"\\\\?\\UNC\\" + absolute.substr(2));
    return std::filesystem::path(L"\\\\?\\" + absolute);
}

// A file opened for reading with the sharing Python's open(path, "rb") grants; invalid when it cannot be opened (a
// missing file or a folder).
class Reader {
public:
    explicit Reader(const std::filesystem::path &path)
        : handle_(CreateFileW(extended(path).c_str(), GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr)) {}
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

// Whether Windows can create a file of this name: 1 to 255 UTF-16 units, no control character and none of
// <>:"/\|?* (wildcards to a search, so the name must never reach one).
bool valid_name(const std::wstring &name) {
    return !name.empty() && name.size() <= 255 && std::none_of(name.begin(), name.end(), [](wchar_t c) {
               return c < 32 || std::wcschr(L"<>:\"/\\|?*", c) != nullptr;
           });
}

// The temporary name for sequence number `seq` (the ruled format, fable-decisions P8):
// `<final>.chainner-c-<pid>-<seq>.tmp`; `.cnc-<pid>-<seq>.tmp` in final's folder where that name would pass the
// 255-unit name limit or, with long paths off, MAX_PATH. In the \\?\ form once its full path reaches MAX_PATH.
// The short name only for a valid final name (valid_name()): an invalid one keeps the ruled name, whose creation then
// fails as open(final, "wb") fails, before anything is written.
std::filesystem::path temporary_candidate(const std::filesystem::path &final, uint64_t seq) {
    const std::wstring tail = std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(seq) + L".tmp";
    const std::filesystem::path ruled(final.native() + L".chainner-c-" + tail);
    if (!valid_name(final.filename().native()) ||
        (ruled.filename().native().size() <= 255 && (long_paths() || full(ruled).size() < MAX_PATH)))
        return extended(ruled);
    return extended(final.parent_path() / (L".cnc-" + tail));
}

// temporary_candidate(), skipping a name some file already has (another process's leftover with this process's id):
// this process never overwrites or deletes a file it did not create.
std::filesystem::path temporary_name(const std::filesystem::path &final) {
    for (;;) {
        std::filesystem::path name = temporary_candidate(final, sequence.fetch_add(1));
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
        {ERROR_INVALID_NAME, EINVAL}, // not in the table: the CRT's EINVAL fallback, as open() reports it
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

// Renames `from` onto `to` by a DELETE handle on `from` with POSIX semantics: the name moves at once, also over a file
// other programs hold open sharing delete (an antivirus, the indexer, a thumbnailer, cloud sync; they keep reading the
// old bytes), while a holder that does not share delete still keeps the file. Both names in the \\?\ form past
// MAX_PATH. 0, or the Windows error.
DWORD rename_by_handle(const std::filesystem::path &from, const std::filesystem::path &to) {
    const std::wstring name = extended(to).native();
    std::vector<uint8_t> buffer(offsetof(FILE_RENAME_INFO, FileName) + (name.size() + 1) * sizeof(wchar_t));
    auto *info = reinterpret_cast<FILE_RENAME_INFO *>(buffer.data());
    info->Flags = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
    info->RootDirectory = nullptr;
    info->FileNameLength = DWORD(name.size() * sizeof(wchar_t));
    std::memcpy(info->FileName, name.c_str(), name.size() * sizeof(wchar_t));
    const Handle file(CreateFileW(extended(from).c_str(), DELETE | SYNCHRONIZE,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0,
                                  nullptr));
    if (!file.valid()) return GetLastError();
    return SetFileInformationByHandle(file.handle, FileRenameInfoEx, info, DWORD(buffer.size())) ? 0 : GetLastError();
}

// attempt() (0, or a Windows error) again while it fails with access denied or a sharing violation (a scanner holding a
// file a moment, a rename racing another), for up to 250 ms. 0, or the last Windows error.
template <class Attempt> DWORD retrying(Attempt attempt) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    for (;;) {
        const DWORD error = attempt();
        if ((error != ERROR_ACCESS_DENIED && error != ERROR_SHARING_VIOLATION) ||
            std::chrono::steady_clock::now() >= deadline)
            return error;
        Sleep(10);
    }
}

// rename_by_handle(), or MoveFileExW(MOVEFILE_REPLACE_EXISTING) where the volume or Windows lacks POSIX renames (FAT,
// exFAT, some network shares, Windows before 10 1607), retrying(). 0, or the last Windows error.
DWORD replace(const std::filesystem::path &from, const std::filesystem::path &to) {
    return retrying([&] {
        const DWORD error = rename_by_handle(from, to);
        if (error != ERROR_INVALID_PARAMETER && error != ERROR_NOT_SUPPORTED) return error;
        return MoveFileExW(extended(from).c_str(), extended(to).c_str(), MOVEFILE_REPLACE_EXISTING) ? DWORD(0)
                                                                                                   : GetLastError();
    });
}

// open(final, "wb") refuses a folder and a read-only, hidden or system file (CREATE_ALWAYS names neither attribute)
// with EACCES before it writes anything; so does this, before any temporary file and again before the rename (which
// would replace a hidden or system file and drop its attributes).
void refuse_unwritable(const std::filesystem::path &final) {
    const DWORD attributes = GetFileAttributesW(extended(final).c_str());
    constexpr DWORD refused =
        FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM;
    if (attributes != INVALID_FILE_ATTRIBUTES && attributes & refused) throw FileError(EACCES, false, final);
}

// final with the case its existing file has on disk, which open(final, "wb") keeps, read from a handle on that very
// file (never a search, to which ? * < > are wildcards); as given when no file has it, or when the handle's name is
// not the given one in another case (a file another writer replaced meanwhile lives on under its file id in
// \$Extend\$Deleted).
std::filesystem::path on_disk(const std::filesystem::path &final) {
    std::wstring name(MAX_PATH, L'\0');
    const Handle file(CreateFileW(extended(final).c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                  nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
                                  nullptr));
    if (!file.valid()) return final;
    DWORD size = GetFinalPathNameByHandleW(file.handle, name.data(), DWORD(name.size()), FILE_NAME_NORMALIZED);
    if (size >= name.size()) {
        name.resize(size);
        size = GetFinalPathNameByHandleW(file.handle, name.data(), DWORD(name.size()), FILE_NAME_NORMALIZED);
    }
    if (!size || size >= name.size()) return final;
    name.resize(size);
    const std::wstring given = final.filename().native(), found = std::filesystem::path(name).filename().native();
    if (CompareStringOrdinal(given.c_str(), int(given.size()), found.c_str(), int(found.size()), TRUE) != CSTR_EQUAL)
        return final;
    std::filesystem::path spelled = final;
    spelled.replace_filename(found);
    return spelled;
}

} // namespace

FileError::FileError(int error, bool windows, std::filesystem::path path)
    : std::runtime_error(describe(error, windows)), error_(error), windows_(windows), path_(std::move(path)) {}

AtomicTarget::AtomicTarget(std::filesystem::path final) : final_(std::move(final)), temp_(temporary_name(final_)) {}

AtomicTarget::~AtomicTarget() { abandon(); }

bool AtomicTarget::commit() {
    if (state_ != State::open) throw std::logic_error("The temporary file was already committed or abandoned");
    if (same_bytes(temp_, final_)) {
        // A temporary file another program still holds stays abandoned: a later abandon() or the destructor deletes it.
        state_ = DeleteFileW(temp_.c_str()) ? State::committed : State::abandoned;
        return false;
    }
    rename();
    return true;
}

void AtomicTarget::check() const {
    refuse_unwritable(final_);
    // The DELETE access the rename needs, of the file itself (a link, not its target), as the rename replaces it.
    const DWORD error = retrying([&] {
        const HANDLE handle =
            CreateFileW(extended(final_).c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
        if (handle == INVALID_HANDLE_VALUE) return GetLastError();
        CloseHandle(handle);
        return DWORD(0);
    });
    if (error && error != ERROR_FILE_NOT_FOUND) throw os_error(error, final_);
}

void AtomicTarget::abandon() noexcept {
    if (state_ == State::committed) return;
    DeleteFileW(temp_.c_str());
    state_ = State::abandoned;
}

void AtomicTarget::rename() {
    try {
        refuse_unwritable(final_);
    } catch (const FileError &) {
        abandon();
        throw;
    }
    if (const DWORD error = replace(temp_, on_disk(final_))) {
        abandon();
        throw os_error(error, final_);
    }
    state_ = State::committed;
}

bool write_atomic(const std::filesystem::path &final, std::span<const uint8_t> data) {
    if (holds(final, data)) return false;
    refuse_unwritable(final);
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
    refuse_unwritable(final);
    AtomicTarget target(final);
    // A rename on one volume; across volumes MoveFileExW copies the file and deletes the source.
    if (!MoveFileExW(extended(source).c_str(), target.temp().c_str(), MOVEFILE_COPY_ALLOWED))
        throw os_error(GetLastError(), final);
    target.rename();
    return true;
}

} // namespace cn::files
