/* Atomic file writes (GPL-3.0): a writer fills `<final>.chainner-c-<pid>-<seq>.tmp` in the target folder (pid this
 * process's id, seq a process-wide counter; `.cnc-<pid>-<seq>.tmp` in that folder where the first would pass the
 * 255-unit name limit or, with long paths off, MAX_PATH; both names of the rename in the \\?\ form past MAX_PATH, errors
 * naming the path as given) and renames it onto the final name by handle with POSIX semantics (no
 * flush; MoveFileExW MOVEFILE_REPLACE_EXISTING where the volume lacks them), so a failure, Stop or crash never leaves a
 * partial file under the final name. The rename replaces a file other programs hold open sharing delete; one held
 * without delete sharing fails as EACCES after up to 250 ms of retries.
 * A final file that already holds the same bytes (size, then 1 MiB blocks) is left untouched, its modification time
 * included, and the temporary file is deleted. Only this process's own temporary files are ever deleted. Nothing here
 * touches Python, so the run engine's threads call it as it is; the graph module raises FileError as the OSError
 * Python itself raises (atomic_file_python.cpp).
 * On Linux the rename is rename(2), which also replaces a file other programs hold open, the replaced file's permission
 * bits carry over as open(final, "wb") keeps them, and every error is the errno of the failed call. */
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>

namespace cn::files {

/* A failed write, naming the final path (never the temporary one): `error` is the errno Python's open() and write()
 * report for the same condition, raised as OSError(errno, strerror(errno), path): the CRT's errno of _wsopen_s/_write,
 * or the CRT's errno of a failed rename's Windows error (its _dosmaperr table: a folder, a read-only file or one held
 * open is EACCES). Only a Windows error that table lacks stays one (`windows`: OSError with winerror and
 * FormatMessage's text). */
class FileError : public std::runtime_error {
public:
    FileError(int error, bool windows, std::filesystem::path path);
    int error() const noexcept { return error_; }
    bool windows() const noexcept { return windows_; }
    const std::filesystem::path &path() const noexcept { return path_; }

private:
    int error_;
    bool windows_;
    std::filesystem::path path_;
};

/* Writes `data` to `final`, whose folder must exist: false when `final` already holds exactly these bytes (compared
 * before any temporary file is written; the file is left untouched), true once written and renamed into place.
 * Throws FileError; no temporary file is left after a failure. */
bool write_atomic(const std::filesystem::path &final, std::span<const uint8_t> data);

/* Moves a finished file (`source`, which may lie on another drive, such as chaiNNer's temporary folder) to `final`
 * through a temporary file in final's folder: false when `final` already holds the same bytes (left untouched, and
 * `source` stays for its owner to delete), true once moved. Throws FileError; no temporary file is left after a
 * failure. */
bool move_atomic(const std::filesystem::path &source, const std::filesystem::path &final);

/* The temporary file of a writer that streams, or hands a path to a library: temp() names a file in final's folder
 * that no other file has; the writer creates and fills it, then commit() renames it onto final (false: final already
 * held the same bytes and is left untouched, the temporary file deleted). A failed commit deletes the temporary file
 * and throws FileError. abandon() deletes the temporary file unless committed, ignoring errors; it may run again (a
 * file another process still held open goes on a later call). The destructor abandons an uncommitted target. */
class AtomicTarget {
public:
    explicit AtomicTarget(std::filesystem::path final);
    AtomicTarget(const AtomicTarget &) = delete;
    AtomicTarget &operator=(const AtomicTarget &) = delete;
    ~AtomicTarget();

    const std::filesystem::path &final_path() const noexcept { return final_; }
    const std::filesystem::path &temp() const noexcept { return temp_; }
    /* Whether commit() or abandon() ran: a context manager commits an open target only. */
    bool open() const noexcept { return state_ == State::open; }
    /* Throws the FileError commit() would throw when final is a folder or a file the rename cannot replace (held open
     * without delete sharing), before a long write starts; nothing when final does not exist. */
    void check() const;
    bool commit();
    void abandon() noexcept;

private:
    enum class State { open, committed, abandoned };
    friend bool write_atomic(const std::filesystem::path &, std::span<const uint8_t>);
    friend bool move_atomic(const std::filesystem::path &, const std::filesystem::path &);
    void rename();

    std::filesystem::path final_, temp_;
    State state_ = State::open;
};

} // namespace cn::files
