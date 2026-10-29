/* The clipboard on Linux (GPL-3.0), compiled into each including unit (chainner_ext's clipboard.c and the node path,
 * utility_clipboard.cpp): text as UTF-8 and images as PNG, the two types arboard's Linux backends offer, handed to the
 * program the desktop session provides for it. Under Wayland (WAYLAND_DISPLAY set) that is wl-copy (the wl-clipboard
 * package); under X11 (DISPLAY set; also the fallback under Wayland, through Xwayland) xclip, or xsel for text. The
 * program reads the data, leaves a copy of itself serving it to other applications, and exits. The including units
 * include Python.h first, whose pyconfig.h defines _GNU_SOURCE (pipe2, memfd_create, environ). */
#ifndef CHAINNER_CLIPBOARD_POSIX_H
#define CHAINNER_CLIPBOARD_POSIX_H
#ifndef _WIN32
#include "run_kernels.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CN_CLIPBOARD_PREFIX "Unknown error while interacting with the clipboard: "
/* How long the program may take to read the data and exit before it is killed. */
#define CN_CLIPBOARD_SECONDS 10

typedef struct cn_clipboard_buffer {
    uint8_t *data;
    size_t size;
} cn_clipboard_buffer;

static uint8_t *cn_clipboard_allocate(void *context, size_t size)
{
    cn_clipboard_buffer *buffer = (cn_clipboard_buffer *)context;
    buffer->data = (uint8_t *)malloc(size ? size : 1);
    buffer->size = buffer->data ? size : 0;
    return buffer->data;
}

/* The PNG of a clipboard image, from cn_clipboard_pixels_f32's bottom-up BGRA rows of width x height: written top-down
 * as RGBA by chaiNNer-C's default encoder. CN_OK with png->data (malloc'd; the caller frees it); CN_INVALID_ARGUMENT
 * for an empty image or one beyond the encoder's range, CN_ALLOCATION_FAILED when memory runs out. */
static cn_status cn_clipboard_png(const uint8_t *bgra, size_t height, size_t width, cn_clipboard_buffer *png)
{
    png->data = NULL;
    png->size = 0;
    if (!height || !width) return CN_INVALID_ARGUMENT;
    const ptrdiff_t row = (ptrdiff_t)(width * 4);
    const cn_status status = cn_png_encode_u8(bgra + (height - 1) * width * 4, height, width, 4, -row, 4, 1, 1,
                                              cn_clipboard_allocate, png);
    if (status != CN_OK) {
        free(png->data);
        png->data = NULL;
        png->size = 0;
    }
    return status;
}

/* Starts argv[0] (searched on PATH) with `input` as its standard input, /dev/null as its output and `errors` as its
 * error stream; no other descriptor (the rest are close-on-exec), an empty signal mask, and SIGPIPE and SIGXFSZ at
 * their defaults (Python ignores them, and subprocess restores them for a child). 0 or the error number. */
static int cn_clipboard_spawn(const char *const *argv, int input, int errors, pid_t *child)
{
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    int status = posix_spawn_file_actions_init(&actions);
    if (status) return status;
    status = posix_spawnattr_init(&attributes);
    if (!status) {
        sigset_t mask, defaults;
        sigemptyset(&mask);
        sigemptyset(&defaults);
        sigaddset(&defaults, SIGPIPE);
        sigaddset(&defaults, SIGXFSZ);
        if (!(status = posix_spawn_file_actions_adddup2(&actions, input, 0)) &&
            !(status = posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0)) &&
            !(status = posix_spawn_file_actions_adddup2(&actions, errors, 2)) &&
            !(status = posix_spawnattr_setsigmask(&attributes, &mask)) &&
            !(status = posix_spawnattr_setsigdefault(&attributes, &defaults)) &&
            !(status = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF)))
            status = posix_spawnp(child, argv[0], &actions, &attributes, (char *const *)argv, environ);
        posix_spawnattr_destroy(&attributes);
    }
    posix_spawn_file_actions_destroy(&actions);
    return status;
}

/* Milliseconds left until `deadline` (CLOCK_MONOTONIC), at least 0. */
static int cn_clipboard_remaining(const struct timespec *deadline)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    const long long left = (long long)(deadline->tv_sec - now.tv_sec) * 1000 + (deadline->tv_nsec - now.tv_nsec) / 1000000;
    return left > 0 ? (int)left : 0;
}

/* The tail of what the program wrote to its error stream, on one line, into `text`. */
static void cn_clipboard_message(int errors, char *text, size_t capacity)
{
    text[0] = '\0';
    const off_t end = lseek(errors, 0, SEEK_END);
    if (end <= 0) return;
    const off_t start = end > (off_t)(capacity - 1) ? end - (off_t)(capacity - 1) : 0;
    const ssize_t got = pread(errors, text, (size_t)(end - start), start);
    size_t length = got > 0 ? (size_t)got : 0;
    while (length && (text[length - 1] == '\n' || text[length - 1] == '\r' || text[length - 1] == ' ')) --length;
    text[length] = '\0';
    for (size_t i = 0; i < length; ++i)
        if (text[i] == '\n' || text[i] == '\r') text[i] = ' ';
}

/* Runs the program `argv` with `bytes` on its standard input, the write and the exit bounded by CN_CLIPBOARD_SECONDS.
 * 0 when it took them all and exited with status 0; ENOENT when it is not installed (nothing else happened); else -1
 * with the failure's message in `scratch`. */
static int cn_clipboard_run(const char *const *argv, const unsigned char *bytes, size_t size, char *scratch,
                            size_t capacity)
{
    int input[2];
    if (pipe2(input, O_CLOEXEC) != 0) {
        snprintf(scratch, capacity, CN_CLIPBOARD_PREFIX "could not start %s: %s", argv[0], strerror(errno));
        return -1;
    }
    const int errors = memfd_create("chainner-clipboard", MFD_CLOEXEC);
    pid_t child = 0;
    const int started = errors < 0 ? errno : cn_clipboard_spawn(argv, input[0], errors, &child);
    close(input[0]);
    if (started) {
        close(input[1]);
        if (errors >= 0) close(errors);
        if (started == ENOENT) return ENOENT;
        snprintf(scratch, capacity, CN_CLIPBOARD_PREFIX "could not start %s: %s", argv[0], strerror(started));
        return -1;
    }
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += CN_CLIPBOARD_SECONDS;
    /* The data, then the end of the input. A program that stopped reading (EPIPE) has exited: its status says why. */
    int failure = fcntl(input[1], F_SETFL, O_NONBLOCK) != 0 ? errno : 0, timed_out = 0;
    size_t done = 0;
    while (!failure && !timed_out && done < size) {
        struct pollfd ready = {input[1], POLLOUT, 0};
        const int polled = poll(&ready, 1, cn_clipboard_remaining(&deadline));
        const ssize_t written = polled > 0 ? write(input[1], bytes + done, size - done) : polled;
        if (polled == 0) timed_out = 1;
        else if (written > 0) done += (size_t)written;
        else if (errno == EPIPE) break;
        else if (errno != EINTR && errno != EAGAIN) failure = errno;
    }
    close(input[1]);
    int exit_status = 0;
    pid_t reaped = 0;
    while (!timed_out && !reaped) {
        reaped = waitpid(child, &exit_status, WNOHANG);
        if (reaped < 0 && errno == EINTR) reaped = 0;
        if (reaped) break;
        if (!cn_clipboard_remaining(&deadline)) timed_out = 1;
        else {
            const struct timespec pause = {0, 2000000};
            nanosleep(&pause, NULL);
        }
    }
    int result = 0;
    if (timed_out) {
        kill(child, SIGKILL);
        while (waitpid(child, &exit_status, 0) < 0 && errno == EINTR) {}
        snprintf(scratch, capacity, CN_CLIPBOARD_PREFIX "%s did not finish within %d seconds", argv[0],
                 CN_CLIPBOARD_SECONDS);
        result = -1;
    } else if (reaped < 0) {
        /* Not this process's child any more (another waiter reaped it): nothing to stop. */
        snprintf(scratch, capacity, CN_CLIPBOARD_PREFIX "could not wait for %s: %s", argv[0], strerror(errno));
        result = -1;
    } else if (failure) {
        snprintf(scratch, capacity, CN_CLIPBOARD_PREFIX "could not write to %s: %s", argv[0], strerror(failure));
        result = -1;
    } else if (!WIFEXITED(exit_status) || WEXITSTATUS(exit_status) != 0 || done < size) {
        char message[256], outcome[64];
        cn_clipboard_message(errors, message, sizeof(message));
        if (WIFSIGNALED(exit_status)) snprintf(outcome, sizeof(outcome), "was ended by signal %d", WTERMSIG(exit_status));
        else if (WEXITSTATUS(exit_status) != 0)
            snprintf(outcome, sizeof(outcome), "exited with code %d", WEXITSTATUS(exit_status));
        else snprintf(outcome, sizeof(outcome), "exited before reading all the data");
        snprintf(scratch, capacity, CN_CLIPBOARD_PREFIX "%s %s%s%s", argv[0], outcome, message[0] ? ": " : "", message);
        result = -1;
    }
    close(errors);
    return result;
}

/* Places `size` bytes on the clipboard: UTF-8 text, or (png) a PNG image. NULL, or the failure's message (static or in
 * `scratch`). Runs without the GIL. */
static const char *cn_clipboard_publish(const unsigned char *bytes, size_t size, int png, char *scratch,
                                        size_t capacity)
{
    static const char *const wl_text[] = {"wl-copy", "--type", "text/plain;charset=utf-8", NULL};
    static const char *const wl_image[] = {"wl-copy", "--type", "image/png", NULL};
    static const char *const xclip_text[] = {"xclip", "-selection", "clipboard", "-target", "UTF8_STRING", NULL};
    static const char *const xclip_image[] = {"xclip", "-selection", "clipboard", "-target", "image/png", NULL};
    static const char *const xsel_text[] = {"xsel", "--clipboard", "--input", NULL};
    const char *wayland = getenv("WAYLAND_DISPLAY"), *x11 = getenv("DISPLAY");
    const int on_wayland = wayland && wayland[0], on_x11 = x11 && x11[0];
    if (!on_wayland && !on_x11)
        return "The clipboard needs a desktop session: neither WAYLAND_DISPLAY nor DISPLAY is set.";
    const char *const *programs[3];
    size_t count = 0;
    if (on_wayland) programs[count++] = png ? wl_image : wl_text;
    if (on_x11) {
        programs[count++] = png ? xclip_image : xclip_text;
        if (!png) programs[count++] = xsel_text;
    }
    for (size_t i = 0; i < count; ++i) {
        const int status = cn_clipboard_run(programs[i], bytes, size, scratch, capacity);
        if (status != ENOENT) return status ? scratch : NULL;
    }
    return png ? "No program to copy an image to the clipboard was found: install wl-clipboard (Wayland) or xclip (X11)."
               : "No program to copy text to the clipboard was found: install wl-clipboard (Wayland) or xclip (X11).";
}
#endif
#endif
