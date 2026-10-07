"""Capture clipboard calls of one loaded test module in a fresh subprocess.

Only its PE import table is temporarily redirected. No clipboard API is called:
four mandatory USER32 slots must all be found before any writer can run. Real
global allocations stay process-owned until the fake clipboard frees them.
This verifies binary payloads/call contracts, not real Windows publication.
"""

from __future__ import annotations

import ctypes
import struct
from ctypes import wintypes as w
from pathlib import Path


class ClipboardCapture:
    def __init__(self, module_path: Path, failure: str | None = None):
        self.module = ctypes.WinDLL(str(module_path), use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        declarations = (
            (
                "VirtualProtect",
                [w.LPVOID, ctypes.c_size_t, w.DWORD, ctypes.POINTER(w.DWORD)],
                w.BOOL,
            ),
            ("GlobalAlloc", [w.UINT, ctypes.c_size_t], w.HANDLE),
            ("GlobalLock", [w.HANDLE], w.LPVOID),
            ("GlobalUnlock", [w.HANDLE], w.BOOL),
            ("GlobalFree", [w.HANDLE], w.HANDLE),
            ("GlobalSize", [w.HANDLE], ctypes.c_size_t),
        )
        for name, arguments, result in declarations:
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        self.failure = failure
        self.calls: list[tuple] = []
        self.data: dict[int, bytes] = {}
        self.allocations: set[int] = set()
        self.transferred: set[int] = set()
        self.callbacks: dict = {}
        self.patches: list[tuple[int, int]] = []
        self.callback_errors: list[str] = []
        self.sleep_count = 0

    def clipboard_open(self, owner):
        self.calls.append(("OpenClipboard", owner))
        if self.failure == "open":
            ctypes.set_last_error(5)
            return False
        return True

    def clipboard_close(self):
        self.calls.append(("CloseClipboard",))
        return True

    def clipboard_empty(self):
        self.calls.append(("EmptyClipboard",))
        if self.failure == "empty":
            ctypes.set_last_error(5)
            return False
        self.data.clear()
        for handle in self.transferred:
            self.kernel.GlobalFree(handle)
            self.allocations.discard(handle)
        self.transferred.clear()
        return True

    def clipboard_set(self, format_id, handle):
        self.calls.append(("SetClipboardData", format_id))
        if self.failure == "set":
            ctypes.set_last_error(5)
            return None
        pointer = self.kernel.GlobalLock(handle)
        if not pointer:
            self.callback_errors.append("Cannot lock published data")
            return None
        try:
            self.data[format_id] = ctypes.string_at(
                pointer, self.kernel.GlobalSize(handle)
            )
        finally:
            self.kernel.GlobalUnlock(handle)
        self.transferred.add(handle)
        return handle

    def global_alloc(self, flags, size):
        self.calls.append(("GlobalAlloc", flags, size))
        if self.failure == "allocate":
            ctypes.set_last_error(8)
            return None
        handle = self.kernel.GlobalAlloc(flags, size)
        if handle:
            self.allocations.add(handle)
        return handle

    def global_lock(self, handle):
        self.calls.append(("GlobalLock",))
        if self.failure == "lock":
            ctypes.set_last_error(6)
            return None
        return self.kernel.GlobalLock(handle)

    def global_unlock(self, handle):
        self.calls.append(("GlobalUnlock",))
        return self.kernel.GlobalUnlock(handle)

    def global_free(self, handle):
        self.calls.append(("GlobalFree",))
        self.allocations.discard(handle)
        return self.kernel.GlobalFree(handle)

    def delete_object(self, _handle):
        # Pinned arboard erroneously calls this for a failed HGLOBAL transfer.
        # Model its failure and retain the allocation for test-process cleanup.
        self.calls.append(("DeleteObject",))
        return False

    def sleep(self, milliseconds):
        self.calls.append(("Sleep", milliseconds))
        self.sleep_count += 1

    def import_slots(self) -> dict[str, int]:
        base = self.module._handle  # ctypes exposes its loaded module handle here.
        header = ctypes.string_at(base, 4096)
        pe = struct.unpack_from("<I", header, 0x3C)[0]
        if (
            header[pe : pe + 4] != b"PE\0\0"
            or struct.unpack_from("<H", header, pe + 24)[0] != 0x20B
        ):
            raise RuntimeError("Expected the pinned 64-bit PE module")
        image_size = struct.unpack_from("<I", header, pe + 24 + 56)[0]
        import_rva = struct.unpack_from("<I", header, pe + 24 + 120)[0]

        def read(offset, size):
            if offset < 0 or offset + size > image_size:
                raise RuntimeError("PE import offset outside owned module")
            return ctypes.string_at(base + offset, size)

        def name(offset):
            return (
                read(offset, min(256, image_size - offset))
                .split(b"\0", 1)[0]
                .decode("ascii")
            )

        slots = {}
        while True:
            lookup, _, _, library, first = struct.unpack("<5I", read(import_rva, 20))
            if not any((lookup, library, first)):
                break
            if not lookup:
                raise RuntimeError("PE import lookup table unavailable")
            index = 0
            while True:
                value = struct.unpack("<Q", read(lookup + index * 8, 8))[0]
                if not value:
                    break
                if not value & (1 << 63):
                    symbol = name(value + 2)
                    if symbol in slots:
                        raise RuntimeError("Duplicate PE import: " + symbol)
                    slots[symbol] = base + first + index * 8
                index += 1
            import_rva += 20
        return slots

    def write_pointer(self, address: int, value: int) -> None:
        protection = w.DWORD()
        if not self.kernel.VirtualProtect(address, 8, 4, ctypes.byref(protection)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            ctypes.c_void_p.from_address(address).value = value
        finally:
            ignored = w.DWORD()
            if not self.kernel.VirtualProtect(
                address, 8, protection, ctypes.byref(ignored)
            ):
                raise ctypes.WinError(ctypes.get_last_error())

    def __enter__(self):
        signatures = {
            "OpenClipboard": (w.BOOL, [w.HWND], self.clipboard_open),
            "CloseClipboard": (w.BOOL, [], self.clipboard_close),
            "EmptyClipboard": (w.BOOL, [], self.clipboard_empty),
            "SetClipboardData": (w.HANDLE, [w.UINT, w.HANDLE], self.clipboard_set),
            "GlobalAlloc": (w.HANDLE, [w.UINT, ctypes.c_size_t], self.global_alloc),
            "GlobalLock": (w.LPVOID, [w.HANDLE], self.global_lock),
            "GlobalUnlock": (w.BOOL, [w.HANDLE], self.global_unlock),
            "GlobalFree": (w.HANDLE, [w.HANDLE], self.global_free),
            "DeleteObject": (w.BOOL, [w.HANDLE], self.delete_object),
            "Sleep": (None, [w.DWORD], self.sleep),
        }
        slots = self.import_slots()
        required = {
            "OpenClipboard",
            "CloseClipboard",
            "EmptyClipboard",
            "SetClipboardData",
            "GlobalAlloc",
            "GlobalLock",
            "GlobalUnlock",
        }
        if not required <= slots.keys():
            raise RuntimeError(
                "Uncaptured required imports; writer forbidden: "
                + repr(required - slots.keys())
            )
        try:
            for symbol, (result, arguments, function) in signatures.items():
                if symbol not in slots:
                    continue
                callback = ctypes.WINFUNCTYPE(result, *arguments, use_last_error=True)(
                    function
                )
                self.callbacks[symbol] = callback
                address = slots[symbol]
                old = ctypes.c_void_p.from_address(address).value
                if old is None:
                    raise RuntimeError("Null import pointer")
                self.patches.append((address, old))
                pointer = ctypes.cast(callback, w.LPVOID).value
                if pointer is None:
                    raise RuntimeError("Null callback pointer")
                self.write_pointer(address, pointer)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        for address, original in reversed(self.patches):
            self.write_pointer(address, original)
        self.patches.clear()
        for handle in self.allocations:
            self.kernel.GlobalFree(handle)
        self.allocations.clear()
        self.transferred.clear()
        self.callbacks.clear()

    def __exit__(self, *_):
        self.close()
