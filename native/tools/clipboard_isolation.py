"""Private-window-station clipboard verification for fresh hidden test processes.

Never call clipboard APIs until PrivateStation.verify succeeds. This module does
not read, preserve, restore, or write the interactive clipboard. Creation-only
objects have a DACL limited to the calling identity. No fallback to WinSta0.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import uuid
from ctypes import wintypes as w
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class SecurityAttributes(ctypes.Structure):
    _fields_ = [("length", w.DWORD), ("descriptor", w.LPVOID), ("inherit", w.BOOL)]


class SidAndAttributes(ctypes.Structure):
    _fields_ = [("sid", w.LPVOID), ("attributes", w.DWORD)]


class PrivateStation:
    """Create and attach only this fresh helper to its own clipboard station."""

    def __init__(self):
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        declarations = (
            (self.kernel, "GetCurrentProcess", [], w.HANDLE),
            (self.kernel, "GetCurrentThreadId", [], w.DWORD),
            (self.kernel, "CloseHandle", [w.HANDLE], w.BOOL),
            (self.kernel, "LocalFree", [w.LPVOID], w.LPVOID),
            (self.kernel, "GlobalLock", [w.HANDLE], w.LPVOID),
            (self.kernel, "GlobalUnlock", [w.HANDLE], w.BOOL),
            (self.kernel, "GlobalSize", [w.HANDLE], ctypes.c_size_t),
            (
                self.advapi,
                "OpenProcessToken",
                [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)],
                w.BOOL,
            ),
            (
                self.advapi,
                "GetTokenInformation",
                [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD)],
                w.BOOL,
            ),
            (
                self.advapi,
                "ConvertSidToStringSidW",
                [w.LPVOID, ctypes.POINTER(w.LPWSTR)],
                w.BOOL,
            ),
            (
                self.advapi,
                "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                [w.LPCWSTR, w.DWORD, ctypes.POINTER(w.LPVOID), w.LPVOID],
                w.BOOL,
            ),
            (self.user, "GetProcessWindowStation", [], w.HANDLE),
            (self.user, "GetThreadDesktop", [w.DWORD], w.HANDLE),
            (
                self.user,
                "GetUserObjectInformationW",
                [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD)],
                w.BOOL,
            ),
            (
                self.user,
                "CreateWindowStationW",
                [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.POINTER(SecurityAttributes)],
                w.HANDLE,
            ),
            (self.user, "SetProcessWindowStation", [w.HANDLE], w.BOOL),
            (self.user, "CloseWindowStation", [w.HANDLE], w.BOOL),
            (
                self.user,
                "CreateDesktopW",
                [
                    w.LPCWSTR,
                    w.LPCWSTR,
                    w.LPVOID,
                    w.DWORD,
                    w.DWORD,
                    ctypes.POINTER(SecurityAttributes),
                ],
                w.HANDLE,
            ),
            (self.user, "SetThreadDesktop", [w.HANDLE], w.BOOL),
            (self.user, "CloseDesktop", [w.HANDLE], w.BOOL),
            (self.user, "OpenClipboard", [w.HWND], w.BOOL),
            (self.user, "CloseClipboard", [], w.BOOL),
            (self.user, "GetClipboardData", [w.UINT], w.HANDLE),
            (self.user, "EnumClipboardFormats", [w.UINT], w.UINT),
        )
        for library, name, arguments, result in declarations:
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, result
        self.station = self.desktop = None
        self.name = "CodexChaiNNerClipboard-" + uuid.uuid4().hex
        self.old_station = self.user.GetProcessWindowStation()
        self.old_desktop = self.user.GetThreadDesktop(self.kernel.GetCurrentThreadId())
        self.original_name = self.object_name(self.old_station)
        self.attached = False

    @staticmethod
    def check(value: T) -> T:
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def object_name(self, handle: int) -> str:
        needed = w.DWORD()
        self.user.GetUserObjectInformationW(handle, 2, None, 0, ctypes.byref(needed))
        if not needed.value:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(needed.value // ctypes.sizeof(w.WCHAR))
        self.check(
            self.user.GetUserObjectInformationW(
                handle, 2, buffer, needed, ctypes.byref(needed)
            )
        )
        return buffer.value

    def security(self):
        token = w.HANDLE()
        self.check(
            self.advapi.OpenProcessToken(
                self.kernel.GetCurrentProcess(), 8, ctypes.byref(token)
            )
        )
        try:
            needed = w.DWORD()
            self.advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
            if not needed.value:
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_string_buffer(needed.value)
            self.check(
                self.advapi.GetTokenInformation(
                    token, 1, buffer, needed, ctypes.byref(needed)
                )
            )
            token_user = ctypes.cast(buffer, ctypes.POINTER(SidAndAttributes)).contents
            sid = w.LPWSTR()
            self.check(
                self.advapi.ConvertSidToStringSidW(token_user.sid, ctypes.byref(sid))
            )
            try:
                if sid.value is None:
                    raise RuntimeError("Windows returned an empty SID")
                descriptor = w.LPVOID()
                self.check(
                    self.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                        "D:P(A;;GA;;;" + sid.value + ")",
                        1,
                        ctypes.byref(descriptor),
                        None,
                    )
                )
                return descriptor
            finally:
                self.kernel.LocalFree(sid)
        finally:
            self.kernel.CloseHandle(token)

    def __enter__(self):
        descriptor = self.security()
        security = SecurityAttributes(
            ctypes.sizeof(SecurityAttributes), descriptor, False
        )
        try:
            # CWF_CREATE_ONLY=1; explicit current-identity DACL, non-inherited handle.
            self.station = self.user.CreateWindowStationW(
                self.name, 1, 0x000F037F, ctypes.byref(security)
            )
            if not self.station and ctypes.get_last_error() == 5:
                # Named creation requires Administrators membership. The public
                # NULL-name API permits a logon-derived noninteractive station;
                # CREATE_ONLY still forbids attaching to any existing object.
                self.station = self.user.CreateWindowStationW(
                    None, 1, 0x000F037F, ctypes.byref(security)
                )
                self.check(self.station)
                self.name = self.object_name(self.station)
            self.check(self.station)
            if self.name.lower() in {"winsta0", self.original_name.lower()}:
                raise RuntimeError(
                    "Creation did not produce a separate private station"
                )
            self.check(self.user.SetProcessWindowStation(self.station))
            self.desktop = self.check(
                self.user.CreateDesktopW(
                    "ClipboardTest", None, None, 0, 0x000F01FF, ctypes.byref(security)
                )
            )
            self.check(self.user.SetThreadDesktop(self.desktop))
            self.attached = True
            self.verify()
            return self
        except BaseException:
            self.close()
            raise
        finally:
            self.kernel.LocalFree(descriptor)

    def verify(self) -> None:
        current = self.object_name(self.user.GetProcessWindowStation())
        desktop = self.object_name(
            self.user.GetThreadDesktop(self.kernel.GetCurrentThreadId())
        )
        if (
            not self.attached
            or current != self.name
            or current.lower() in {"winsta0", self.original_name.lower()}
            or desktop != "ClipboardTest"
        ):
            raise RuntimeError(
                "Clipboard isolation identity check failed; no clipboard operation permitted"
            )

    def read(self, format_id: int) -> tuple[bytes, list[int]]:
        self.verify()
        self.check(self.user.OpenClipboard(None))
        try:
            formats, current = [], 0
            while True:
                current = self.user.EnumClipboardFormats(current)
                if not current:
                    break
                formats.append(current)
            handle = self.check(self.user.GetClipboardData(format_id))
            size = self.kernel.GlobalSize(handle)
            pointer = self.check(self.kernel.GlobalLock(handle))
            try:
                return ctypes.string_at(pointer, size), formats
            finally:
                self.kernel.GlobalUnlock(handle)
        finally:
            self.check(self.user.CloseClipboard())

    def close(self) -> None:
        # This helper creates no windows/hooks, so its original desktop can be
        # restored before closing the two owned objects. No clipboard call here.
        if self.station:
            self.check(self.user.SetProcessWindowStation(self.old_station))
        if self.desktop:
            self.check(self.user.SetThreadDesktop(self.old_desktop))
            self.check(self.user.CloseDesktop(self.desktop))
            self.desktop = None
        if self.station:
            self.check(self.user.CloseWindowStation(self.station))
            self.station = None
        self.attached = False

    def __exit__(self, *_: object):
        self.close()


def probe(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "success": False,
        "interactive_clipboard_accessed": False,
        "cases": [],
    }
    try:
        with PrivateStation() as station:
            report["station"] = station.name
            report["private_station_verified"] = True
            # Import extension/array libraries only after station attachment.
            import numpy as np

            from chainner_ext import Clipboard

            cases: list[tuple[str, str | np.ndarray, int]] = [
                ("text", "Synthetic é🙂\x00tail", 13)
            ]
            for channels in (1, 3, 4):
                data = (
                    np.arange(2 * 3 * channels, dtype=np.float32).reshape(
                        2, 3, channels
                    )
                    % 17
                ) / np.float32(16)
                np.save(directory / f"image-{channels}.npy", data)
                cases.append((f"image-{channels}", data, 17))
            for name, value, format_id in cases:
                station.verify()
                clipboard = Clipboard.create_instance()
                try:
                    if format_id == 13:
                        assert isinstance(value, str)
                        clipboard.write_text(value)
                    else:
                        assert isinstance(value, np.ndarray)
                        clipboard.write_image(value, "BGR")
                    data, formats = station.read(format_id)
                    (directory / (name + ".bin")).write_bytes(data)
                    report["cases"].append(
                        {
                            "name": name,
                            "success": True,
                            "bytes": len(data),
                            "formats": formats,
                        }
                    )
                except Exception as error:
                    report["cases"].append(
                        {
                            "name": name,
                            "success": False,
                            "type": type(error).__name__,
                            "error": str(error),
                        }
                    )
            report["success"] = all(case["success"] for case in report["cases"])
        report["owned_station_closed"] = True
    except Exception as error:
        report["error"] = str(error)
    finally:
        (directory / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    # This entrypoint is launched as a separate hidden process by the test host.
    if sys.platform != "win32":
        raise SystemExit("Windows test helper required")
    probe(args.probe.resolve())
