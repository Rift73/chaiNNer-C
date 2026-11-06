#pragma once
inline void* GetModuleHandleW(wchar_t const*) { return reinterpret_cast<void*>(1); }
inline void* LoadLibraryExW(wchar_t const*, void*, unsigned) { return nullptr; }
inline void* LoadLibraryW(wchar_t const*) { return nullptr; }
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0
