#pragma once
#ifdef _WIN32
#define DUAL_EXPORT __declspec(dllexport)
#else
#define DUAL_EXPORT __attribute__((visibility("default")))
#endif
