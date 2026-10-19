/* chainner_ext's regex engine for the native entry points (native/src/run/nodes): utility_text.cpp fetches the
 * chn_regex_api capsule (chainner_regex.h) once, under the GIL, through the same routine as its own regex_engine(),
 * and keeps the pointer. */
#pragma once

#include "chainner_regex.h"

/* The engine's API, or nullptr when chainner_ext cannot provide it (the node's Python body then raises its own
 * error). The first call takes the GIL; later calls only read the kept pointer. */
const chn_regex_api *cn_regex_api();
