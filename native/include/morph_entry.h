/* A morphology deque entry (D11): the src value beside its index, so the compares read
 * no src. Defined once for the two deque units, morphology_ops.c and filter_ops.c. */
#ifndef CHAINNER_MORPH_ENTRY_H
#define CHAINNER_MORPH_ENTRY_H
#include <stdint.h>

struct morph_entry {
    float value;
    uint32_t index;
};
#endif
