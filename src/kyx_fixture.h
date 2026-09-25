#ifndef KYX_FIXTURE_H
#define KYX_FIXTURE_H

/*
 * The decoder fixture: a few cells of head output and the boxes they
 * must become.  Shared by the tests, the command's `decode` and `bench`
 * verbs and its self-test, so that all four read the same file the same
 * way.  Not part of the public header: it allocates, and it is a test
 * format, not an interface.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "kilix_yolox.h"

typedef struct kyx_fixture {
    bool raw;                 /* nine raw tensors, else the export's (N, 85) */
    int input_width;
    int input_height;
    float conf;
    float iou;
    size_t cell_count;        /* cells present in the file */
    size_t total_cells;       /* cells the input size implies */
    float *flat;              /* (total_cells, 85), absent cells filled */
    float *planes[KYX_LEVELS];/* raw form: reg, obj, cls planar per level */
    kyx_raw_level raw_levels[KYX_LEVELS];
    size_t box_count;
    kyx_box *boxes;           /* expected, score order, input pixels */
} kyx_fixture;

/* Load from an open stream.  On failure `error` holds a short reason. */
bool kyx_fixture_read(kyx_fixture *fixture, FILE *in, char *error, size_t size);
bool kyx_fixture_load(kyx_fixture *fixture, const char *path, char *error, size_t size);
void kyx_fixture_free(kyx_fixture *fixture);

/* Decode and suppress the way the fixture says, into `out`. */
size_t kyx_fixture_decode(
    const kyx_fixture *fixture, bool per_class, kyx_box *out, size_t capacity,
    size_t *candidates, size_t *dropped);

/* Compare a decode against the expected boxes.  Returns the index of the
 * first mismatch, or the box count when all match. */
size_t kyx_fixture_compare(
    const kyx_fixture *fixture, const kyx_box *got, size_t count,
    float score_tolerance, float pixel_tolerance);

#endif /* KYX_FIXTURE_H */
