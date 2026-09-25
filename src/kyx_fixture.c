#include "kyx_fixture.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* sigmoid(-30) is 9e-14: a cell the file does not mention never fires. */
#define RAW_ABSENT_OBJ (-30.0f)

static bool fail(char *error, size_t size, const char *message)
{
    if (error != NULL && size > 0u) {
        strncpy(error, message, size - 1u);
        error[size - 1u] = '\0';
    }
    return false;
}

static bool read_line(FILE *in, char **buffer, size_t *size)
{
    size_t used = 0u;

    for (;;) {
        int c;

        if (used + 2u > *size) {
            size_t grown = *size == 0u ? 4096u : *size * 2u;
            char *next = realloc(*buffer, grown);
            if (next == NULL) {
                return false;
            }
            *buffer = next;
            *size = grown;
        }
        c = fgetc(in);
        if (c == EOF) {
            (*buffer)[used] = '\0';
            return used > 0u;
        }
        if (c == '\n') {
            (*buffer)[used] = '\0';
            return true;
        }
        (*buffer)[used++] = (char)c;
    }
}

bool kyx_fixture_read(kyx_fixture *fixture, FILE *in, char *error, size_t size)
{
    char *line = NULL;
    size_t line_size = 0u;
    kyx_level levels[KYX_LEVELS];
    size_t offsets[KYX_LEVELS];
    char form[16];
    int version;

    memset(fixture, 0, sizeof(*fixture));
    if (!read_line(in, &line, &line_size) ||
        sscanf(line, "kilix-yolox-fixture %d", &version) != 1 || version != 1) {
        free(line);
        return fail(error, size, "not a kilix-yolox fixture");
    }
    if (!read_line(in, &line, &line_size) ||
        sscanf(line, "form %15s", form) != 1) {
        free(line);
        return fail(error, size, "missing form");
    }
    if (strcmp(form, "raw") == 0) {
        fixture->raw = true;
    } else if (strcmp(form, "flat") == 0) {
        fixture->raw = false;
    } else {
        free(line);
        return fail(error, size, "form is neither raw nor flat");
    }
    if (!read_line(in, &line, &line_size) ||
        sscanf(line, "input %d %d", &fixture->input_width,
               &fixture->input_height) != 2 ||
        !kyx_levels(fixture->input_width, fixture->input_height, levels)) {
        free(line);
        return fail(error, size, "bad input size");
    }
    if (!read_line(in, &line, &line_size) ||
        sscanf(line, "conf %f", &fixture->conf) != 1 ||
        !read_line(in, &line, &line_size) ||
        sscanf(line, "iou %f", &fixture->iou) != 1) {
        free(line);
        return fail(error, size, "missing conf or iou");
    }
    if (!read_line(in, &line, &line_size) ||
        sscanf(line, "cells %zu", &fixture->cell_count) != 1) {
        free(line);
        return fail(error, size, "missing cell count");
    }

    fixture->total_cells = kyx_cells(fixture->input_width, fixture->input_height);
    fixture->flat = calloc(fixture->total_cells * KYX_CELL_VALUES, sizeof(float));
    if (fixture->flat == NULL) {
        free(line);
        return fail(error, size, "out of memory");
    }
    {
        size_t offset = 0u;
        for (int level = 0; level < KYX_LEVELS; level++) {
            const size_t area =
                (size_t)levels[level].width * (size_t)levels[level].height;
            offsets[level] = offset;
            offset += area;
            if (fixture->raw) {
                fixture->planes[level] = calloc(area * KYX_CELL_VALUES, sizeof(float));
                if (fixture->planes[level] == NULL) {
                    free(line);
                    kyx_fixture_free(fixture);
                    return fail(error, size, "out of memory");
                }
                fixture->raw_levels[level].reg = fixture->planes[level];
                fixture->raw_levels[level].obj = fixture->planes[level] + 4u * area;
                fixture->raw_levels[level].cls = fixture->planes[level] + 5u * area;
                for (size_t cell = 0u; cell < area; cell++) {
                    fixture->planes[level][4u * area + cell] = RAW_ABSENT_OBJ;
                }
            }
        }
        if (fixture->raw) {
            for (size_t cell = 0u; cell < fixture->total_cells; cell++) {
                fixture->flat[cell * KYX_CELL_VALUES + 4] = RAW_ABSENT_OBJ;
            }
        }
    }

    for (size_t i = 0u; i < fixture->cell_count; i++) {
        int level;
        int gy;
        int gx;
        char *cursor;
        size_t index;
        size_t area;
        float *row;

        if (!read_line(in, &line, &line_size)) {
            free(line);
            kyx_fixture_free(fixture);
            return fail(error, size, "short cell list");
        }
        level = (int)strtol(line, &cursor, 10);
        gy = (int)strtol(cursor, &cursor, 10);
        gx = (int)strtol(cursor, &cursor, 10);
        if (level < 0 || level >= KYX_LEVELS || gy < 0 || gx < 0 ||
            gy >= levels[level].height || gx >= levels[level].width) {
            free(line);
            kyx_fixture_free(fixture);
            return fail(error, size, "cell outside its grid");
        }
        area = (size_t)levels[level].width * (size_t)levels[level].height;
        index = (size_t)gy * (size_t)levels[level].width + (size_t)gx;
        row = fixture->flat + (offsets[level] + index) * KYX_CELL_VALUES;
        for (int k = 0; k < KYX_CELL_VALUES; k++) {
            char *end;
            float v = strtof(cursor, &end);
            if (end == cursor) {
                free(line);
                kyx_fixture_free(fixture);
                return fail(error, size, "cell has fewer than 85 values");
            }
            cursor = end;
            row[k] = v;
            if (fixture->raw) {
                fixture->planes[level][(size_t)k * area + index] = v;
            }
        }
    }

    if (!read_line(in, &line, &line_size) ||
        sscanf(line, "boxes %zu", &fixture->box_count) != 1) {
        free(line);
        kyx_fixture_free(fixture);
        return fail(error, size, "missing box count");
    }
    fixture->boxes = calloc(fixture->box_count + 1u, sizeof(kyx_box));
    if (fixture->boxes == NULL) {
        free(line);
        kyx_fixture_free(fixture);
        return fail(error, size, "out of memory");
    }
    for (size_t i = 0u; i < fixture->box_count; i++) {
        kyx_box *box = &fixture->boxes[i];
        if (!read_line(in, &line, &line_size) ||
            sscanf(line, "%d %f %f %f %f %f", &box->class_id, &box->score,
                   &box->x0, &box->y0, &box->x1, &box->y1) != 6) {
            free(line);
            kyx_fixture_free(fixture);
            return fail(error, size, "short box list");
        }
    }
    free(line);
    return true;
}

bool kyx_fixture_load(kyx_fixture *fixture, const char *path, char *error, size_t size)
{
    FILE *in = fopen(path, "r");
    bool ok;

    if (in == NULL) {
        memset(fixture, 0, sizeof(*fixture));
        return fail(error, size, "cannot open fixture");
    }
    ok = kyx_fixture_read(fixture, in, error, size);
    fclose(in);
    return ok;
}

void kyx_fixture_free(kyx_fixture *fixture)
{
    free(fixture->flat);
    for (int level = 0; level < KYX_LEVELS; level++) {
        free(fixture->planes[level]);
    }
    free(fixture->boxes);
    memset(fixture, 0, sizeof(*fixture));
}

size_t kyx_fixture_decode(
    const kyx_fixture *fixture, bool per_class, kyx_box *out, size_t capacity,
    size_t *candidates, size_t *dropped)
{
    size_t count;

    if (fixture->raw) {
        count = kyx_decode_raw(
            fixture->raw_levels, fixture->input_width, fixture->input_height,
            fixture->conf, out, capacity, dropped);
    } else {
        count = kyx_decode_flat(
            fixture->flat, fixture->total_cells, fixture->input_width,
            fixture->input_height, fixture->conf, out, capacity, dropped);
    }
    if (candidates != NULL) {
        *candidates = count;
    }
    return kyx_nms(out, count, fixture->iou, per_class);
}

size_t kyx_fixture_compare(
    const kyx_fixture *fixture, const kyx_box *got, size_t count,
    float score_tolerance, float pixel_tolerance)
{
    size_t i;

    if (count != fixture->box_count) {
        return count < fixture->box_count ? count : fixture->box_count;
    }
    for (i = 0u; i < count; i++) {
        const kyx_box *want = &fixture->boxes[i];
        if (got[i].class_id != want->class_id ||
            fabsf(got[i].score - want->score) > score_tolerance ||
            fabsf(got[i].x0 - want->x0) > pixel_tolerance ||
            fabsf(got[i].y0 - want->y0) > pixel_tolerance ||
            fabsf(got[i].x1 - want->x1) > pixel_tolerance ||
            fabsf(got[i].y1 - want->y1) > pixel_tolerance) {
            return i;
        }
    }
    return count;
}
