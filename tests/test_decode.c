#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "kilix_yolox.h"
#include "kyx_fixture.h"

#ifndef KYX_FIXTURE_DIR
#define KYX_FIXTURE_DIR "tests/fixtures"
#endif

#define CAPACITY 1024

static void test_geometry(void)
{
    kyx_level levels[KYX_LEVELS];

    CHECK(kyx_levels(320, 320, levels));
    CHECK(levels[0].stride == 8 && levels[0].width == 40 && levels[0].height == 40);
    CHECK(levels[1].stride == 16 && levels[1].width == 20);
    CHECK(levels[2].stride == 32 && levels[2].height == 10);
    CHECK(kyx_cells(320, 320) == 2100u);
    CHECK(kyx_cells(640, 640) == 8400u);
    CHECK(kyx_cells(416, 416) == 3549u);
    CHECK(kyx_cells(320, 224) == 1470u);
    CHECK(!kyx_levels(300, 300, levels));
    CHECK(!kyx_levels(0, 320, levels));
    CHECK(kyx_cells(-32, 32) == 0u);
}

/* Every cell, no gate: what the gate must be indistinguishable from. */
static size_t naive_raw_count(const kyx_fixture *fixture)
{
    kyx_level levels[KYX_LEVELS];
    size_t count = 0u;

    kyx_levels(fixture->input_width, fixture->input_height, levels);
    for (int level = 0; level < KYX_LEVELS; level++) {
        const size_t area = (size_t)levels[level].width * (size_t)levels[level].height;
        const kyx_raw_level *raw = &fixture->raw_levels[level];
        for (size_t cell = 0u; cell < area; cell++) {
            float best = raw->cls[cell];
            for (int c = 1; c < KYX_CLASSES; c++) {
                float v = raw->cls[(size_t)c * area + cell];
                if (v > best) {
                    best = v;
                }
            }
            if (kyx_sigmoid(raw->obj[cell]) * kyx_sigmoid(best) > fixture->conf) {
                count++;
            }
        }
    }
    return count;
}

static void test_fixture(const char *name)
{
    kyx_fixture fixture;
    kyx_box boxes[CAPACITY];
    char error[128];
    char path[512];
    size_t candidates;
    size_t dropped;
    size_t count;
    size_t first;

    snprintf(path, sizeof(path), "%s/%s", KYX_FIXTURE_DIR, name);
    CHECK(kyx_fixture_load(&fixture, path, error, sizeof(error)));
    if (fixture.flat == NULL) {
        fprintf(stderr, "%s: %s\n", path, error);
        return;
    }
    CHECK(fixture.box_count > 0u);

    count = kyx_fixture_decode(&fixture, false, boxes, CAPACITY, &candidates, &dropped);
    CHECK(dropped == 0u);
    CHECK(candidates >= count);
    CHECK(count == fixture.box_count);
    first = kyx_fixture_compare(&fixture, boxes, count, 0.001f, 0.05f);
    CHECK(first == count);
    if (first != count && first < count) {
        fprintf(stderr, "%s: box %zu: got %d %.3f (%.1f %.1f %.1f %.1f)\n",
                name, first, boxes[first].class_id, (double)boxes[first].score,
                (double)boxes[first].x0, (double)boxes[first].y0,
                (double)boxes[first].x1, (double)boxes[first].y1);
    }
    /* Score order out of NMS. */
    for (size_t i = 1u; i < count; i++) {
        CHECK(boxes[i - 1u].score >= boxes[i].score);
    }

    /* The gate rejects exactly what the full computation would. */
    if (fixture.raw) {
        size_t naive = naive_raw_count(&fixture);
        size_t gated = kyx_decode_raw(
            fixture.raw_levels, fixture.input_width, fixture.input_height,
            fixture.conf, boxes, CAPACITY, NULL);
        CHECK(naive == gated);
        CHECK(naive == candidates);
    }

    /* Capacity is honoured and truncation is reported, not hidden. */
    if (candidates > 2u) {
        size_t small;
        if (fixture.raw) {
            small = kyx_decode_raw(
                fixture.raw_levels, fixture.input_width, fixture.input_height,
                fixture.conf, boxes, 2u, &dropped);
        } else {
            small = kyx_decode_flat(
                fixture.flat, fixture.total_cells, fixture.input_width,
                fixture.input_height, fixture.conf, boxes, 2u, &dropped);
        }
        CHECK(small == 2u);
        CHECK(dropped == candidates - 2u);
    }

    /* A threshold nothing can pass, and one everything present passes. */
    if (fixture.raw) {
        CHECK(kyx_decode_raw(fixture.raw_levels, fixture.input_width,
                             fixture.input_height, 0.999f, boxes, CAPACITY,
                             NULL) == 0u);
        CHECK(kyx_decode_raw(fixture.raw_levels, fixture.input_width,
                             fixture.input_height, 0.0f, boxes, CAPACITY,
                             &dropped) + dropped >= fixture.cell_count - 8u);
    } else {
        CHECK(kyx_decode_flat(fixture.flat, fixture.total_cells,
                              fixture.input_width, fixture.input_height,
                              0.999f, boxes, CAPACITY, NULL) == 0u);
        /* The wrong cell count is refused outright. */
        CHECK(kyx_decode_flat(fixture.flat, fixture.total_cells - 1u,
                              fixture.input_width, fixture.input_height,
                              fixture.conf, boxes, CAPACITY, NULL) == 0u);
    }

    /* Per-class suppression keeps at least as many boxes. */
    {
        size_t per_class = kyx_fixture_decode(&fixture, true, boxes, CAPACITY, NULL, NULL);
        CHECK(per_class >= count);
    }
    kyx_fixture_free(&fixture);
}

static void test_arithmetic(void)
{
    /* One cell at stride 32, by hand: grid (3, 2), reg (0.5, -0.25, log 2, log 4),
     * obj logit 2, class 7 logit 1.  cx = 3.5*32 = 112, cy = 1.75*32 = 56,
     * w = 64, h = 128; score = sig(2) * sig(1). */
    float reg[4 * 100];
    float obj[100];
    float cls[80 * 100];
    kyx_raw_level raw[KYX_LEVELS];
    kyx_box boxes[4];
    size_t count;
    const size_t cell = 2u * 10u + 3u;

    memset(reg, 0, sizeof(reg));
    memset(cls, 0, sizeof(cls));
    for (size_t i = 0u; i < 100u; i++) {
        obj[i] = -30.0f;
    }
    reg[0 * 100 + cell] = 0.5f;
    reg[1 * 100 + cell] = -0.25f;
    reg[2 * 100 + cell] = 0.6931472f;
    reg[3 * 100 + cell] = 1.3862944f;
    obj[cell] = 2.0f;
    cls[7 * 100 + cell] = 1.0f;
    raw[0].reg = NULL; raw[0].obj = NULL; raw[0].cls = NULL;
    raw[1] = raw[0];
    raw[2].reg = reg; raw[2].obj = obj; raw[2].cls = cls;

    count = kyx_decode_raw(raw, 320, 320, 0.25f, boxes, 4u, NULL);
    CHECK(count == 1u);
    CHECK(boxes[0].class_id == 7);
    CHECK_NEAR(boxes[0].score, kyx_sigmoid(2.0f) * kyx_sigmoid(1.0f), 1e-6f);
    CHECK_NEAR(boxes[0].x0, 112.0f - 32.0f, 1e-3f);
    CHECK_NEAR(boxes[0].y0, 56.0f - 64.0f, 1e-3f);
    CHECK_NEAR(boxes[0].x1, 112.0f + 32.0f, 1e-3f);
    CHECK_NEAR(boxes[0].y1, 56.0f + 64.0f, 1e-3f);

    /* The same cell in the export's form: 1600 + 400 cells precede stride 32. */
    {
        static float cells[2100 * KYX_CELL_VALUES];
        float *row = cells + (2000u + cell) * KYX_CELL_VALUES;
        memset(cells, 0, sizeof(cells));
        row[0] = 0.5f; row[1] = -0.25f; row[2] = 0.6931472f; row[3] = 1.3862944f;
        row[4] = kyx_sigmoid(2.0f);
        row[5 + 7] = kyx_sigmoid(1.0f);
        count = kyx_decode_flat(cells, 2100u, 320, 320, 0.25f, boxes, 4u, NULL);
        CHECK(count == 1u);
        CHECK(boxes[0].class_id == 7);
        CHECK_NEAR(boxes[0].score, kyx_sigmoid(2.0f) * kyx_sigmoid(1.0f), 1e-6f);
        CHECK_NEAR(boxes[0].x0, 80.0f, 1e-3f);
        CHECK_NEAR(boxes[0].y1, 120.0f, 1e-3f);
    }

    CHECK_NEAR(kyx_logit(kyx_sigmoid(1.5f)), 1.5f, 1e-5f);
    CHECK_NEAR(kyx_sigmoid(0.0f), 0.5f, 1e-7f);
}

/*
 * What a model must not be able to do to the caller.  A head that emits a
 * NaN, or a size logit big enough to overflow exp(), produces a box with
 * no coordinates; passed on, the caller converts it to int, which is
 * undefined, or draws a frame-sized box round nothing.  Not a detection.
 */
static void test_non_finite(void)
{
    float reg[4 * 100];
    float obj[100];
    float cls[80 * 100];
    kyx_raw_level raw[KYX_LEVELS];
    kyx_box boxes[8];
    size_t dropped = 99u;
    const float nan = strtof("nan", NULL);

    memset(reg, 0, sizeof(reg));
    memset(cls, 0, sizeof(cls));
    for (size_t i = 0u; i < 100u; i++) {
        obj[i] = -30.0f;
    }
    raw[0].reg = NULL; raw[0].obj = NULL; raw[0].cls = NULL;
    raw[1] = raw[0];
    raw[2].reg = reg; raw[2].obj = obj; raw[2].cls = cls;

    /* Cell 0: objectness is NaN.  Cell 1: a width logit that overflows
     * exp().  Cell 2: a NaN centre.  Cell 3: the control, a plain box. */
    obj[0] = nan;                 cls[7 * 100 + 0] = 3.0f;
    obj[1] = 3.0f; reg[2 * 100 + 1] = 1000.0f; cls[7 * 100 + 1] = 3.0f;
    obj[2] = 3.0f; reg[0 * 100 + 2] = nan;     cls[7 * 100 + 2] = 3.0f;
    obj[3] = 3.0f;                cls[7 * 100 + 3] = 3.0f;

    CHECK(kyx_decode_raw(raw, 320, 320, 0.25f, boxes, 8u, &dropped) == 1u);
    CHECK(dropped == 0u);
    CHECK(boxes[0].x0 == boxes[0].x0);   /* not NaN */
    CHECK_NEAR(boxes[0].x0, 3.0f * 32.0f - 16.0f, 1e-3f);

    /* The same three in the export's form. */
    {
        static float cells[2100 * KYX_CELL_VALUES];
        float *row;
        memset(cells, 0, sizeof(cells));
        row = cells + (2000u + 0u) * KYX_CELL_VALUES;
        row[4] = nan; row[5 + 7] = 0.9f;
        row = cells + (2000u + 1u) * KYX_CELL_VALUES;
        row[2] = 1000.0f; row[4] = 0.9f; row[5 + 7] = 0.9f;
        row = cells + (2000u + 2u) * KYX_CELL_VALUES;
        row[1] = nan; row[4] = 0.9f; row[5 + 7] = 0.9f;
        row = cells + (2000u + 3u) * KYX_CELL_VALUES;
        row[4] = 0.9f; row[5 + 7] = 0.9f;
        CHECK(kyx_decode_flat(cells, 2100u, 320, 320, 0.25f, boxes, 8u, &dropped) == 1u);
        CHECK(dropped == 0u);
        CHECK_NEAR(boxes[0].x0, 3.0f * 32.0f - 16.0f, 1e-3f);
    }

    /* A NaN score cannot be compared, so it must not sort, suppress or
     * be kept: kyx_nms() drops it. */
    {
        kyx_box list[3];
        list[0].class_id = 0; list[0].score = 0.5f;
        list[0].x0 = 0; list[0].y0 = 0; list[0].x1 = 10; list[0].y1 = 10;
        list[1] = list[0]; list[1].score = nan; list[1].x0 = 100; list[1].x1 = 110;
        list[2] = list[0]; list[2].score = 0.7f; list[2].x0 = 200; list[2].x1 = 210;
        CHECK(kyx_nms(list, 3u, 0.45f, false) == 2u);
        CHECK_NEAR(list[0].score, 0.7f, 1e-6f);
        CHECK_NEAR(list[1].score, 0.5f, 1e-6f);
    }
}

int main(void)
{
    test_geometry();
    test_arithmetic();
    test_non_finite();
    test_fixture("raw_320.txt");
    test_fixture("flat_640.txt");
    return finish("test-decode");
}
