#include <string.h>

#include "check.h"
#include "kilix_yolox.h"

static kyx_box box(int class_id, float score, float x0, float y0, float x1, float y1)
{
    kyx_box b;
    b.class_id = class_id; b.score = score;
    b.x0 = x0; b.y0 = y0; b.x1 = x1; b.y1 = y1;
    return b;
}

static void test_duplicates(void)
{
    kyx_box boxes[3];

    boxes[0] = box(2, 0.8f, 10, 10, 50, 50);
    boxes[1] = box(2, 0.9f, 12, 11, 52, 49);
    boxes[2] = box(2, 0.3f, 200, 200, 240, 240);
    CHECK(kyx_nms(boxes, 3u, 0.45f, false) == 2u);
    CHECK_NEAR(boxes[0].score, 0.9f, 1e-6f);   /* the best of the pair, first */
    CHECK_NEAR(boxes[1].score, 0.3f, 1e-6f);
    CHECK(boxes[1].x0 == 200.0f);
}

static void test_class_handling(void)
{
    kyx_box boxes[2];

    /* A truck and a car on the same object: one box, unless asked otherwise. */
    boxes[0] = box(7, 0.36f, 380, 60, 580, 145);
    boxes[1] = box(2, 0.57f, 384, 66, 578, 143);
    CHECK(kyx_nms(boxes, 2u, 0.45f, false) == 1u);
    CHECK(boxes[0].class_id == 2);

    boxes[0] = box(7, 0.36f, 380, 60, 580, 145);
    boxes[1] = box(2, 0.57f, 384, 66, 578, 143);
    CHECK(kyx_nms(boxes, 2u, 0.45f, true) == 2u);
    CHECK(boxes[0].class_id == 2 && boxes[1].class_id == 7);
}

static void test_threshold_edge(void)
{
    kyx_box boxes[2];

    /* Two 10x10 boxes overlapping by half their width: IoU = 50/150 = 1/3. */
    boxes[0] = box(0, 0.9f, 0, 0, 10, 10);
    boxes[1] = box(0, 0.8f, 5, 0, 15, 10);
    CHECK(kyx_nms(boxes, 2u, 0.34f, false) == 2u);
    boxes[0] = box(0, 0.9f, 0, 0, 10, 10);
    boxes[1] = box(0, 0.8f, 5, 0, 15, 10);
    CHECK(kyx_nms(boxes, 2u, 0.33f, false) == 1u);
}

static void test_chain(void)
{
    /* A suppresses B; B would have suppressed C, but B is gone, so C
     * survives: greedy, not transitive. */
    kyx_box boxes[3];

    boxes[0] = box(0, 0.9f, 0, 0, 10, 10);     /* A */
    boxes[1] = box(0, 0.8f, 6, 0, 16, 10);     /* B: IoU with A = 4/16 = 0.25 */
    boxes[2] = box(0, 0.7f, 12, 0, 22, 10);    /* C: IoU with B = 0.25, with A = 0 */
    CHECK(kyx_nms(boxes, 3u, 0.2f, false) == 2u);
    CHECK_NEAR(boxes[0].score, 0.9f, 1e-6f);
    CHECK_NEAR(boxes[1].score, 0.7f, 1e-6f);
}

static void test_empty_and_order(void)
{
    kyx_box boxes[4];

    CHECK(kyx_nms(NULL, 3u, 0.45f, false) == 0u);
    CHECK(kyx_nms(boxes, 0u, 0.45f, false) == 0u);

    boxes[0] = box(1, 0.2f, 0, 0, 5, 5);
    boxes[1] = box(1, 0.7f, 100, 0, 105, 5);
    boxes[2] = box(1, 0.5f, 200, 0, 205, 5);
    boxes[3] = box(1, 0.9f, 300, 0, 305, 5);
    CHECK(kyx_nms(boxes, 4u, 0.45f, false) == 4u);
    CHECK(boxes[0].score == 0.9f && boxes[1].score == 0.7f &&
          boxes[2].score == 0.5f && boxes[3].score == 0.2f);
}

/*
 * The sort is this file's own - qsort() allocates past a kilobyte of
 * boxes - so it is held to what qsort promised: every score in order,
 * across a list as long as a caller's capacity, and, being stable, equal
 * scores in the order they arrived.
 */
static void test_long_list(void)
{
    static kyx_box boxes[1024];
    const size_t count = sizeof(boxes) / sizeof(boxes[0]);
    size_t kept;

    /* Worst case for insertion: ascending scores, so every box travels
     * to the front.  Placed apart so nothing suppresses anything. */
    for (size_t i = 0u; i < count; i++) {
        boxes[i] = box((int)(i % 80u), 0.001f * (float)(i + 1u),
                       (float)i * 20.0f, 0.0f, (float)i * 20.0f + 10.0f, 10.0f);
    }
    kept = kyx_nms(boxes, count, 0.45f, false);
    CHECK(kept == count);
    for (size_t i = 1u; i < kept; i++) {
        if (boxes[i - 1u].score < boxes[i].score) {
            CHECK(boxes[i - 1u].score >= boxes[i].score);
            break;
        }
    }
    CHECK_NEAR(boxes[0].score, 1.024f, 1e-6f);
    CHECK_NEAR(boxes[kept - 1u].score, 0.001f, 1e-6f);

    /* Ties keep their arrival order: the x0 of each tied box climbs. */
    for (size_t i = 0u; i < count; i++) {
        boxes[i] = box(0, i % 2u == 0u ? 0.9f : 0.8f,
                       (float)i * 20.0f, 0.0f, (float)i * 20.0f + 10.0f, 10.0f);
    }
    kept = kyx_nms(boxes, count, 0.45f, false);
    CHECK(kept == count);
    CHECK(boxes[0].score == 0.9f && boxes[count / 2u].score == 0.8f);
    for (size_t i = 1u; i < count / 2u; i++) {
        if (boxes[i - 1u].x0 >= boxes[i].x0) {
            CHECK(boxes[i - 1u].x0 < boxes[i].x0);
            break;
        }
    }
    CHECK(boxes[1].x0 == 40.0f);   /* the second 0.9 was box 2 */
}

int main(void)
{
    test_long_list();
    test_duplicates();
    test_class_handling();
    test_threshold_edge();
    test_chain();
    test_empty_and_order();
    return finish("test-nms");
}
