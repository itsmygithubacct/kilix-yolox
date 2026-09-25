#include <string.h>

#include "check.h"
#include "kilix_yolox.h"

static void test_pack(void)
{
    kyx_box boxes[3];
    float reply[KYX_REPLY_VALUES];

    boxes[0].class_id = 16; boxes[0].score = 0.9f;
    boxes[0].x0 = 20; boxes[0].y0 = 10; boxes[0].x1 = 120; boxes[0].y1 = 60;
    boxes[1].class_id = 1; boxes[1].score = 0.5f;
    boxes[1].x0 = 0; boxes[1].y0 = 0; boxes[1].x1 = 200; boxes[1].y1 = 100;
    boxes[2].class_id = 7; boxes[2].score = 0.3f;
    boxes[2].x0 = -5; boxes[2].y0 = 90; boxes[2].x1 = 250; boxes[2].y1 = 130;

    memset(reply, 0x7f, sizeof(reply));
    CHECK(kyx_reply(boxes, 3u, 200, 100, reply) == 3u);
    CHECK(KYX_REPLY_BYTES == 480);
    /* [class, score, y0, x0, y1, x1] */
    CHECK(reply[0] == 16.0f);
    CHECK_NEAR(reply[1], 0.9f, 1e-6f);
    CHECK_NEAR(reply[2], 0.1f, 1e-6f);
    CHECK_NEAR(reply[3], 0.1f, 1e-6f);
    CHECK_NEAR(reply[4], 0.6f, 1e-6f);
    CHECK_NEAR(reply[5], 0.6f, 1e-6f);
    CHECK(reply[6] == 1.0f && reply[8] == 0.0f && reply[11] == 1.0f);
    /* Out of frame is clamped, not wrapped and not refused. */
    CHECK(reply[12] == 7.0f);
    CHECK_NEAR(reply[15], 0.0f, 1e-6f);
    CHECK_NEAR(reply[16], 1.0f, 1e-6f);
    CHECK_NEAR(reply[17], 1.0f, 1e-6f);
    /* Everything after the last box is zero, whatever was there before. */
    for (int i = 18; i < KYX_REPLY_VALUES; i++) {
        if (reply[i] != 0.0f) {
            CHECK(reply[i] == 0.0f);
            break;
        }
    }
}

static void test_overflow(void)
{
    kyx_box boxes[25];
    float reply[KYX_REPLY_VALUES];

    for (int i = 0; i < 25; i++) {
        boxes[i].class_id = i; boxes[i].score = 1.0f - (float)i * 0.01f;
        boxes[i].x0 = 0; boxes[i].y0 = 0; boxes[i].x1 = 10; boxes[i].y1 = 10;
    }
    CHECK(kyx_reply(boxes, 25u, 100, 100, reply) == (size_t)KYX_REPLY_ROWS);
    CHECK(reply[19 * KYX_REPLY_COLUMNS] == 19.0f);
    CHECK(kyx_reply(boxes, 0u, 100, 100, reply) == 0u);
    CHECK(reply[0] == 0.0f && reply[1] == 0.0f);
    CHECK(kyx_reply(NULL, 3u, 100, 100, reply) == 0u);
    CHECK(kyx_reply(boxes, 3u, 0, 100, reply) == 0u);
}

static void test_unletterbox(void)
{
    kyx_box boxes[2];

    boxes[0].class_id = 0; boxes[0].score = 0.5f;
    boxes[0].x0 = 10; boxes[0].y0 = 20; boxes[0].x1 = 30; boxes[0].y1 = 40;
    boxes[1] = boxes[0];
    boxes[1].x1 = 400; boxes[1].y0 = -3;
    kyx_unletterbox(boxes, 2u, 0.5f, 640, 360);
    CHECK_NEAR(boxes[0].x0, 20.0f, 1e-5f);
    CHECK_NEAR(boxes[0].y0, 40.0f, 1e-5f);
    CHECK_NEAR(boxes[0].x1, 60.0f, 1e-5f);
    CHECK_NEAR(boxes[0].y1, 80.0f, 1e-5f);
    CHECK_NEAR(boxes[1].x1, 640.0f, 1e-5f);   /* 800 clamped to the frame */
    CHECK_NEAR(boxes[1].y0, 0.0f, 1e-5f);
    /* A ratio of zero would divide by it; it is refused, the boxes untouched. */
    kyx_unletterbox(boxes, 2u, 0.0f, 640, 360);
    CHECK_NEAR(boxes[0].x0, 20.0f, 1e-5f);
}

static void test_names(void)
{
    CHECK(strcmp(kyx_label(0), "person") == 0);
    CHECK(strcmp(kyx_label(16), "dog") == 0);
    CHECK(strcmp(kyx_label(79), "toothbrush") == 0);
    CHECK(kyx_label(80) == NULL && kyx_label(-1) == NULL);
    CHECK(kyx_class_from_name("truck") == 7);
    CHECK(kyx_class_from_name("unicorn") == -1);
    CHECK(kyx_class_from_name(NULL) == -1);
}

int main(void)
{
    test_pack();
    test_overflow();
    test_unletterbox();
    test_names();
    return finish("test-reply");
}
