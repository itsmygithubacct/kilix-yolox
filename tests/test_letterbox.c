#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "kilix_yolox.h"

#ifndef KYX_FIXTURE_DIR
#define KYX_FIXTURE_DIR "tests/fixtures"
#endif

static char *read_line(FILE *in)
{
    size_t size = 1u << 16;
    char *line = malloc(size);
    size_t used = 0u;

    for (;;) {
        int c = fgetc(in);
        if (c == EOF || c == '\n') {
            line[used] = '\0';
            return line;
        }
        if (used + 2u >= size) {
            size *= 2u;
            line = realloc(line, size);
        }
        line[used++] = (char)c;
    }
}

static size_t parse_bytes(const char *text, uint8_t *out, size_t capacity)
{
    size_t count = 0u;
    const char *cursor = text;

    while (count < capacity) {
        char *end;
        long v = strtol(cursor, &end, 10);
        if (end == cursor) {
            break;
        }
        out[count++] = (uint8_t)v;
        cursor = end;
    }
    return count;
}

static void test_fixture(void)
{
    FILE *in = fopen(KYX_FIXTURE_DIR "/letterbox.txt", "r");
    char *line;
    int sw;
    int sh;
    int tw;
    int th;
    float ratio;
    uint8_t src[4096];
    uint8_t want[4096];
    uint8_t got[4096];
    uint8_t alt[4096];
    size_t src_bytes;
    size_t dst_bytes;

    CHECK(in != NULL);
    if (in == NULL) {
        return;
    }
    line = read_line(in);
    CHECK(strcmp(line, "kilix-yolox-letterbox 1") == 0);
    free(line);
    line = read_line(in);
    CHECK(sscanf(line, "source %d %d bgra", &sw, &sh) == 2);
    free(line);
    line = read_line(in);
    CHECK(sscanf(line, "target %d %d", &tw, &th) == 2);
    free(line);
    line = read_line(in);
    CHECK(sscanf(line, "ratio %f", &ratio) == 1);
    free(line);
    line = read_line(in);
    CHECK(strncmp(line, "src ", 4) == 0);
    src_bytes = parse_bytes(line + 4, src, sizeof(src));
    free(line);
    line = read_line(in);
    CHECK(strncmp(line, "dst ", 4) == 0);
    dst_bytes = parse_bytes(line + 4, want, sizeof(want));
    free(line);
    fclose(in);

    CHECK(src_bytes == (size_t)(sw * sh * 4));
    CHECK(dst_bytes == (size_t)(tw * th * 3));

    /* BGRA, as the reference produced it. */
    CHECK_NEAR(kyx_letterbox(src, sw, sh, KYX_PIXFMT_BGRA, got, tw, th),
               ratio, 1e-6f);
    CHECK(memcmp(got, want, dst_bytes) == 0);

    /* The same pixels as RGBA must give the same planes: the channel
     * order is the input's problem, not the output's. */
    for (int i = 0; i < sw * sh; i++) {
        alt[i * 4 + 0] = src[i * 4 + 2];
        alt[i * 4 + 1] = src[i * 4 + 1];
        alt[i * 4 + 2] = src[i * 4 + 0];
        alt[i * 4 + 3] = src[i * 4 + 3];
    }
    memset(got, 0, sizeof(got));
    CHECK(kyx_letterbox(alt, sw, sh, KYX_PIXFMT_RGBA, got, tw, th) > 0.0f);
    CHECK(memcmp(got, want, dst_bytes) == 0);

    /* And as packed BGR without the alpha. */
    for (int i = 0; i < sw * sh; i++) {
        alt[i * 3 + 0] = src[i * 4 + 0];
        alt[i * 3 + 1] = src[i * 4 + 1];
        alt[i * 3 + 2] = src[i * 4 + 2];
    }
    memset(got, 0, sizeof(got));
    CHECK(kyx_letterbox(alt, sw, sh, KYX_PIXFMT_BGR, got, tw, th) > 0.0f);
    CHECK(memcmp(got, want, dst_bytes) == 0);
}

static void test_upscale_and_fill(void)
{
    /* A 4x2 source into an 8x8 square: ratio 2, the image fills the top
     * half, the bottom half is 114 in every plane. */
    uint8_t src[4 * 2 * 4];
    uint8_t dst[8 * 8 * 3];
    float ratio;

    for (int i = 0; i < 8; i++) {
        src[i * 4 + 0] = (uint8_t)(10 + i);   /* B */
        src[i * 4 + 1] = (uint8_t)(100 + i);  /* G */
        src[i * 4 + 2] = (uint8_t)(200 + i);  /* R */
        src[i * 4 + 3] = 255;
    }
    ratio = kyx_letterbox(src, 4, 2, KYX_PIXFMT_BGRA, dst, 8, 8);
    CHECK_NEAR(ratio, 2.0f, 1e-6f);
    /* Blue plane, row 0: each source pixel twice. */
    CHECK(dst[0] == 10 && dst[1] == 10 && dst[2] == 11 && dst[7] == 13);
    /* Row 3 is still source row 1. */
    CHECK(dst[3 * 8 + 0] == 14 && dst[3 * 8 + 7] == 17);
    /* Rows 4-7 are fill, in every plane. */
    for (int plane = 0; plane < 3; plane++) {
        for (int i = 4 * 8; i < 8 * 8; i++) {
            if (dst[plane * 64 + i] != KYX_FILL) {
                CHECK(dst[plane * 64 + i] == KYX_FILL);
                break;
            }
        }
    }
    /* Green and red planes carry their own channels. */
    CHECK(dst[64 + 0] == 100 && dst[128 + 0] == 200);
}

static void test_bad_arguments(void)
{
    uint8_t src[16];
    uint8_t dst[48];

    CHECK(kyx_letterbox(NULL, 2, 2, KYX_PIXFMT_BGRA, dst, 4, 4) == 0.0f);
    CHECK(kyx_letterbox(src, 0, 2, KYX_PIXFMT_BGRA, dst, 4, 4) == 0.0f);
    CHECK(kyx_letterbox(src, 2, 2, KYX_PIXFMT_BGRA, NULL, 4, 4) == 0.0f);
    CHECK(kyx_letterbox(src, 2, 2, KYX_PIXFMT_BGRA, dst, 4, 0) == 0.0f);
    CHECK(kyx_letterbox(src, 2, 2, (kyx_pixfmt)99, dst, 4, 4) == 0.0f);
}

/*
 * The shapes a caller can hand this that the fixture does not: a source
 * of one pixel, a source thinner than one output pixel, a square smaller
 * than the model's input, and one larger.  None may read outside the
 * source or write outside the destination; the sanitizer build holds
 * them to that, this holds them to the reference's arithmetic.
 */
static void test_odd_sizes(void)
{
    static uint8_t dst[64 * 64 * 3];
    uint8_t src[4 * 200];
    float ratio;

    /* One pixel into 64x64: every output pixel is that pixel. */
    src[0] = 1; src[1] = 2; src[2] = 3; src[3] = 255;
    ratio = kyx_letterbox(src, 1, 1, KYX_PIXFMT_BGRA, dst, 64, 64);
    CHECK_NEAR(ratio, 64.0f, 1e-6f);
    CHECK(dst[0] == 1 && dst[64 * 64 - 1] == 1);
    CHECK(dst[64 * 64] == 2 && dst[2 * 64 * 64 - 1] == 2);
    CHECK(dst[2 * 64 * 64] == 3 && dst[3 * 64 * 64 - 1] == 3);

    /* 200x1 into 64x64: the ratio is 0.32 and one source row scales to
     * int(0.32) = 0 rows, so the reference draws nothing: all fill. */
    for (int i = 0; i < 200; i++) {
        src[i * 4 + 0] = 9; src[i * 4 + 1] = 9; src[i * 4 + 2] = 9; src[i * 4 + 3] = 255;
    }
    ratio = kyx_letterbox(src, 200, 1, KYX_PIXFMT_BGRA, dst, 64, 64);
    CHECK_NEAR(ratio, 0.32f, 1e-6f);
    CHECK(dst[0] == KYX_FILL && dst[63] == KYX_FILL);

    /* 1x200 the other way: the one column scales to int(0.32) = 0
     * columns, so this too is all fill - the reference's truncation, kept
     * rather than rounded up, because a box found in a column the model
     * was never shown would be a box in nothing. */
    ratio = kyx_letterbox(src, 1, 200, KYX_PIXFMT_BGRA, dst, 64, 64);
    CHECK_NEAR(ratio, 0.32f, 1e-6f);
    CHECK(dst[0] == KYX_FILL && dst[63 * 64] == KYX_FILL);

    /* A 32x32 square into 64x64 (a small frame into a bigger model) and
     * a 64x64 square into 32x32 (the other way): the drawn area is the
     * whole destination both times and the ratio is exact. */
    for (int i = 0; i < 4 * 200; i++) {
        src[i] = (uint8_t)i;
    }
    ratio = kyx_letterbox(src, 10, 10, KYX_PIXFMT_BGRA, dst, 64, 64);
    CHECK_NEAR(ratio, 6.4f, 1e-6f);
    CHECK(dst[63] == src[9 * 4]);                  /* last column, row 0 */
    CHECK(dst[63 * 64] == src[9 * 10 * 4]);        /* row 63 is source row 9 */
    ratio = kyx_letterbox(src, 10, 20, KYX_PIXFMT_BGRA, dst, 4, 4);
    CHECK_NEAR(ratio, 0.2f, 1e-6f);
    CHECK(dst[0] == src[0] && dst[1] == src[5 * 4]);   /* x=1 is source x=5 */
    CHECK(dst[2] == KYX_FILL);                          /* 10*0.2 = 2 columns */
}

int main(void)
{
    test_fixture();
    test_upscale_and_fill();
    test_odd_sizes();
    test_bad_arguments();
    return finish("test-letterbox");
}
