/*
 * kilix-yolox: the arithmetic, exercised.
 *
 * There is no model here.  The command reads a fixture - cells of head
 * output and the boxes they must become - and decodes it, times it, or
 * checks it.  That is what a build server can run, and it is what tells
 * a port to a new machine that its decode is right before a single
 * inference has happened.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kilix_yolox.h"
#include "kyx_fixture.h"
#include "selftest_fixture.h"

#define BOX_CAPACITY 1024
#define SCORE_TOLERANCE 0.001f
#define PIXEL_TOLERANCE 0.05f

static void usage(FILE *out)
{
    fprintf(out,
        "usage:\n"
        "  kilix-yolox decode FIXTURE [--per-class] [--check]\n"
        "  kilix-yolox bench FIXTURE [ITERATIONS]\n"
        "  kilix-yolox classes\n"
        "  kilix-yolox --selftest\n"
        "  kilix-yolox --version\n"
        "\n"
        "  decode   boxes from a fixture's head output; --check compares them\n"
        "           with the boxes the fixture expects\n"
        "  bench    time decode and suppression, in C, on a fixture\n"
        "  classes  the eighty labels, by id\n");
}

static int fail_load(const char *path, const char *error)
{
    fprintf(stderr, "kilix-yolox: %s: %s\n", path, error);
    return 2;
}

static void print_boxes(const kyx_box *boxes, size_t count)
{
    for (size_t i = 0u; i < count; i++) {
        const char *label = kyx_label(boxes[i].class_id);
        printf("%-14s %2d  %.3f  %8.2f %8.2f %8.2f %8.2f\n",
               label != NULL ? label : "?", boxes[i].class_id, boxes[i].score,
               boxes[i].x0, boxes[i].y0, boxes[i].x1, boxes[i].y1);
    }
}

static int check(const kyx_fixture *fixture, const kyx_box *got, size_t count)
{
    size_t first = kyx_fixture_compare(
        fixture, got, count, SCORE_TOLERANCE, PIXEL_TOLERANCE);

    if (count == fixture->box_count && first == count) {
        printf("check: %zu boxes match the fixture\n", count);
        return 0;
    }
    if (count != fixture->box_count) {
        fprintf(stderr, "check: got %zu boxes, fixture expects %zu\n",
                count, fixture->box_count);
    } else {
        fprintf(stderr, "check: box %zu differs from the fixture\n", first);
    }
    fprintf(stderr, "expected:\n");
    print_boxes(fixture->boxes, fixture->box_count);
    return 1;
}

static int cmd_decode(int argc, char **argv)
{
    kyx_fixture fixture;
    kyx_box boxes[BOX_CAPACITY];
    char error[128];
    bool per_class = false;
    bool do_check = false;
    const char *path = NULL;
    size_t candidates;
    size_t dropped;
    size_t count;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--per-class") == 0) {
            per_class = true;
        } else if (strcmp(argv[i], "--check") == 0) {
            do_check = true;
        } else if (path == NULL) {
            path = argv[i];
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (path == NULL) {
        usage(stderr);
        return 2;
    }
    if (!kyx_fixture_load(&fixture, path, error, sizeof(error))) {
        return fail_load(path, error);
    }
    count = kyx_fixture_decode(
        &fixture, per_class, boxes, BOX_CAPACITY, &candidates, &dropped);
    printf("%s form, %dx%d, conf %.2f, iou %.2f, %s nms: "
           "%zu candidates, %zu kept, %zu dropped\n",
           fixture.raw ? "raw" : "flat", fixture.input_width,
           fixture.input_height, (double)fixture.conf, (double)fixture.iou,
           per_class ? "per-class" : "class-agnostic", candidates, count,
           dropped);
    print_boxes(boxes, count);
    if (do_check) {
        int status = check(&fixture, boxes, count);
        kyx_fixture_free(&fixture);
        return status;
    }
    kyx_fixture_free(&fixture);
    return 0;
}

static double now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int cmd_bench(int argc, char **argv)
{
    kyx_fixture fixture;
    kyx_box boxes[BOX_CAPACITY];
    char error[128];
    long iterations = 1000;
    size_t candidates = 0u;
    size_t count = 0u;
    double best = 1e9;
    double total = 0.0;

    if (argc < 1) {
        usage(stderr);
        return 2;
    }
    if (argc >= 2) {
        iterations = strtol(argv[1], NULL, 10);
        if (iterations <= 0) {
            iterations = 1;
        }
    }
    if (!kyx_fixture_load(&fixture, argv[0], error, sizeof(error))) {
        return fail_load(argv[0], error);
    }
    /* One untimed pass so the first iteration is not paying for pages. */
    kyx_fixture_decode(&fixture, false, boxes, BOX_CAPACITY, NULL, NULL);
    for (long i = 0; i < iterations; i++) {
        double t0 = now_seconds();
        double dt;

        count = kyx_fixture_decode(
            &fixture, false, boxes, BOX_CAPACITY, &candidates, NULL);
        dt = now_seconds() - t0;
        total += dt;
        if (dt < best) {
            best = dt;
        }
    }
    printf("%s form %dx%d: %zu cells, %zu candidates, %zu kept\n",
           fixture.raw ? "raw" : "flat", fixture.input_width,
           fixture.input_height, fixture.total_cells, candidates, count);
    printf("decode+nms in C: %ld iterations, mean %.1f us, best %.1f us\n",
           iterations, total / (double)iterations * 1e6, best * 1e6);
    kyx_fixture_free(&fixture);
    return 0;
}

static int cmd_classes(void)
{
    for (int i = 0; i < KYX_CLASSES; i++) {
        printf("%2d %s\n", i, kyx_label(i));
    }
    return 0;
}

static int selftest(void)
{
    kyx_fixture fixture;
    kyx_box boxes[BOX_CAPACITY];
    char error[128];
    FILE *in;
    size_t count;
    int status;

    /* The fixture travels as one literal per line - a single literal of
     * this size is past what pedantic C promises to accept - and is
     * joined here, once, into a buffer fmemopen() can read. */
    size_t total = 0u;
    char *text;
    char *cursor;

    for (size_t i = 0u; kyx_selftest_lines[i] != NULL; i++) {
        total += strlen(kyx_selftest_lines[i]);
    }
    text = malloc(total + 1u);
    if (text == NULL) {
        fprintf(stderr, "selftest: out of memory\n");
        return 2;
    }
    cursor = text;
    for (size_t i = 0u; kyx_selftest_lines[i] != NULL; i++) {
        size_t n = strlen(kyx_selftest_lines[i]);
        memcpy(cursor, kyx_selftest_lines[i], n);
        cursor += n;
    }
    *cursor = '\0';
    in = fmemopen(text, total, "r");
    if (in == NULL) {
        free(text);
        fprintf(stderr, "selftest: cannot open the embedded fixture\n");
        return 2;
    }
    if (!kyx_fixture_read(&fixture, in, error, sizeof(error))) {
        fclose(in);
        free(text);
        return fail_load("embedded fixture", error);
    }
    fclose(in);
    free(text);
    count = kyx_fixture_decode(&fixture, false, boxes, BOX_CAPACITY, NULL, NULL);
    status = check(&fixture, boxes, count);
    if (status == 0) {
        printf("selftest: %s form %dx%d decodes to the expected boxes\n",
               fixture.raw ? "raw" : "flat", fixture.input_width,
               fixture.input_height);
    }
    kyx_fixture_free(&fixture);
    return status;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    if (strcmp(argv[1], "--selftest") == 0) {
        return selftest();
    }
    if (strcmp(argv[1], "--version") == 0) {
        printf("kilix-yolox %d.%d.%d\n", KILIX_YOLOX_VERSION_MAJOR,
               KILIX_YOLOX_VERSION_MINOR, KILIX_YOLOX_VERSION_PATCH);
        return 0;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout);
        return 0;
    }
    if (strcmp(argv[1], "decode") == 0) {
        return cmd_decode(argc - 2, argv + 2);
    }
    if (strcmp(argv[1], "bench") == 0) {
        return cmd_bench(argc - 2, argv + 2);
    }
    if (strcmp(argv[1], "classes") == 0) {
        return cmd_classes();
    }
    usage(stderr);
    return 2;
}
