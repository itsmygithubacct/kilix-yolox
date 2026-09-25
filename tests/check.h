#ifndef KYX_CHECK_H
#define KYX_CHECK_H

#include <math.h>
#include <stdio.h>

static int checks_run = 0;
static int checks_failed = 0;

#define CHECK(condition) \
    do { \
        checks_run++; \
        if (!(condition)) { \
            checks_failed++; \
            fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, \
                    #condition); \
        } \
    } while (0)

#define CHECK_NEAR(a, b, tolerance) \
    do { \
        checks_run++; \
        if (fabs((double)(a) - (double)(b)) > (double)(tolerance)) { \
            checks_failed++; \
            fprintf(stderr, "%s:%d: failed: %s ~ %s (%g vs %g)\n", __FILE__, \
                    __LINE__, #a, #b, (double)(a), (double)(b)); \
        } \
    } while (0)

static int finish(const char *name)
{
    printf("%s: %d checks, %d failed\n", name, checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}

#endif
