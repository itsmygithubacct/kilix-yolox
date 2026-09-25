#include "kilix_yolox.h"

#include <math.h>
#include <string.h>

static const int STRIDES[KYX_LEVELS] = {8, 16, 32};

static const char *const LABELS[KYX_CLASSES] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

/* ------------------------------- names --------------------------------- */

const char *kyx_label(int class_id)
{
    if (class_id < 0 || class_id >= KYX_CLASSES) {
        return NULL;
    }
    return LABELS[class_id];
}

int kyx_class_from_name(const char *name)
{
    if (name == NULL) {
        return -1;
    }
    for (int i = 0; i < KYX_CLASSES; i++) {
        if (strcmp(LABELS[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

float kyx_sigmoid(float x)
{
    if (x > 30.0f) {
        x = 30.0f;
    } else if (x < -30.0f) {
        x = -30.0f;
    }
    return 1.0f / (1.0f + expf(-x));
}

float kyx_logit(float p)
{
    /* Outside (0, 1) there is no logit; clamp so a caller asking for
     * "everything" or "nothing" gets a gate that does that. */
    if (p < 1e-6f) {
        p = 1e-6f;
    } else if (p > 1.0f - 1e-6f) {
        p = 1.0f - 1e-6f;
    }
    return logf(p / (1.0f - p));
}

/* ------------------------------- geometry ------------------------------ */

bool kyx_levels(int input_width, int input_height, kyx_level out[KYX_LEVELS])
{
    if (input_width <= 0 || input_height <= 0 || input_width % 32 != 0 ||
        input_height % 32 != 0) {
        return false;
    }
    for (int i = 0; i < KYX_LEVELS; i++) {
        out[i].stride = STRIDES[i];
        out[i].width = input_width / STRIDES[i];
        out[i].height = input_height / STRIDES[i];
    }
    return true;
}

size_t kyx_cells(int input_width, int input_height)
{
    kyx_level levels[KYX_LEVELS];
    size_t total = 0u;

    if (!kyx_levels(input_width, input_height, levels)) {
        return 0u;
    }
    for (int i = 0; i < KYX_LEVELS; i++) {
        total += (size_t)levels[i].width * (size_t)levels[i].height;
    }
    return total;
}

/* ------------------------------- letterbox ----------------------------- */

float kyx_letterbox(
    const uint8_t *src, int src_width, int src_height, kyx_pixfmt pixfmt,
    uint8_t *dst, int dst_width, int dst_height)
{
    int bpp;
    int blue;
    int red;
    double ratio;
    int scaled_width;
    int scaled_height;
    size_t plane;

    if (src == NULL || dst == NULL || src_width <= 0 || src_height <= 0 ||
        dst_width <= 0 || dst_height <= 0) {
        return 0.0f;
    }
    switch (pixfmt) {
    case KYX_PIXFMT_BGRA: bpp = 4; blue = 0; red = 2; break;
    case KYX_PIXFMT_BGR:  bpp = 3; blue = 0; red = 2; break;
    case KYX_PIXFMT_RGBA: bpp = 4; blue = 2; red = 0; break;
    case KYX_PIXFMT_RGB:  bpp = 3; blue = 2; red = 0; break;
    default: return 0.0f;
    }

    /* Double, and truncation, so the indices agree with the reference's
     * float64 arithmetic: a pixel row that lands on 12.9999 in one and
     * 13.0 in the other is a different source row. */
    ratio = (double)dst_height / (double)src_height;
    if ((double)dst_width / (double)src_width < ratio) {
        ratio = (double)dst_width / (double)src_width;
    }
    scaled_width = (int)((double)src_width * ratio);
    scaled_height = (int)((double)src_height * ratio);
    if (scaled_width > dst_width) {
        scaled_width = dst_width;
    }
    if (scaled_height > dst_height) {
        scaled_height = dst_height;
    }

    plane = (size_t)dst_width * (size_t)dst_height;
    memset(dst, KYX_FILL, plane * 3u);

    for (int y = 0; y < scaled_height; y++) {
        int sy = (int)((double)y / ratio);
        const uint8_t *row;
        uint8_t *out_b;
        uint8_t *out_g;
        uint8_t *out_r;

        if (sy > src_height - 1) {
            sy = src_height - 1;
        }
        row = src + (size_t)sy * (size_t)src_width * (size_t)bpp;
        out_b = dst + (size_t)y * (size_t)dst_width;
        out_g = out_b + plane;
        out_r = out_g + plane;
        for (int x = 0; x < scaled_width; x++) {
            int sx = (int)((double)x / ratio);
            const uint8_t *pixel;

            if (sx > src_width - 1) {
                sx = src_width - 1;
            }
            pixel = row + (size_t)sx * (size_t)bpp;
            out_b[x] = pixel[blue];
            out_g[x] = pixel[1];
            out_r[x] = pixel[red];
        }
    }
    return (float)ratio;
}

/* ------------------------------- decode -------------------------------- */

static bool emit(
    kyx_box *out, size_t capacity, size_t *count, size_t *dropped,
    int class_id, float score, float cx, float cy, float w, float h)
{
    kyx_box *box;

    /* A size logit past what exp() holds, or a NaN from a broken graph,
     * makes a box with no coordinates.  Passed on, the caller converts it
     * to int - undefined - or draws a frame-sized box round nothing.  Not
     * a detection, and not a candidate that failed to fit either. */
    if (!isfinite(cx) || !isfinite(cy) || !isfinite(w) || !isfinite(h)) {
        return false;
    }
    if (*count >= capacity) {
        if (dropped != NULL) {
            (*dropped)++;
        }
        return false;
    }
    box = &out[*count];
    box->class_id = class_id;
    box->score = score;
    box->x0 = cx - w * 0.5f;
    box->y0 = cy - h * 0.5f;
    box->x1 = cx + w * 0.5f;
    box->y1 = cy + h * 0.5f;
    (*count)++;
    return true;
}

size_t kyx_decode_raw(
    const kyx_raw_level raw[KYX_LEVELS], int input_width, int input_height,
    float conf, kyx_box *out, size_t capacity, size_t *dropped)
{
    kyx_level levels[KYX_LEVELS];
    size_t count = 0u;
    float gate;

    if (dropped != NULL) {
        *dropped = 0u;
    }
    if (raw == NULL || out == NULL ||
        !kyx_levels(input_width, input_height, levels)) {
        return 0u;
    }
    gate = kyx_logit(conf);

    for (int level = 0; level < KYX_LEVELS; level++) {
        const int w = levels[level].width;
        const int h = levels[level].height;
        const float stride = (float)levels[level].stride;
        const size_t area = (size_t)w * (size_t)h;
        const float *reg = raw[level].reg;
        const float *obj = raw[level].obj;
        const float *cls = raw[level].cls;

        if (reg == NULL || obj == NULL || cls == NULL) {
            continue;
        }
        for (size_t cell = 0u; cell < area; cell++) {
            float best;
            int best_class;
            float score;

            /* The gate: one compare, and most cells end here.  Written
             * as `not greater` rather than `less or equal` so a NaN, which
             * is neither, fails it too. */
            if (!(obj[cell] > gate)) {
                continue;
            }
            best = cls[cell];
            best_class = 0;
            for (int c = 1; c < KYX_CLASSES; c++) {
                float v = cls[(size_t)c * area + cell];
                if (v > best) {
                    best = v;
                    best_class = c;
                }
            }
            score = kyx_sigmoid(obj[cell]) * kyx_sigmoid(best);
            if (!(score > conf)) {
                continue;
            }
            {
                const float gx = (float)(cell % (size_t)w);
                const float gy = (float)(cell / (size_t)w);
                const float cx = (reg[cell] + gx) * stride;
                const float cy = (reg[area + cell] + gy) * stride;
                const float bw = expf(reg[2u * area + cell]) * stride;
                const float bh = expf(reg[3u * area + cell]) * stride;

                emit(out, capacity, &count, dropped, best_class, score,
                     cx, cy, bw, bh);
            }
        }
    }
    return count;
}

size_t kyx_decode_flat(
    const float *cells, size_t cell_count, int input_width, int input_height,
    float conf, kyx_box *out, size_t capacity, size_t *dropped)
{
    kyx_level levels[KYX_LEVELS];
    size_t count = 0u;
    size_t offset = 0u;

    if (dropped != NULL) {
        *dropped = 0u;
    }
    if (cells == NULL || out == NULL ||
        !kyx_levels(input_width, input_height, levels) ||
        cell_count != kyx_cells(input_width, input_height)) {
        return 0u;
    }

    for (int level = 0; level < KYX_LEVELS; level++) {
        const int w = levels[level].width;
        const size_t area = (size_t)w * (size_t)levels[level].height;
        const float stride = (float)levels[level].stride;

        for (size_t cell = 0u; cell < area; cell++) {
            const float *row = cells + (offset + cell) * KYX_CELL_VALUES;
            const float obj = row[4];
            float best;
            int best_class;
            float score;

            /* Already a probability here, and still a bound on the score. */
            if (!(obj > conf)) {
                continue;
            }
            best = row[5];
            best_class = 0;
            for (int c = 1; c < KYX_CLASSES; c++) {
                if (row[5 + c] > best) {
                    best = row[5 + c];
                    best_class = c;
                }
            }
            score = obj * best;
            if (!(score > conf)) {
                continue;
            }
            {
                const float gx = (float)(cell % (size_t)w);
                const float gy = (float)(cell / (size_t)w);
                const float cx = (row[0] + gx) * stride;
                const float cy = (row[1] + gy) * stride;
                const float bw = expf(row[2]) * stride;
                const float bh = expf(row[3]) * stride;

                emit(out, capacity, &count, dropped, best_class, score,
                     cx, cy, bw, bh);
            }
        }
        offset += area;
    }
    return count;
}

/* ------------------------------- nms ----------------------------------- */

/*
 * Insertion sort, best first, and not qsort(): glibc's qsort allocates a
 * buffer for anything past a kilobyte of elements - 42 boxes - and a
 * busy frame at the default threshold produces more than that, so the
 * promise of no allocation per frame would have been false exactly when
 * it mattered.  The candidate list is dozens long, where insertion is as
 * fast as anything and stable, so two boxes with one score keep the
 * order the head produced them in.
 */
static void sort_by_score(kyx_box *boxes, size_t count)
{
    for (size_t i = 1u; i < count; i++) {
        const kyx_box key = boxes[i];
        size_t j = i;

        while (j > 0u && boxes[j - 1u].score < key.score) {
            boxes[j] = boxes[j - 1u];
            j--;
        }
        boxes[j] = key;
    }
}

static float overlap(const kyx_box *a, const kyx_box *b)
{
    const float x0 = a->x0 > b->x0 ? a->x0 : b->x0;
    const float y0 = a->y0 > b->y0 ? a->y0 : b->y0;
    const float x1 = a->x1 < b->x1 ? a->x1 : b->x1;
    const float y1 = a->y1 < b->y1 ? a->y1 : b->y1;
    const float iw = x1 - x0 > 0.0f ? x1 - x0 : 0.0f;
    const float ih = y1 - y0 > 0.0f ? y1 - y0 : 0.0f;
    const float inter = iw * ih;
    const float area_a = (a->x1 - a->x0 > 0.0f ? a->x1 - a->x0 : 0.0f) *
                         (a->y1 - a->y0 > 0.0f ? a->y1 - a->y0 : 0.0f);
    const float area_b = (b->x1 - b->x0 > 0.0f ? b->x1 - b->x0 : 0.0f) *
                         (b->y1 - b->y0 > 0.0f ? b->y1 - b->y0 : 0.0f);

    return inter / (area_a + area_b - inter + 1e-9f);
}

size_t kyx_nms(kyx_box *boxes, size_t count, float iou, bool per_class)
{
    size_t kept = 0u;

    if (boxes == NULL || count == 0u) {
        return 0u;
    }
    /* A score that is not a number cannot be ordered or held to a
     * threshold: marked suppressed here, so it neither sorts nor survives.
     * (`not at least zero` rather than `negative`, because NaN is neither.) */
    for (size_t i = 0u; i < count; i++) {
        if (!(boxes[i].score >= 0.0f)) {
            boxes[i].score = -1.0f;
        }
    }
    sort_by_score(boxes, count);

    /* Suppression is recorded in the score itself - a negative score is
     * not a detection - so this needs no scratch memory, which is what
     * keeps it off the allocator on the per-frame path. */
    for (size_t i = 0u; i < count; i++) {
        if (boxes[i].score < 0.0f) {
            continue;
        }
        for (size_t j = i + 1u; j < count; j++) {
            if (boxes[j].score < 0.0f) {
                continue;
            }
            if (per_class && boxes[j].class_id != boxes[i].class_id) {
                continue;
            }
            if (overlap(&boxes[i], &boxes[j]) > iou) {
                boxes[j].score = -1.0f;
            }
        }
    }
    for (size_t i = 0u; i < count; i++) {
        if (boxes[i].score >= 0.0f) {
            if (kept != i) {
                boxes[kept] = boxes[i];
            }
            kept++;
        }
    }
    return kept;
}

/* ------------------------------- back to the source -------------------- */

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

void kyx_unletterbox(
    kyx_box *boxes, size_t count, float ratio, int src_width, int src_height)
{
    const float w = (float)src_width;
    const float h = (float)src_height;

    if (boxes == NULL || !(ratio > 0.0f)) {
        return;
    }
    for (size_t i = 0u; i < count; i++) {
        boxes[i].x0 = clampf(boxes[i].x0 / ratio, 0.0f, w);
        boxes[i].y0 = clampf(boxes[i].y0 / ratio, 0.0f, h);
        boxes[i].x1 = clampf(boxes[i].x1 / ratio, 0.0f, w);
        boxes[i].y1 = clampf(boxes[i].y1 / ratio, 0.0f, h);
    }
}

size_t kyx_reply(
    const kyx_box *boxes, size_t count, int src_width, int src_height,
    float reply[KYX_REPLY_VALUES])
{
    const float w = (float)src_width;
    const float h = (float)src_height;
    size_t rows;

    if (reply == NULL) {
        return 0u;
    }
    memset(reply, 0, sizeof(float) * KYX_REPLY_VALUES);
    if (boxes == NULL || src_width <= 0 || src_height <= 0) {
        return 0u;
    }
    rows = count < KYX_REPLY_ROWS ? count : KYX_REPLY_ROWS;
    for (size_t i = 0u; i < rows; i++) {
        float *row = reply + i * KYX_REPLY_COLUMNS;

        row[0] = (float)boxes[i].class_id;
        row[1] = boxes[i].score;
        row[2] = clampf(boxes[i].y0 / h, 0.0f, 1.0f);
        row[3] = clampf(boxes[i].x0 / w, 0.0f, 1.0f);
        row[4] = clampf(boxes[i].y1 / h, 0.0f, 1.0f);
        row[5] = clampf(boxes[i].x1 / w, 0.0f, 1.0f);
    }
    return rows;
}
