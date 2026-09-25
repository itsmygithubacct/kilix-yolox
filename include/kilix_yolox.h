#ifndef KILIX_YOLOX_H
#define KILIX_YOLOX_H

/*
 * YOLOX, the parts that are arithmetic.
 *
 * The model is somebody else's problem: an ONNX file run by whatever
 * runtime the machine has, or a compiled graph on a small NPU.  What is
 * the same everywhere, and what is easy to get subtly wrong, is the work
 * on either side of it - putting an image into the model's frame and
 * turning its head tensors back into boxes.  That is what this library
 * is: C11, no runtime linked, no allocation on the per-frame path.
 *
 * YOLOX's conventions are not Ultralytics', and each one produces
 * plausible garbage rather than an error when it is carried over from
 * the other family:
 *
 *   input       raw 0-255 BGR, planar               (not RGB x 1/255)
 *   letterbox   fill 114, image at the TOP-LEFT     (not centred)
 *   boxes       centre and size in stride units,
 *               exp() on the size                   (not ltrb distances)
 *   head        one-to-many: NMS is required        (YOLO26 needs none)
 *
 * Two forms of head output are decoded.  The export form is one (N, 85)
 * tensor with objectness and class scores already through their sigmoids
 * and the boxes still in grid units.  The raw form is nine tensors - the
 * regression, objectness and class convolutions per stride, before any
 * activation - which is where a graph has to be cut to quantise it well
 * and to run it at a size other than the export's.  Both end in the same
 * boxes; the tests prove it on the same weights.
 *
 * The reply packer writes kilix-object-detect's 480-byte contract, so a
 * subprocess built on this is a drop-in detector for kilix-look.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KILIX_YOLOX_VERSION_MAJOR 0
#define KILIX_YOLOX_VERSION_MINOR 1
#define KILIX_YOLOX_VERSION_PATCH 0

/* COCO's eighty classes, in the order every YOLO family uses. */
#define KYX_CLASSES 80
/* Strides 8, 16 and 32: three levels, in that order. */
#define KYX_LEVELS 3
/* One cell of head output: 4 box values, objectness, 80 class values. */
#define KYX_CELL_VALUES (4 + 1 + KYX_CLASSES)
/* The letterbox padding value YOLOX trains with. */
#define KYX_FILL 114

/* kilix-object-detect's reply: float32[20][6], rows of
 * [class, score, y0, x0, y1, x1], coordinates normalised 0-1. */
#define KYX_REPLY_ROWS 20
#define KYX_REPLY_COLUMNS 6
#define KYX_REPLY_VALUES (KYX_REPLY_ROWS * KYX_REPLY_COLUMNS)
#define KYX_REPLY_BYTES (KYX_REPLY_VALUES * (int)sizeof(float))

/* Measured defaults: kilix-look's threshold, and YOLOX's own NMS overlap. */
#define KYX_CONF_DEFAULT 0.25f
#define KYX_IOU_DEFAULT 0.45f

typedef enum kyx_pixfmt {
    KYX_PIXFMT_BGRA = 0,   /* 4 bytes, byte order B,G,R,A */
    KYX_PIXFMT_BGR,        /* 3 bytes */
    KYX_PIXFMT_RGBA,       /* 4 bytes, byte order R,G,B,A */
    KYX_PIXFMT_RGB         /* 3 bytes */
} kyx_pixfmt;

/*
 * A detection.  Coordinates are pixels in whatever frame the box was last
 * expressed in: the model's input after decoding, the source image after
 * kyx_unletterbox().  Floats, because a box that is scaled twice and
 * rounded twice drifts, and this family has shipped that bug before.
 */
typedef struct kyx_box {
    int class_id;
    float score;
    float x0;
    float y0;
    float x1;
    float y1;
} kyx_box;

typedef struct kyx_level {
    int stride;
    int width;    /* cells across */
    int height;   /* cells down */
} kyx_level;

/* ------------------------------- geometry ------------------------------ */

/*
 * The three grids for an input size.  False when the size is not a
 * positive multiple of 32, which is the only size the head can produce:
 * 320x320 is 40x40, 20x20 and 10x10 cells, 2,100 in all.
 */
bool kyx_levels(int input_width, int input_height, kyx_level out[KYX_LEVELS]);

/* Cells across all levels, or 0 for an invalid size. */
size_t kyx_cells(int input_width, int input_height);

/* ------------------------------- letterbox ----------------------------- */

/*
 * Source pixels into the model's frame: scaled by the limiting ratio,
 * placed top-left, the rest filled with 114, written as BGR planar bytes
 * (all the blue, then all the green, then all the red) into `dst`, which
 * holds 3 * dst_width * dst_height bytes.
 *
 * Nearest neighbour, on purpose: it is what the reference does, it needs
 * no image library, and the cost of bilinear here is larger than its
 * benefit on a detector that was trained on resized COCO anyway.
 *
 * Returns the ratio the image was scaled by - keep it, kyx_unletterbox()
 * needs it - or 0 for bad arguments.
 */
float kyx_letterbox(
    const uint8_t *src, int src_width, int src_height, kyx_pixfmt pixfmt,
    uint8_t *dst, int dst_width, int dst_height);

/* ------------------------------- decode -------------------------------- */

/*
 * The nine raw head tensors, one triple per level in kyx_levels() order,
 * each planar: reg is (4, H, W), obj is (1, H, W), cls is (80, H, W).
 * Values are what the convolutions produced - no sigmoid anywhere.
 */
typedef struct kyx_raw_level {
    const float *reg;
    const float *obj;
    const float *cls;
} kyx_raw_level;

/*
 * Raw tensors into candidate boxes in input pixels, before NMS.
 *
 * Objectness is gated on its logit first: sigmoid is monotonic and
 * score = sig(obj) * sig(cls) can never exceed sig(obj), so one compare
 * per cell rejects nearly every cell before any of the eighty class
 * values are read, let alone exponentiated.
 *
 * Returns how many were written.  `dropped`, when not NULL, receives how
 * many candidates did not fit: silent truncation would read as "nothing
 * else was there".  A cell whose values are not finite - a NaN from a
 * broken graph, a size logit past what exp() holds - is not a detection
 * and is skipped, neither written nor counted as dropped.
 */
size_t kyx_decode_raw(
    const kyx_raw_level raw[KYX_LEVELS], int input_width, int input_height,
    float conf, kyx_box *out, size_t capacity, size_t *dropped);

/*
 * The export's (N, 85) tensor into candidate boxes in input pixels.
 * Rows are cells in level order, row-major within a level; columns are
 * [x, y, log w, log h, objectness, class * 80] with the last 81 already
 * through their sigmoids and the first four still in grid units.
 * `cell_count` must equal kyx_cells() for the input size.
 */
size_t kyx_decode_flat(
    const float *cells, size_t cell_count, int input_width, int input_height,
    float conf, kyx_box *out, size_t capacity, size_t *dropped);

/*
 * Greedy non-maximum suppression, in place.  Sorts by score, keeps the
 * best of every overlapping group, returns how many remain at the front
 * of the array in score order.  A box whose score is negative or NaN is
 * taken as already suppressed and does not come back.
 *
 * Class-agnostic by default.  Per-class is what YOLOX's own demo does,
 * and it leaves a `car` on top of every `truck` it is unsure about; one
 * object, one box, is what a caller drawing them wants.
 */
size_t kyx_nms(kyx_box *boxes, size_t count, float iou, bool per_class);

/*
 * Input pixels back to source pixels: undo the ratio and clamp.  There
 * is no offset to remove because the image sat top-left.  A ratio that
 * is not positive - zero, or NaN - leaves the boxes as they are.
 */
void kyx_unletterbox(
    kyx_box *boxes, size_t count, float ratio, int src_width, int src_height);

/*
 * Boxes, already in source pixels and score order, into the 480-byte
 * reply normalised by the source size.  Writes all 120 floats: rows past
 * the boxes are zero.  Returns how many rows carry a box, at most
 * KYX_REPLY_ROWS.
 */
size_t kyx_reply(
    const kyx_box *boxes, size_t count, int src_width, int src_height,
    float reply[KYX_REPLY_VALUES]);

/* ------------------------------- names --------------------------------- */

/* COCO label for a class id, or NULL outside 0-79. */
const char *kyx_label(int class_id);
/* Class id for a COCO label, or -1. */
int kyx_class_from_name(const char *name);

/* The two functions the gate is built on, public so a caller reasoning
 * about thresholds can use the same arithmetic. */
float kyx_sigmoid(float x);
float kyx_logit(float p);

#ifdef __cplusplus
}
#endif

#endif /* KILIX_YOLOX_H */
