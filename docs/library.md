# The library

The C API for letterboxing, decoding, suppression and the kilix-look reply. Moved from the README.

```c
ratio = kyx_letterbox(bgra, w, h, KYX_PIXFMT_BGRA, planar, 320, 320);
/* ... the model ... */
n = kyx_decode_raw(raw, 320, 320, 0.25f, boxes, capacity, &dropped);
n = kyx_nms(boxes, n, 0.45f, false);
kyx_unletterbox(boxes, n, ratio, w, h);
kyx_reply(boxes, n, w, h, reply);        /* 480 bytes, kilix-look's rows */
```

YOLOX's conventions are not Ultralytics', and each produces plausible
garbage rather than an error when carried over from the other family:
raw 0-255 **BGR**, letterbox fill 114 with the image at the **top-left**,
boxes as centre and size in stride units with an exponent on the size, and
a one-to-many head that needs NMS. Each is one function with one test.

Two head forms decode to the same boxes. `kyx_decode_flat()` takes the
export's `(N, 85)` tensor. `kyx_decode_raw()` takes the nine raw
convolution outputs - regression, objectness and class per stride - which
is where a graph must be cut to quantise well or to run at a size other
than the export's. The raw path gates on the objectness logit, one compare
per cell, and reads the eighty class values only for the survivors; a test
proves the gate rejects exactly what the full computation would.

Suppression is class-agnostic by default. YOLOX's own demo suppresses
within a class, and that leaves a `car` on top of every `truck` it is
unsure of: one object, one box, is what a caller drawing them wants.

No allocation on the per-frame path, no runtime linked, `-lm` only.
