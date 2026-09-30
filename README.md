# kilix-yolox

The same detections, under a licence that lets you ship them.

A small C11 library holding the arithmetic on either side of a YOLOX
model - letterbox in, boxes out - and **`kilix-yolox-detect`**, a
subprocess that speaks [`kilix-look`](https://github.com/itsmygithubacct/kilix-object-detect)'s
contract over ONNX Runtime, so switching the object detector to YOLOX is
one environment variable on the caller's side.

```sh
make
make test

kilix install yolox                 # runtime, explicit model agreement, 320 cut
kilix yolox check
kilix look image photo.jpg
```

No weights are part of this repository. Kilix owns runtime installation and
uses its pinned Content catalog and licence authority to acquire them.
The measurements below are recorded measurements, not results of installation.

## Why a second detector

`kilix-look` does not care which model answers it. It writes a square of
BGRA to a subprocess and reads back a fixed 480-byte reply. The reference
subprocess it ships runs Ultralytics YOLO, which is AGPL-3.0 including its
weights. That is fine on a desk and a problem in anything redistributed.

YOLOX (Megvii, Apache-2.0, weights included) is the permissively licensed
detector of the same family. Measured on an INT8 NPU against the
Ultralytics sizes, the `s` model was the most accurate of everything tried
and the `nano` model the fastest and smallest, so the licence costs nothing
at either end of the trade.

| Model | Published input | Published mAP | ONNX size |
| --- | --- | --- | --- |
| `yolox_nano` | 416 | 25.8 | 3.7 MB |
| `yolox_tiny` | 416 | 32.8 | 20 MB |
| `yolox_s` | 640 | 40.5 | 36 MB |

Figures from Megvii's release `0.1.1rc0`. `tiny` produced a confident false
positive in measurement and is listed for completeness, not recommended.

## What is measured

A 320×320 BGRA frame through the subprocess, fifty times, on a six-core
laptop CPU with ONNX Runtime's default threads; the first reply is the
warm-up. The frame is the test image letterboxed into the square by the
subprocess's own rule - top-left, fill 114, nearest neighbour - which is
not the square kilix-look cuts (centred, black, area-averaged), so
`kilix-look image` reports the same image a little differently: with
`yolox_s_320`, bicycle 0.92, dog 0.88, car 0.80. The `_320` rows are the
export cut by `kilix-yolox-cut` to run at kilix-look's own square; the
others are the export at its native size, fed the same square upscaled.

| Model | Warm-up | Mean | p95 | On the test image |
| --- | --- | --- | --- | --- |
| `yolox_nano_320` | 0.17 s | 8.1 ms | 8.7 ms | dog 0.86, bicycle 0.74, car 0.72 |
| `yolox_tiny_320` | 0.21 s | 16.0 ms | 18.6 ms | bicycle 0.80, truck 0.66, dog 0.54 |
| `yolox_s_320` | 0.25 s | 26.5 ms | 29.3 ms | dog 0.89, bicycle 0.88, car 0.59 |
| `yolox_nano` (416) | 0.23 s | 11.2 ms | 13.6 ms | dog 0.85, bicycle 0.78, car 0.71 |
| `yolox_tiny` (416) | 0.23 s | 24.2 ms | 31.0 ms | truck 0.87, bicycle 0.85, dog 0.63 |
| `yolox_s` (640) | 0.35 s | 93.2 ms | 109.2 ms | truck 0.87, bicycle 0.78, dog 0.52 |

Two things the table says. **Warm-up is a third of a second**, not the
tens of seconds a framework import costs; kilix-look's 90-second
allowance for a cold detector is never touched. And **feed the model the
square it was cut for**: the 640 export handed a 320 square sees a 2x
nearest-neighbour upscale and scores the dog at 0.52 where the same
weights cut to 320 score it at 0.89. So a bare model name resolves to
`NAME_320.onnx` when the frame is 320 square and the cut exists. Running
`kilix-look --size 640` against the export is the other way round the
same fact, and the most accurate: bicycle 0.97, dog 0.92, truck 0.68.

**Decode and suppression in C cost 9 µs** per frame on the same machine
against 320 µs in numpy, on the same tensors. That is the number a small
NPU port needs, where decode on one slow core was a third of the frame
budget.

## The library

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

## Commands

```sh
kilix-yolox decode FIXTURE [--per-class] [--check]   # boxes from head output
kilix-yolox bench FIXTURE [ITERATIONS]               # time decode and NMS
kilix-yolox classes
kilix-yolox --selftest
```

The fixtures under `tests/fixtures/` are a few cells of real head output
and the boxes they must become, generated once from the weights by
`tests/make_fixtures.py`; the raw and flat forms were produced from the
same weights and agree at native size before either was trusted. One
fixture is compiled into the binary, so `--selftest` runs anywhere the
binary does. The tests need no model.

```sh
kilix-yolox-detect --geometry 320x320 --model yolox_s [--conf 0.25] [--nms 0.45]
                   [--per-class] [--input-size N] [--threads N] [--verbose]
kilix-yolox-fetch [yolox_s|yolox_tiny|yolox_nano ...] [--from FILE]
kilix-yolox-cut MODEL.onnx --size 320
```

`kilix-yolox-fetch` downloads a model from Megvii's release into
`~/.local/gpu_terminal/runtimes/yolox/models/`, refuses any file whose
sha256 is not in `models/SHA256SUMS`, and writes the Apache-2.0 licence
and a `NOTICE` beside it. `--from` verifies and copies a file already to
hand. `kilix-yolox-cut` writes `MODEL_320.onnx` cut at the nine raw
outputs; `docs/npu-recipe.md` says why that is the only cut that
quantises.

## Getting a runtime

```sh
kilix models show yolox_s
kilix models install yolox_s         # terminal confirmation and typed agreement
kilix install yolox
KILIX_YOLOX_MODEL=yolox_nano kilix install yolox
```

Kilix creates the virtualenv, installs ONNX Runtime, numpy and onnx, verifies
the existing Content agreement and manifest, cuts the model, and writes the
wrapper and `KILIX_OBJECT_DETECTOR` setting consumed by look and NVR. Runtime
installation never accepts a model licence on the user's behalf; `--yes`
only skips the runtime confirmation. Reinstalling with another supported model
refreshes the wrapper. A bare `KILIX_YOLOX_MODEL` name resolves inside
`KILIX_YOLOX_DIR/models` and prefers the cut matching the frame geometry;
an explicit `.onnx` path remains an explicit path.

`kilix yolox check` checks the agreement and installed byte binding without
setup. An older runtime without that binding needs `kilix install yolox` once.
These checks do not perform inference or certify model accuracy. The standalone
fetch tool is a low-level checksum utility; it does not create the Content
agreement needed by the integrated installer.

## What it reuses

Crops, motion gating, the pipe reader, the bundled-tool lookup and the
drawing all stay in `kilix-object-detect`. This module adds a model, not a
second copy of the pipeline.

## Dependencies

C11 and POSIX for the library and command. The subprocess wants Python 3
with `onnxruntime` and `numpy`; `kilix-yolox-cut` and the fixture generator
want `onnx` as well. Nothing here links a runtime.

## License

MIT for the code in this repository. See `LICENSE`. YOLOX and its weights
are Apache-2.0 and are not part of this repository; the fetch step places
the Apache-2.0 licence and notice beside the weights it downloads.
