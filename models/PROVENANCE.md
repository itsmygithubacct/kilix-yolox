# Where the weights come from

This repository ships no weights. `kilix-yolox-fetch` downloads them and
refuses any file whose checksum is not in `SHA256SUMS`; that list is the
artefact under version control.

## Source

Megvii-BaseDetection/YOLOX, release `0.1.1rc0`, published under the Apache
License 2.0 with the weights included. The ONNX exports are attached to the
release:

| File | Export input | Export output | Published mAP | bytes | sha256 |
| --- | --- | --- | --- | --- | --- |
| `yolox_s.onnx` | 1×3×640×640 | (1, 8400, 85) | 40.5 at 640 | 35858002 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| `yolox_tiny.onnx` | 1×3×416×416 | (1, 3549, 85) | 32.8 at 416 | 20219662 | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` |
| `yolox_nano.onnx` | 1×3×416×416 | (1, 3549, 85) | 25.8 at 416 | 3659407 | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |

The fetch tool writes `LICENSE.yolox` (the Apache-2.0 text from the same
release) and a `NOTICE` carrying the attribution and these checksums beside
the files it places, so the runtime directory carries its own provenance.

## What the export contains

Opset 11. The head, per level, ends in
`Concat([reg, Sigmoid(obj), Sigmoid(cls)])` followed by a `Reshape`,
`Concat` and `Transpose` whose shape constants are fixed to the export size.
The output rows are `[x, y, log w, log h, objectness, class × 80]` with the
first four still in grid units: decoding is the caller's job. Class ids are
COCO's eighty, in the order every YOLO family uses.

The nine raw convolution outputs that `kilix-yolox-cut` exposes are, by ONNX
tensor name:

| Model | stride 8 (reg, obj, cls) | stride 16 | stride 32 |
| --- | --- | --- | --- |
| `yolox_s`, `yolox_tiny` | 794, 795, 785 | 820, 821, 811 | 846, 847, 837 |
| `yolox_nano` | 1062, 1063, 1045 | 1104, 1105, 1087 | 1146, 1147, 1129 |

`kilix-yolox-cut` finds them from the graph rather than from this table, so
a differently numbered export still cuts in the right place.

## Preprocessing the weights expect

Raw 0-255, mean 0, scale 1, **BGR**, letterbox fill 114 with the image at
the **top-left**, NCHW. This is not the Ultralytics convention (RGB, ×1/255,
centred), and feeding one family the other's produces plausible garbage
rather than an error.
