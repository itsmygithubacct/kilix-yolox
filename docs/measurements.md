# Why YOLOX, and what was measured

Why Kilix carries a second, permissively licensed detector, and the recorded timings and detections. Moved from the README.

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
