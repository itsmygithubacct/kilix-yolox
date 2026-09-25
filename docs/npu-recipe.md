# Compiling YOLOX for a small INT8 NPU

What was learned compiling these exports for a 1 TOPS INT8 accelerator
with a MLIR-based vendor toolchain, written down so the next port does not
rediscover it. The toolchain's own commands differ per vendor; the shape of
the problem does not.

## 1. Cut at the nine raw outputs, not at the head's Concat

The obvious cut point is the three per-level `Concat` outputs: one tensor
per level, sigmoids already applied, and it dodges the `Reshape` shape
constants that are baked for the export size. Models cut there compile
cleanly, run at full speed, and detect nothing.

The `Concat` puts bounded sigmoid outputs (0..1) and unbounded box
regressions under one quantisation threshold. The head's values are
overwhelmingly near zero, so KL calibration picks a small threshold and
clips the rare high values. The tell is exact: per level,
`reg max == obj max == cls max`, and at stride 16 objectness can never
exceed about 0.19, so no threshold above that ever fires.

Cut instead at the nine raw convolution outputs, `reg`, `obj` and `cls`
per level, before the sigmoids. `kilix-yolox-cut` does exactly this and
prints the tensor names. Each tensor then gets its own scale, and logits
quantise well because they are wide and well spread. Two things fall out:
the input size can be changed (the tail with its fixed shapes is gone),
and the decoder can gate on the objectness logit, one compare per cell,
before touching the eighty class values.

## 2. Bake YOLOX's preprocessing, not Ultralytics'

Mean `0,0,0`, scale `1,1,1`, pixel format BGR, letterbox with fill 114 and
the image placed top-left. Calibration must see what inference will see,
so the same letterbox goes into the calibration pass. A confident,
wrong result on a known image means one of these three was carried over
from a YOLO26 or YOLOv5 recipe.

## 3. Calibrate on images with a clean licence chain

One hundred COCO val2017 images, fetched individually from the COCO
distribution rather than from a copy packaged with an AGPL detector, KL
calibration. The list of filenames is small enough to ship beside the
recipe; the images are not.

## 4. Two deploy variants, and why both exist

- **Quantised tensor input.** The compiled model takes an int8 tensor with
  a quantisation scale; the caller supplies already quantised, already
  letterboxed bytes. This is the byte path: the caller owns preprocessing
  and `kyx_letterbox()` produces exactly the planar BGR it wants.
- **Fused preprocessing.** BGR ordering and quantisation are compiled into
  the model, so its input is a planar uint8 frame. This is what a zero-copy
  video path needs: handed a frame through the runtime's video-frame entry
  point, a quantised-input model copies raw pixels with no channel swap and
  no quantisation, and every objectness logit sits near −2. That is
  confident-looking noise, and it was found on hardware, not in a
  simulator.

Same weights, same graph, same calibration table; only the deploy step
differs. Measured on the target the fused variant matched the fp32
reference within quantisation noise on a known image.

## 5. What to measure before choosing a size

At 320×224 on the 1 TOPS part: `yolox_s` forward 18.4 ms and the most
accurate of every model tried; `yolox_nano` 3.8 ms and 1.1 MB, faster and
smaller than the Ultralytics nano; `yolox_tiny` in between on speed but
with a confident false positive on the test image. Numbers move with the
chip and the calibration; the ordering is the durable part. Run the
compiler's similarity check between the INT8 and fp32 outputs for the
chosen size, and recalibrate on frames from the cameras it will watch.
