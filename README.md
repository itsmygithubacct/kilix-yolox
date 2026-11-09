# kilix-yolox

The same detections, under a licence that lets you ship them.

**Status:** not part of the Plebian-OS image. Kilix builds a pinned commit of
this repository as its object-detector runtime when a user runs
`kilix install yolox` (or `kilix yolox install`), in
`~/.local/gpu_terminal/runtimes/yolox`, and points `KILIX_OBJECT_DETECTOR` at
it for `kilix look` and `kilix nvr`. The YOLOX weights are Kilix Content
assets (`yolox_nano`, `yolox_tiny`, `yolox_s`) that download only after the
user's typed licence agreement. The exception is the Plebian-OS 0.2.3 ISO: its
offline default model bundle carries `yolox_nano` and `yolox_s`, and first
boot installs them into the Kilix Content store.

A small C11 library holding the arithmetic on either side of a YOLOX
model - letterbox in, boxes out - and **`kilix-yolox-detect`**, a
subprocess that speaks [`kilix-look`](https://github.com/itsmygithubacct/kilix-object-detect#readme)'s
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

## What it does

- `libkilix-yolox`: letterbox, decode (flat or nine raw head outputs),
  class-agnostic NMS and the 480-byte kilix-look reply, in C11 with no
  allocation per frame and no runtime linked.
- `kilix-yolox-detect`: an ONNX Runtime subprocess that answers kilix-look's
  frame protocol, so the detector is swapped by one environment variable.
- `kilix-yolox-cut`: re-exports a model cut at the nine raw outputs for a
  320x320 square (the cut that also quantises for small NPUs).
- `kilix-yolox-fetch`: downloads a YOLOX release export and refuses any file
  not listed in `models/SHA256SUMS`.

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

## How it works

```
kilix-look / kilix-nvr
   |  square BGRA frame on stdin          (KILIX_OBJECT_DETECTOR)
   v
kilix-yolox-detect   letterbox -> ONNX Runtime on yolox_*_320.onnx
   |                 -> decode -> NMS -> 480-byte reply   (numpy)
   v
kilix-look draws and reports the boxes
```

The detector reads fixed-size frames and writes fixed-size replies, so the
caller never links a model runtime. The subprocess does its letterbox, decode
and suppression in numpy. `libkilix-yolox` is the same arithmetic in C11 for a
detector that cannot run Python, such as a compiled graph on a small NPU:
YOLOX's own conventions (raw BGR, fill 114 at the top-left, centre-and-size
boxes in stride units, a head that needs NMS) are each one tested C function;
see [docs/library.md](docs/library.md).
Feeding the model the square it was cut for matters more than model size; the
numbers are in [docs/measurements.md](docs/measurements.md).

### What it reuses

Crops, motion gating, the pipe reader, the bundled-tool lookup and the
drawing all stay in `kilix-object-detect`. This module adds a model, not a
second copy of the pipeline.

## Dependencies

C11 and POSIX for the library and command. The subprocess wants Python 3
with `onnxruntime` and `numpy`; `kilix-yolox-cut` and the fixture generator
want `onnx` as well. Nothing here links a runtime.

## Documentation

- [docs/library.md](docs/library.md): the C API and YOLOX's conventions.
- [docs/measurements.md](docs/measurements.md): why a second detector, and the
  recorded timings and detections per model.
- [docs/npu-recipe.md](docs/npu-recipe.md): compiling the exports for a small
  INT8 NPU.
- [models/PROVENANCE.md](models/PROVENANCE.md): where the weights come from and
  their checksums.

## Related repositories

- [kilix-object-detect](https://github.com/itsmygithubacct/kilix-object-detect#readme):
  `kilix-look`, the pipeline whose detector slot this fills.
- [kilix-nvr](https://github.com/itsmygithubacct/kilix-nvr#readme): the camera
  recorder that runs the same detector.
- [kilix-motion-detect](https://github.com/itsmygithubacct/kilix-motion-detect#readme):
  the motion gate in front of detection.
- [kilix](https://github.com/itsmygithubacct/kilix#readme): `kilix install
  yolox` and `kilix yolox`.
- [kilix-content](https://github.com/itsmygithubacct/kilix-content#readme): the
  catalog records and licence agreement for the YOLOX weights.

## Part of Plebian-OS

kilix-yolox is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
Kilix builds a pinned commit of it on demand, as the detector runtime behind
`kilix look` and `kilix nvr`. The map of every repository and how it ships is
on the [components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## License

MIT for the code in this repository. See `LICENSE`. YOLOX and its weights
are Apache-2.0 and are not part of this repository; the fetch step places
the Apache-2.0 licence and notice beside the weights it downloads.
