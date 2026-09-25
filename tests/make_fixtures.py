#!/usr/bin/env python3
"""Generate the decoder fixtures the C tests run against.

The fixtures are the contract between the model and the arithmetic: a few
cells of raw head output plus the boxes they must decode to.  They are kept
sparse - only cells whose objectness clears a low floor, plus a handful that
do not - so a fixture is kilobytes and the tests need no model, no runtime and
no image.  Absent cells are filled by the reader with a logit that can never
pass the gate (raw form) or a zero score (flat form).

Needs onnx, onnxruntime and numpy, an image decoded to BGRA by ffmpeg, and
the Megvii ONNX exports.  Run once when the model or the decode changes:

  make_fixtures.py --model ~/.local/gpu_terminal/runtimes/yolox/models/yolox_s.onnx \
                   --image dog.jpg --out tests/fixtures

Writes:
  letterbox.txt   a tiny synthetic BGRA source and its expected planar output
  flat_640.txt    the export's (1,8400,85) form at native size, survivors only
  raw_320.txt     the nine raw head tensors at 320x320, survivors only
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

STRIDES = (8, 16, 32)
FILL = 114
FLOOR = 0.02          # objectness below this is not worth a fixture line
RAW_ABSENT = -30.0    # sigmoid(-30) is 1e-13: an absent cell never fires


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(p):
    return float(np.log(p / (1.0 - p)))


def decode_bgra(path):
    """ffmpeg is already a runtime dependency of the family; PIL is not."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True).stdout.strip()
    w, h = (int(v) for v in probe.split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
         "-pix_fmt", "bgra", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))


def letterbox(bgra, size):
    """YOLOX's own preprocessing: limiting ratio, top-left, fill 114,
    nearest neighbour so the C side can match it exactly."""
    tw, th = size
    ih, iw = bgra.shape[:2]
    r = min(th / ih, tw / iw)
    nh, nw = int(ih * r), int(iw * r)
    canvas = np.full((th, tw, 3), FILL, dtype=np.uint8)
    yi = (np.arange(nh) / r).astype(np.int32).clip(0, ih - 1)
    xi = (np.arange(nw) / r).astype(np.int32).clip(0, iw - 1)
    canvas[:nh, :nw] = bgra[yi][:, xi][:, :, :3]
    return canvas.transpose(2, 0, 1).copy(), r


def levels(size):
    tw, th = size
    return [(s, th // s, tw // s) for s in STRIDES]


def decode_cells(cells, size, conf, raw):
    """One decode for both forms.  `cells` is (N, 85) in level order, rows
    major inside a level.  Raw form holds logits; flat form holds the export's
    regressions plus already-sigmoided objectness and class scores."""
    boxes, scores, cids = [], [], []
    offset = 0
    for stride, h, w in levels(size):
        n = h * w
        block = cells[offset:offset + n]
        offset += n
        gy, gx = np.divmod(np.arange(n), w)
        obj = block[:, 4]
        cls = block[:, 5:]
        if raw:
            gate = obj > logit(conf)
            score = sigmoid(obj) * sigmoid(cls.max(1))
        else:
            gate = obj > conf
            score = obj * cls.max(1)
        keep = gate & (score > conf)
        if not keep.any():
            continue
        reg = block[keep, :4]
        cx = (reg[:, 0] + gx[keep]) * stride
        cy = (reg[:, 1] + gy[keep]) * stride
        bw = np.exp(reg[:, 2]) * stride
        bh = np.exp(reg[:, 3]) * stride
        boxes.append(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1))
        scores.append(score[keep])
        cids.append(cls[keep].argmax(1))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(cids)


def nms(boxes, scores, cids, thr, per_class):
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes.T
    area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        drop = iou > thr
        if per_class:
            drop &= cids[rest] == cids[i]
        order = rest[~drop]
    return keep


def raw_head_names(model):
    """The nine raw conv outputs: the input of each head Sigmoid and the
    regression tensor beside them, found from the per-level Concat."""
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    names = []
    for n in g.node:
        if n.op_type == "Concat" and len(n.input) == 3:
            reg, obj_s, cls_s = n.input
            if producer[obj_s].op_type != "Sigmoid":
                continue
            names.append((reg, producer[obj_s].input[0], producer[cls_s].input[0]))
    if len(names) != 3:
        raise SystemExit("expected three head concats, found %d" % len(names))
    return names


def cut_model(src, size, out_path):
    """The export's Reshape/Concat/Transpose tail carries shape constants for
    the export size, so a smaller input needs the graph cut before it.  The
    cut is done by hand rather than with onnx.utils.extract_model, because
    that runs shape inference over the whole graph first and the tail's fixed
    shapes contradict the new input size before anything is extracted."""
    import onnx
    m = onnx.load(src)
    names = raw_head_names(m)
    wanted = {x for level in names for x in level}
    producer = {o: n for n in m.graph.node for o in n.output}
    keep, todo = [], list(wanted)
    seen = set()
    while todo:
        name = todo.pop()
        node = producer.get(name)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        keep.append(node)
        todo.extend(node.input)
    nodes = [n for n in m.graph.node if id(n) in seen]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    del m.graph.output[:]
    for level in names:
        for x in level:
            m.graph.output.append(onnx.helper.make_tensor_value_info(
                x, onnx.TensorProto.FLOAT, ["N", "C", "H", "W"]))
    del m.graph.value_info[:]
    dims = m.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_value, dims[3].dim_value = size[1], size[0]
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return names


def run(session, chw):
    name = session.get_inputs()[0].name
    return session.run(None, {name: chw[None].astype(np.float32)})




def write_cells(path, form, size, conf, iou, cells, per_level_hw, keep_mask,
                expected):
    with open(path, "w") as f:
        f.write("kilix-yolox-fixture 1\n")
        f.write("form %s\n" % form)
        f.write("input %d %d\n" % size)
        f.write("conf %.4f\n" % conf)
        f.write("iou %.4f\n" % iou)
        f.write("cells %d\n" % int(keep_mask.sum()))
        offset = 0
        for level, (stride, h, w) in enumerate(per_level_hw):
            n = h * w
            for i in np.nonzero(keep_mask[offset:offset + n])[0]:
                gy, gx = divmod(int(i), w)
                vals = " ".join("%.6g" % v for v in cells[offset + i])
                f.write("%d %d %d %s\n" % (level, gy, gx, vals))
            offset += n
        f.write("boxes %d\n" % len(expected))
        for cid, score, box in expected:
            f.write("%d %.6f %.4f %.4f %.4f %.4f\n" % (cid, score, *box))


def fixture_from_cells(path, form, size, conf, iou, cells, seed):
    raw = form == "raw"
    obj = sigmoid(cells[:, 4]) if raw else cells[:, 4]
    keep = obj > FLOOR
    rng = np.random.default_rng(seed)
    # A few cells that must NOT fire, so the gate is exercised too.
    extra = rng.choice(np.nonzero(~keep)[0], size=8, replace=False)
    keep[extra] = True
    boxes, scores, cids = decode_cells(cells, size, conf, raw)
    kept = nms(boxes, scores, cids, iou, per_class=False)
    expected = [(int(cids[i]), float(scores[i]), boxes[i].tolist()) for i in kept]
    expected.sort(key=lambda e: -e[1])
    write_cells(path, form, size, conf, iou, cells, levels(size), keep, expected)
    return expected


def letterbox_fixture(path, seed):
    rng = np.random.default_rng(seed)
    sw, sh, tw, th = 13, 9, 16, 8
    src = rng.integers(0, 256, size=(sh, sw, 4), dtype=np.uint8)
    dst, r = letterbox(src, (tw, th))
    with open(path, "w") as f:
        f.write("kilix-yolox-letterbox 1\n")
        f.write("source %d %d bgra\n" % (sw, sh))
        f.write("target %d %d\n" % (tw, th))
        f.write("ratio %.9f\n" % r)
        f.write("src %s\n" % " ".join(str(int(v)) for v in src.reshape(-1)))
        f.write("dst %s\n" % " ".join(str(int(v)) for v in dst.reshape(-1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="tests/fixtures")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--small", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    os.makedirs(args.out, exist_ok=True)
    letterbox_fixture(os.path.join(args.out, "letterbox.txt"), args.seed)

    bgra = decode_bgra(args.image)
    model = onnx.load(args.model)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    native = (dims[3].dim_value, dims[2].dim_value)

    # Flat form at native size, straight from the export.
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    chw, _ = letterbox(bgra, native)
    flat = run(sess, chw)[0][0]
    flat_expected = fixture_from_cells(
        os.path.join(args.out, "flat_%d.txt" % native[0]), "flat", native,
        args.conf, args.iou, flat, args.seed)

    # Raw form at native size too, to prove the two decodes agree on the same
    # weights before either is trusted at a smaller size.
    with tempfile.TemporaryDirectory() as tmp:
        cut_native = os.path.join(tmp, "native.onnx")
        names = cut_model(args.model, native, cut_native)
        sess_n = ort.InferenceSession(cut_native, providers=["CPUExecutionProvider"])
        raw_native = assemble(sess_n, run(sess_n, chw), names, native)
        b, s, c = decode_cells(raw_native, native, args.conf, raw=True)
        k = nms(b, s, c, args.iou, False)
        got = sorted(((int(c[i]), round(float(s[i]), 3)) for i in k), key=lambda e: -e[1])
        want = [(cid, round(sc, 3)) for cid, sc, _ in flat_expected]
        if got != want:
            raise SystemExit("raw and flat decodes disagree at native size:\n  raw  %s\n  flat %s" % (got, want))
        print("native raw == flat:", want)

        small = (args.small, args.small)
        cut_small = os.path.join(tmp, "small.onnx")
        names = cut_model(args.model, small, cut_small)
        sess_s = ort.InferenceSession(cut_small, providers=["CPUExecutionProvider"])
        chw_s, _ = letterbox(bgra, small)
        raw_small = assemble(sess_s, run(sess_s, chw_s), names, small)
        raw_expected = fixture_from_cells(
            os.path.join(args.out, "raw_%d.txt" % small[0]), "raw", small,
            args.conf, args.iou, raw_small, args.seed)
        print("raw %dx%d:" % small, [(cid, round(sc, 3)) for cid, sc, _ in raw_expected])
    print("flat %dx%d:" % native, [(cid, round(sc, 3)) for cid, sc, _ in flat_expected])


def assemble(session, outputs, names, size):
    """Nine raw tensors into (N, 85) rows of logits, level order, row major."""
    by_name = {o.name: v for o, v in zip(session.get_outputs(), outputs)}
    rows = []
    for (reg, obj, cls), (stride, h, w) in zip(names, levels(size)):
        r = by_name[reg][0].reshape(4, -1).T
        o = by_name[obj][0].reshape(1, -1).T
        c = by_name[cls][0].reshape(80, -1).T
        assert r.shape[0] == h * w, (reg, r.shape, h, w)
        rows.append(np.concatenate([r, o, c], 1))
    return np.concatenate(rows)


if __name__ == "__main__":
    sys.exit(main())
