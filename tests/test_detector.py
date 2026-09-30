"""Model selection and the fixed-size subprocess contract, without weights."""
import contextlib
import importlib.machinery
import importlib.util
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / "tools/kilix-yolox-detect"
loader = importlib.machinery.SourceFileLoader("yolox_detector", str(TOOL))
spec = importlib.util.spec_from_loader(loader.name, loader)
detector = importlib.util.module_from_spec(spec)
loader.exec_module(detector)


class ModelSelection(unittest.TestCase):
    def test_named_environment_override_resolves_inside_runtime(self):
        with tempfile.TemporaryDirectory() as runtime:
            with patch.dict(os.environ, {"KILIX_YOLOX_DIR": runtime,
                                         "KILIX_YOLOX_MODEL": "yolox_nano"}):
                self.assertEqual(detector.resolve_model("yolox_s", 320, 320),
                                 str(Path(runtime) / "models/yolox_nano.onnx"))

    def test_override_uses_cut_matching_geometry(self):
        with tempfile.TemporaryDirectory() as runtime:
            cut = Path(runtime) / "models/yolox_nano_320.onnx"
            cut.parent.mkdir()
            cut.write_bytes(b"private selection fixture, not a model")
            with patch.dict(os.environ, {"KILIX_YOLOX_DIR": runtime,
                                         "KILIX_YOLOX_MODEL": "yolox_nano"}):
                self.assertEqual(detector.resolve_model("yolox_s", 320, 320), str(cut))
                self.assertEqual(detector.resolve_model("yolox_s", 640, 640),
                                 str(cut.parent / "yolox_nano.onnx"))

    def test_explicit_model_path_override_stays_explicit(self):
        path = "/private model directory/selected.onnx"
        with patch.dict(os.environ, {"KILIX_YOLOX_MODEL": path}):
            self.assertEqual(detector.resolve_model("yolox_s", 320, 320), path)


class _BinaryStream:
    def __init__(self, data=b""):
        self.buffer = io.BytesIO(data)


class SubprocessContract(unittest.TestCase):
    def run_pipe(self, frames, constructor):
        source, output, errors = _BinaryStream(frames), _BinaryStream(), io.StringIO()
        with patch.object(sys, "argv", [str(TOOL), "--geometry", "2x2"]), \
             patch.object(sys, "stdin", source), patch.object(sys, "stdout", output), \
             contextlib.redirect_stderr(errors), patch.object(detector, "Detector", constructor):
            status = detector.main()
        return status, output.buffer.getvalue(), errors.getvalue()

    def test_missing_runtime_preserves_one_reply_per_complete_frame(self):
        def unavailable(*_):
            raise ImportError("private unavailable-runtime fixture")

        status, payload, errors = self.run_pipe(b"\0" * (16 * 2 + 3), unavailable)
        self.assertEqual(status, 0)
        self.assertEqual(payload, bytes(480 * 2))
        self.assertIn("unavailable-runtime", errors)

    def test_reply_failure_stays_framed_and_later_frames_continue(self):
        class BrokenInference:
            class np:
                uint8 = object()

                @staticmethod
                def frombuffer(*_, **__):
                    raise ValueError("private inference failure fixture")

        status, payload, errors = self.run_pipe(b"\0" * 32, lambda *_: BrokenInference())
        self.assertEqual(status, 0)
        self.assertEqual(len(payload), 960)
        self.assertEqual(struct.unpack("<240f", payload), (0.0,) * 240)
        self.assertEqual(errors.count("private inference failure fixture"), 2)

    def test_detection_rows_are_binary_normalized_contract_rows(self):
        row = (16, 0.75, 0.1, 0.2, 0.6, 0.8)

        class SuccessfulInference:
            class np:
                uint8 = object()

                @staticmethod
                def frombuffer(*_, **__):
                    class Image:
                        def reshape(self, _):
                            return self
                    return Image()

            def __call__(self, _):
                return [row]

        status, payload, errors = self.run_pipe(b"\0" * 32,
                                               lambda *_: SuccessfulInference())
        self.assertEqual((status, len(payload), errors), (0, 960, ""))
        for start in (0, 480):
            values = struct.unpack("<120f", payload[start:start + 480])
            for actual, expected in zip(values[:6], row):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(values[6:], (0.0,) * 114)

    def test_empty_input_produces_no_reply(self):
        status, payload, _ = self.run_pipe(b"", lambda *_: None)
        self.assertEqual((status, payload), (0, b""))


if __name__ == "__main__":
    unittest.main()
