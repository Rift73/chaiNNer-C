"""Bounded CPU-only contract tests; no build, compiler, GPU or user checkpoint."""

import ast
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .export import load_arch, make_model
from .graph import Graph, phase_weight
from .spec import FACTORIES, ROOT, SOURCE, SOURCE_SHA256, from_model, sha256


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.guard = patch(
            "torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")
        )
        cls.guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop()

    def test_python_syntax(self):
        for path in (ROOT / "scripts/dual_tensorrt").glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_all_factories_scales(self):
        expected = {
            "dual_light": (128, 4, True),
            "dual_xs": (128, 4, False),
            "dual_s": (160, 6, False),
            "dual_m": (192, 6, False),
            "dual_l": (224, 10, False),
            "dual_xl": (256, 12, False),
        }
        for factory in FACTORIES:
            for scale in (1, 2, 4):
                with self.subTest(factory=factory, scale=scale):
                    _, model = make_model(factory, scale, meta=True)
                    spec = from_model(model, factory, scale, 512, 768)
                    c, groups, pu = expected[factory]
                    self.assertEqual(
                        (spec.channels, len(spec.depths), spec.unshuffle),
                        (c, groups, pu),
                    )
                    self.assertEqual(spec.internal_scale, scale * (2 if pu else 1))
                    self.assertEqual(spec.trunk, (256, 384) if pu else (512, 768))
                    self.assertEqual(len(model.dynamic_conv_names), groups)
                    self.assertEqual(model.xg is not None, spec.internal_scale >= 4)
                    self.assertEqual(model.xr.hr is not None, spec.internal_scale >= 2)
                    del model
        self.assertEqual(sha256(SOURCE), SOURCE_SHA256)

    def test_invalid_static_shapes(self):
        _, model = make_model("dual_light", 1, meta=True)
        for h, w in ((0, 4), (3, 4), (5, 8), (8, 7)):
            with self.assertRaises(ValueError):
                from_model(model, "dual_light", 1, h, w)

    def test_polyphase_matches_explicit_convolution(self):
        generator = torch.Generator().manual_seed(1024)
        for scale, grandchildren in ((2, False), (4, True)):
            x = torch.randn(1, 2 * scale**2, 3, 5, generator=generator)
            weight = torch.randn(3, 2, 3, 3, generator=generator)
            if grandchildren:
                unpacked = load_arch().unpack_grandchildren(x)
            else:
                unpacked = torch.nn.functional.pixel_shuffle(x, scale)
            expected = torch.nn.functional.conv2d(unpacked, weight, padding=1)
            actual = torch.nn.functional.pixel_shuffle(
                torch.nn.functional.conv2d(
                    x,
                    torch.from_numpy(
                        phase_weight(weight.numpy(), scale, grandchildren)
                    ),
                    padding=1,
                ),
                scale,
            )
            torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)

    def test_build_rejects_modified_external_weights(self):
        from .build import read_verified

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin").write_bytes(b"wrong")
            (root / "export.json").write_text(
                json.dumps(
                    {"source_sha256": SOURCE_SHA256, "files": {"weights.bin": "0" * 64}}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_verified(root, "export.json")

    def test_lowering_six_scale_regimes(self):
        self.check_lowerings(
            [
                (factory, scale)
                for factory in ("dual_xs", "dual_light")
                for scale in (1, 2, 4)
            ]
        )

    def test_lowering_wider_factories(self):
        self.check_lowerings(
            [(factory, 4) for factory in ("dual_s", "dual_m", "dual_l", "dual_xl")]
        )

    def check_lowerings(self, cases):
        from .cpu_graph_oracle import FP32GraphOracle
        import onnx

        torch.manual_seed(1024)
        for factory, scale in cases:
            with self.subTest(factory=factory, scale=scale):
                arch, model = make_model(factory, scale)
                with torch.no_grad():
                    for name in model.dynamic_conv_names:
                        model.get_submodule(name).experts.normal_(std=0.005)
                    model.xr.project.weight.normal_(std=0.01)
                    if model.xr.hr is not None:
                        model.xr.hr.weight.normal_(std=0.01)
                    if model.xg is not None:
                        model.xg.out.direct.reduce.weight.normal_(std=0.01)
                    model.prepare_for_export()
                    x = torch.randn(1, 3, 8, 12)
                    spec = from_model(model, factory, scale, 8, 12)
                    expected = model(x)
                    actual = FP32GraphOracle(arch, spec, x).build(model)
                    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
                    self.assertEqual(tuple(actual.shape), (1, 3, 8 * scale, 12 * scale))
                    graph = Graph(arch, spec).build(model)
                    onnx.checker.check_model(graph)
                    self.assertEqual(
                        sum(
                            n.op_type == "Conv"
                            and n.input[1]
                            not in {i.name for i in graph.graph.initializer}
                            for n in graph.graph.node
                        ),
                        len(model.groups),
                    )
                    del graph, model
                    gc.collect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
