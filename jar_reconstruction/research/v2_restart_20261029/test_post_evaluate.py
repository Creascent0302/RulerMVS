"""评测队列的CPU协议测试：不启动GPU、不加载训练标签。"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.v2_restart_20261029 import post_evaluate as post


class PostEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.templates = self.root / "templates"
        self.templates.mkdir()
        manifest = self.root / "frames.jsonl"
        manifest.write_text('{"sequence":"test","frame_id":1}\n')
        domain = {"name": "public_domain", "num_views": 4, "reference_index": 2,
                  "near": 2., "far": 80., "depth_scale": 1., "geometry_units": "meters",
                  "manifests": {split: str(manifest) for split in ("train", "val", "test")}}
        baseline = {"kind": "mvsanywhere", "weights": str(self.root / "baseline.ckpt"),
                    "refinements": 1, "shared_train_range": True, "drop_missing_views": True}
        common = {"image_size": [256, 448], "matching_image_size": [512, 896],
                  "datasets": [domain], "methods": {"mvsanywhere_refined": baseline}}
        for group in ("public", "carla", "carla_other_refs"):
            config = copy.deepcopy(common)
            if group != "public":
                config["datasets"][0]["name"] = "CARLA_new_upper" if group == "carla" else "CARLA_new_left"
            post.write_json(self.templates / (group + ".json"), config)
        post.write_json(self.root / "public.json", common)
        post.write_json(self.root / "carla.json", common)
        self.settings = {"output": str(self.root / "results"), "training_root": str(self.root / "training"),
                         "public_training_config": str(self.root / "public.json"),
                         "carla_training_config": str(self.root / "carla.json"),
                         "gpus": [0, 1], "timing_gpu": 0, "seed": 42, "run_speed": False}

    def test_prepare_preserves_distinct_checkpoints_and_manifest_bytes(self):
        plan = post.prepare(self.settings, self.templates)
        again = post.prepare(self.settings, self.templates)
        self.assertEqual(plan, again)
        self.assertEqual(set(plan["groups"]), {"public", "carla", "carla_other_refs"})
        public = post.read(plan["groups"]["public"]["config"])
        self.assertEqual(len(public["methods"]), 4)
        self.assertNotEqual(public["methods"]["v2_restart_public_best"]["weights"],
                            public["methods"]["v2_restart_public_final"]["weights"])
        self.assertEqual(Path(public["datasets"][0]["manifests"]["test"]).read_bytes(),
                         (self.root / "frames.jsonl").read_bytes())
        self.assertFalse(Path(public["methods"]["v2_restart_public_best"]["weights"]).exists())
        carla = post.read(plan["groups"]["carla"]["config"])
        self.assertIn("v2_previous_carla", carla["methods"])
        self.assertIn("v2_restart_public_best", carla["methods"])

    def test_changed_range_and_protocol_cannot_silently_reuse_output(self):
        post.prepare(self.settings, self.templates)
        changed = dict(self.settings, gpus=[0])
        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            post.prepare(changed, self.templates)
        config = post.read(self.root / "public.json")
        config["datasets"][0]["far"] = 100.
        post.write_json(self.root / "public.json", config)
        with self.assertRaisesRegex(ValueError, "far改变"):
            post.prepare(self.settings, self.templates)

    def test_test_is_blocked_until_both_training_phases_complete(self):
        plan = post.prepare(self.settings, self.templates)
        for stage in ("public", "carla_new"):
            folder = Path(self.settings["training_root"]) / stage
            folder.mkdir(parents=True)
            (folder / "events.jsonl").write_text('{"event":"start"}\n')
        with self.assertRaisesRegex(RuntimeError, "尚未结束"):
            post.validate_ready(plan)
        for stage in ("public", "carla_new"):
            (Path(self.settings["training_root"]) / stage / "events.jsonl").write_text('{"event":"complete"}\n')
        weight = self.root / "tiny.pt"
        weight.write_bytes(b"fixed weights")
        plan["required_weights"] = [str(weight)]
        with patch.object(post, "PROTECTED", {weight: post.digest(weight)}):
            self.assertEqual(post.validate_ready(plan), {str(weight): post.digest(weight)})
        with patch.object(post, "PROTECTED", {weight: "wrong"}):
            with self.assertRaisesRegex(ValueError, "保护对照权重改变"):
                post.validate_ready(plan)

    def test_summary_rejects_incomplete_or_incomparable_results(self):
        plan = post.prepare(self.settings, self.templates)
        for item in plan["groups"].values():
            config = post.read(item["config"])
            methods = {name: {"frames": 1, "depth": {"full": {"gt_pixels": 10, "MAE": 1.}},
                              "stages_stride8": {}, "surface": {}}
                       for name in config["methods"]}
            report = {"completed": True, "split": "test", "datasets": {
                config["datasets"][0]["name"]: {"units": "meters", "methods": methods}}}
            post.write_json(Path(item["output"]) / "full_test/summary.json", report)
        report = post.summarize(plan, {})
        self.assertTrue(report["completed"])
        self.assertEqual(report["prediction_calls"], 14)
        filename = Path(plan["groups"]["public"]["output"]) / "full_test/summary.json"
        result = post.read(filename)
        result["datasets"]["public_domain"]["methods"]["v2_old_public"]["depth"]["full"]["gt_pixels"] = 11
        post.write_json(filename, result)
        with self.assertRaisesRegex(ValueError, "GT分母不同"):
            post.summarize(plan, {})


if __name__ == "__main__":
    unittest.main()
