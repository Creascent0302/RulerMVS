"""采集格式、缺失相机、部分深度监督和无监督更新保护。"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from jar.carla_capture import WIDE_IDS, prepare, camera_calibration, write_jsonl
from jar.config import load_config
from jar.data import build_dataset, collate_frames
from jar.data_transforms import resize_frame
from jar.losses import ReconstructionLoss
from jar.model import ReconstructionModel
from jar.training import train


def fixture(root, name):
    folder = root / name
    folder.mkdir()
    positions = {
        "stereo_left": (1.5, -1.5, 1.5),
        "stereo_right": (1.5, 1.5, 1.5),
        "stereo_up": (2.0, 0.0, 2.5),
        "stereo_down": (3.0, 0.0, 1.0),
    }
    calibration = {
        k: {
            "type": "rgb",
            "extrinsic": dict(zip(("x", "y", "z"), v)),
            "intrinsic": {"fx": 12.0, "fy": 12.0, "cx": 12.0, "cy": 8.0, "width": 24, "height": 16},
        }
        for k, v in positions.items()
    }
    calibration["depth"] = {**calibration["stereo_up"], "type": "depth"}
    calibration["radar"] = {"type": "radar"}
    (folder / "calibration.json").write_text(json.dumps(calibration))
    index = {}
    for fid, ids in [
        (1, WIDE_IDS),
        (2, ("stereo_up", "stereo_left")),
        (3, ("stereo_right",)),
        (4, ()),
    ]:
        entries = {}
        for cid in ids:
            path = folder / f"camera/{cid}/{cid}_{fid:08d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((16, 24, 3), 100 + fid, dtype=np.uint8)).save(path)
            entries[cid] = str(path.relative_to(folder))
        if fid in (1, 2, 4):
            path = folder / f"depth/depth_{fid:08d}.npy"
            path.parent.mkdir(exist_ok=True)
            np.save(path, np.full((16, 24), 8, dtype=np.float32))
            entries["depth"] = str(path.relative_to(folder))
        if fid == 1:
            path = folder / "radar/radar_00000001.npy"
            path.parent.mkdir(exist_ok=True)
            np.save(path, np.zeros((1, 4), np.float32))
            entries["radar"] = str(path.relative_to(folder))
        entries["tele_left"] = (
            "camera/tele_left/missing.png"  # 排除长焦，因此不读取它也不依赖其frameID。
        )
        index[f"{fid:08d}"] = entries
    (folder / "sync_index.json").write_text(json.dumps(index))
    path = folder / "camera/stereo_up/stereo_up_00000999.png"
    Image.fromarray(np.zeros((16, 24, 3), np.uint8)).save(path)


def config(root, out):
    cfg = load_config("configs/mf_et_jar_wide4_tiny.json")
    cfg["camera"] = {
        "mode": "wide4",
        "wide_ids": list(WIDE_IDS),
        "tele_ids": ["tele_left", "tele_right"],
        "allow_missing": True,
    }
    cfg["data"] = {
        "root": str(root),
        "train_manifest": str(out / "train_all.jsonl"),
        "test_manifest": str(out / "test_all.jsonl"),
        "image_size": [8, 12],
    }
    return cfg


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_index_groups_missing_labels_and_inference_label_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, "a")
            fixture(root, "b")
            out = root / "index"
            report = prepare(root, out, "a", "b", "optical_z")
            cfg = config(root, out)
            self.assertEqual(report["sequences"]["train"]["complete_rgb"], 1)
            self.assertEqual(report["sequences"]["train"]["partial_rgb"], 2)
            self.assertEqual(report["sequences"]["train"]["supervised_training_candidates"], 2)
            self.assertEqual(report["sequences"]["train"]["unindexed_file_count_excluding_tele"], 1)
            ds = build_dataset(cfg)
            complete, partial, single = ds[0], ds[1], ds[2]
            self.assertEqual(partial["inputs"]["view_mask"].tolist(), [True, False, True, False])
            self.assertEqual(partial["inputs"]["rgb"].shape, (4, 3, 8, 12))
            self.assertEqual(
                partial["inputs"]["image_sizes"].tolist(), [[8, 12], [0, 0], [8, 12], [0, 0]]
            )
            self.assertEqual(
                partial["targets"]["depth_valid"].flatten(1).any(-1).tolist(),
                [False, False, True, False],
            )
            self.assertEqual(float(partial["inputs"]["K"][2, 0, 2]), 5.75)
            self.assertIsNone(partial["metadata"]["timestamp"])
            batch = collate_frames([complete, partial, single])
            self.assertEqual(batch["inputs"]["rgb"].shape, (3, 4, 3, 8, 12))
            self.assertFalse(batch["targets"]["depth_valid"][2].any())
            for p in (root / "a/depth").glob("*.npy"):
                p.unlink()
            unlabeled = build_dataset(cfg, load_targets=False)
            self.assertEqual(
                unlabeled[0]["targets"], {}
            )  # 即使标签路径已损坏，正常推理也不会访问。

    def test_unverified_depth_blocks_supervision_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, "a")
            fixture(root, "b")
            out = root / "index"
            prepare(root, out, "a", "b")
            cfg = config(root, out)
            with self.assertRaisesRegex(ValueError, "深度定义未核实"):
                build_dataset(cfg)
            self.assertEqual(len(build_dataset(cfg, load_targets=False)), 3)
            cfg["camera"]["allow_missing"] = False
            with self.assertRaisesRegex(ValueError, "缺少必需相机"):
                build_dataset(cfg, load_targets=False)

    def test_empirical_depth_evidence_preserves_provenance_and_checks_index(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, "a")
            fixture(root, "b")
            evidence = {
                "sequence": "a",
                "calibration_sha256": hashlib.sha256(
                    (root / "a/calibration.json").read_bytes()
                ).hexdigest(),
                "index_sha256": hashlib.sha256(
                    (root / "a/sync_index.json").read_bytes()
                ).hexdigest(),
            }
            path = root / "audit.json"
            path.write_text(json.dumps(evidence))
            out = root / "index"
            report = prepare(root, out, "a", "b", "optical_z", depth_evidence=[path])
            provenance = report["sequences"]["train"]["depth_evidence"]
            self.assertEqual(provenance["basis"], "empirical_radar_projection_comparison")
            self.assertFalse(provenance["capture_decode_formula_verified"])
            self.assertTrue(report["sequences"]["train"]["supervision_ready"])
            rows = [
                json.loads(line)
                for line in (out / "train_supervised.jsonl").read_text().splitlines()
            ]
            label = next(c["targets"] for c in rows[0]["cameras"] if "targets" in c)
            self.assertEqual(label["depth_scale"], 1)
            self.assertEqual(label["depth_evidence"], provenance)
            evidence["index_sha256"] = "stale"
            path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "同步索引不一致"):
                prepare(root, out, "a", "b", "optical_z", depth_evidence=[path])

    def test_carla_rotation_and_camera_center(self):
        sensor = {
            "extrinsic": {"x": 1.5, "y": -1.5, "z": 1.5, "yaw": 12.0, "pitch": 3.0, "roll": -4.0},
            "intrinsic": {
                "fx": 960,
                "fy": 960,
                "cx": 960,
                "cy": 540,
                "height": 1080,
                "width": 1920,
            },
        }
        a = camera_calibration(sensor)
        T = torch.tensor(a["T_cam_from_rig"])
        R = T[:3, :3]
        torch.testing.assert_close(R @ R.T, torch.eye(3), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            -R.T @ T[:3, 3], torch.tensor([1.5, 1.5, 1.5]), atol=1e-6, rtol=1e-6
        )
        self.assertAlmostEqual(float(torch.det(R)), 1.0, places=5)

    def test_depth_resize_never_blends_foreground_and_background(self):
        rgb = torch.zeros(1, 3, 4, 4)
        depth = torch.tensor([[[2.0, 2.0, 80.0, 80.0]]]).expand(1, 4, 4).clone()
        item = {
            "inputs": {
                "rgb": rgb,
                "K": torch.eye(3)[None],
                "image_mask": torch.ones(1, 4, 4, dtype=torch.bool),
            },
            "targets": {"depth": depth, "depth_valid": torch.ones_like(depth, dtype=torch.bool)},
            "metadata": {},
        }
        got = resize_frame(item, [2, 2])
        self.assertEqual(set(got["targets"]["depth"].flatten().tolist()), {2.0, 80.0})
        self.assertEqual(got["inputs"]["K"][0, 0, 2].item(), -0.25)

    def test_saturated_ray_distance_stays_invalid_after_z_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root, "a")
            fixture(root, "b")
            out = root / "index"
            prepare(root, out, "a", "b", "ray_range")
            np.save(
                root / "a/depth/depth_00000001.npy", np.full((16, 24), 1000.0, dtype=np.float32)
            )
            ds = build_dataset(config(root, out))
            self.assertFalse(ds[0]["targets"]["depth_valid"].any())

    def test_absent_source_depth_is_not_negative_matching_supervision(self):
        from jar.synthetic import synthetic_rig

        K, T = synthetic_rig(2, (32, 32))
        rgb = torch.ones(1, 2, 3, 32, 32)
        inputs = {
            "rgb": rgb,
            "K": K[None],
            "T_cam_from_rig": T[None],
            "view_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        depths = torch.full((1, 2, 32, 32), 4.0)
        valid = torch.zeros_like(depths, dtype=torch.bool)
        valid[:, 0] = True
        logits = torch.zeros(1, 2, 4, 2, 2, requires_grad=True)
        coarse = {
            "depth": torch.full((1, 2, 2, 2), 4.0),
            "logits": logits,
            "inverse_depths": torch.tensor([0.5, 0.25, 0.125]),
            "texture": torch.ones(1, 2, 2, 2),
        }
        loss = ReconstructionLoss()(
            {"coarse": coarse, "scenes": []}, inputs, {"depth": depths, "depth_valid": valid}
        )
        loss["total"].backward()
        # 中心候选投影在第二视图内，其深度缺失；no-match通道不能获得负例梯度。
        self.assertEqual(float(logits.grad[0, 0, -1, 1, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 0, :-1, 1, 1].abs().sum()), 0.0)
        self.assertIn("coarse_depth_class_partial", loss)

    def test_no_supervision_does_not_decay_weights_or_count_step(self):
        from jar.synthetic import SyntheticRigDataset

        cfg = load_config("configs/mf_et_jar_wide4_tiny.json")
        cfg["train"]["max_steps"] = 1
        sample = SyntheticRigDataset(length=1, image_size=(32, 48), num_cameras=4)[0]
        sample["targets"] = {}
        model = ReconstructionModel(cfg)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            path.write_text(json.dumps(cfg))
            args = SimpleNamespace(
                config=str(path),
                synthetic=True,
                device="cpu",
                output=str(Path(d) / "run"),
            )
            with (
                patch("jar.training.build_dataset", return_value=[sample]),
                patch("jar.training.build_model", return_value=model),
            ):
                with self.assertRaisesRegex(ValueError, "没有可用监督"):
                    train(args)
            events = [json.loads(v) for v in (Path(d) / "run/train.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["event"], "no_supervision")
            self.assertEqual(events[0]["step"], 0)
            self.assertFalse((Path(d) / "run/last.pt").exists())
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(before[k], v), k)


class PartialArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_final_architecture_accepts_stable_missing_camera_slots(self):
        from jar.synthetic import SyntheticRigDataset
        from jar.multifocal.options import configure

        item = SyntheticRigDataset(length=1, image_size=(32, 48), num_cameras=4, seed=29)[0]
        a = item["inputs"]
        a["view_mask"][1] = False
        a["rgb"][1] = 0
        a["image_mask"][1] = False
        a.update(
            camera_ids=list(WIDE_IDS),
            camera_roles=torch.zeros(4, dtype=torch.long),
            camera_valid=a["view_mask"].clone(),
            image_sizes=torch.tensor([[32, 48], [0, 0], [32, 48], [32, 48]]),
        )
        item["targets"]["depth_valid"][1] = False
        batch = collate_frames([item])
        for name in ("mf_et_jar",):
            with self.subTest(model=name):
                cfg = load_config("configs/mf_et_jar_wide4_tiny.json")
                cfg["model"]["name"] = name
                cfg["camera"] = {"mode": "wide4", "wide_ids": list(WIDE_IDS), "allow_missing": True}
                cfg = configure(cfg)
                model = ReconstructionModel(cfg)
                out = model(batch["inputs"])
                loss = ReconstructionLoss(cfg)(out, batch["inputs"], batch["targets"])["total"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(torch.isfinite(out["scenes"][0]["points"]).all())


class PartialMetricsTests(unittest.TestCase):
    def test_unlabeled_field_of_view_not_counted_as_false_surface(self):
        from jar.metrics import evaluate_batch

        a = {
            "rgb": torch.zeros(1, 1, 3, 4, 4),
            "K": torch.eye(3)[None, None],
            "T_cam_from_rig": torch.eye(4)[None, None],
            "view_mask": torch.ones(1, 1, dtype=torch.bool),
            "image_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        }
        depth = torch.zeros(1, 1, 4, 4)
        depth[0, 0, 0, 0] = 4.0
        t = {"depth": depth, "depth_valid": depth > 0}
        scene = {
            "points": torch.tensor([[0.0, 0.0, 4.0], [100.0, 0.0, 4.0]]),
            "state": torch.ones(2, dtype=torch.long),
            "valid": torch.ones(2, dtype=torch.bool),
        }
        output = {"scenes": [scene]}
        got = evaluate_batch(output, a, t, roi=[[-200, 200]] * 3, observed_only=True)
        self.assertEqual(got["surface_pred_samples"], 1)
        self.assertEqual(got["surface_fscore"], 1.0)
        t["depth_valid"].zero_()
        self.assertIsNone(evaluate_batch(output, a, t, observed_only=True)["surface_fscore"])


if __name__ == "__main__":
    unittest.main()
