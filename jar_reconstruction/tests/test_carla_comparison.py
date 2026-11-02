"""固定GT分母、共同ROI、表示隔离与CARLA时间隔离合同。"""

import json
import unittest
from pathlib import Path

import torch

from jar.carla_comparison import (
    aggregate_depth,
    depth_counts,
    evaluate_depth,
    finish_counts,
    gt_reference,
    prediction_streams,
)


def fixture():
    images = torch.zeros(1, 4, 3, 4, 6)
    inputs = {
        "rgb": images,
        "K": torch.eye(3).repeat(1, 4, 1, 1),
        "T_cam_from_rig": torch.eye(4).repeat(1, 4, 1, 1),
        "view_mask": torch.ones(1, 4, dtype=torch.bool),
        "image_mask": torch.ones(1, 4, 4, 6, dtype=torch.bool),
    }
    targets = {
        "depth": torch.full((1, 4, 4, 6), 5.0),
        "depth_valid": torch.zeros(1, 4, 4, 6, dtype=torch.bool),
    }
    targets["depth_valid"][:, 2] = True
    settings = {
        "reference_index": 2,
        "depth_range": [2, 80],
        "roi": [[-100, 100], [-100, 100], [0, 80]],
        "absolute_thresholds_m": [0.1, 0.2, 1.0],
        "distance_bands_m": [[2, 10], [10, 24], [24, 80]],
    }
    return inputs, targets, settings


class ComparisonTests(unittest.TestCase):
    def test_device_without_index_resolves_to_visible_zero(self):
        from tools.carla_compare import resolve_device

        self.assertEqual(resolve_device("cuda"), torch.device("cuda:0"))
        self.assertEqual(resolve_device("cuda:3"), torch.device("cuda:3"))
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_missing_predictions_are_failures_in_fixed_denominator(self):
        gt = torch.tensor([5.0, 5.0, 5.0, 5.0])
        prediction = torch.tensor([5.0, 7.0, float("nan"), 5.0])
        accepted = torch.tensor([True, True, True, False])
        counts = depth_counts(prediction, accepted, gt, torch.ones(4, dtype=torch.bool), [0.2])
        score = finish_counts(counts, [0.2])
        self.assertEqual(score["coverage"], 0.5)
        self.assertEqual(score["success_0.2m_all_gt"], 0.25)
        self.assertEqual(score["wrong_accept_0.2m_of_accepted"], 0.5)
        self.assertEqual(score["delta1_all_gt"], 0.25)
        self.assertEqual(counts["declared_nonfinite_pixels"], 1)
        self.assertEqual(score["MAE_m"], 1)

    def test_no_gt_and_no_prediction_are_different(self):
        gt = torch.ones(4)
        empty = torch.zeros(4, dtype=torch.bool)
        no_gt = finish_counts(depth_counts(gt, empty, gt, empty, [0.2]), [0.2])
        rejected = finish_counts(depth_counts(gt, empty, gt, ~empty, [0.2]), [0.2])
        self.assertIsNone(no_gt["coverage"])
        self.assertEqual(rejected["coverage"], 0)
        self.assertEqual(rejected["success_0.2m_all_gt"], 0)
        self.assertIsNone(rejected["MAE_m"])

    def test_reference_roi_and_missing_camera(self):
        inputs, targets, settings = fixture()
        settings["roi"][0] = [0, 10]
        _, valid = gt_reference(inputs, targets, settings)
        self.assertEqual(int(valid.sum()), 12)
        inputs["view_mask"][0, 2] = False
        _, valid = gt_reference(inputs, targets, settings)
        self.assertEqual(int(valid.sum()), 0)

    def test_pixel_weighted_not_frame_weighted(self):
        _, _, settings = fixture()
        rows = []
        for target, prediction in [
            (torch.ones(1) * 5, torch.ones(1) * 6),
            (torch.ones(3) * 5, torch.ones(3) * 5),
        ]:
            valid = torch.ones_like(target, dtype=torch.bool)
            counts = finish_counts(
                depth_counts(prediction, valid, target, valid, settings["absolute_thresholds_m"]),
                settings["absolute_thresholds_m"],
            )
            rows.append({"streams": {"native_depth": {"all": counts}}})
        self.assertEqual(aggregate_depth(rows, settings)["native_depth"]["all"]["MAE_m"], 0.25)

    def test_official_reference_order_does_not_change_score(self):
        inputs, _, _ = fixture()
        depth = torch.stack([torch.full((4, 6), 5.0), torch.full((4, 6), 9.0)])[None]
        outputs = {
            "depth": depth,
            "reference_indices": [2, 0],
            "depth_K": inputs["K"][:, [2, 0]],
        }
        stream = prediction_streams(outputs, inputs, "fast_mvsnet", {}, 2)["native_depth"]
        torch.testing.assert_close(stream["depth"], torch.full((4, 6), 5.0))
        self.assertTrue(stream["valid"].all())

    def test_sparse_zbuffer_keeps_front_surface_and_does_not_fill(self):
        inputs, targets, settings = fixture()
        scene = {
            "points": torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 8.0]]),
            "valid": torch.ones(2, dtype=torch.bool),
            "state": torch.ones(2, dtype=torch.long),
        }
        streams = prediction_streams({"scenes": [scene]}, inputs, "mf_et_jar", {}, 2)
        splat = streams["final_point_splat"]
        self.assertEqual(int(splat["valid"].sum()), 1)
        self.assertEqual(splat["depth"][0, 0], 5)
        target, valid = gt_reference(inputs, targets, settings)
        score = evaluate_depth(splat["depth"], splat["valid"], target, valid, settings)
        self.assertAlmostEqual(score["all"]["coverage"], 1 / 24)

    def test_reference_query_depth_does_not_import_other_views_background(self):
        inputs, _, _ = fixture()
        points = torch.tensor([[8.0, 8.0, 8.0], [0, 0, 5], [0, 0, 7], [12, 12, 6]])
        stage = {
            "points": points,
            "valid": torch.tensor([True, True, True, False]),
            "state": torch.ones(4, dtype=torch.long),
            "aux": {
                "ref_index": torch.tensor([0, 2, 2, 2]),
                "ref_uv": torch.tensor([[1.0, 1.0], [0, 0], [0, 0], [2, 2]]),
                "reference_depth": torch.tensor([8.0, 5.0, 7.0, 6.0]),
            },
        }
        scene = dict(stage, stages=[stage])
        streams = prediction_streams({"scenes": [scene]}, inputs, "mf_et_jar", {}, 2)
        self.assertTrue(streams["final_point_splat"]["valid"][1, 1])
        query = streams["unfused_reference_query_depth"]
        self.assertFalse(query["valid"][1, 1])
        self.assertEqual(int(query["valid"].sum()), 1)
        self.assertEqual(query["depth"][0, 0], 5)
        inputs["view_mask"][0, 2] = False
        streams = prediction_streams({"scenes": [scene]}, inputs, "mf_et_jar", {}, 2)
        self.assertFalse(streams["unfused_reference_query_depth"]["valid"].any())

    def test_coarse_stream_not_presented_as_final_depth(self):
        inputs, _, _ = fixture()
        outputs = {"coarse": {"depth": torch.full((1, 4, 2, 3), 5.0), "stride": 2}}
        streams = prediction_streams(outputs, inputs, "mf_et_jar", {})
        self.assertIn("coarse_depth_diagnostic", streams)
        self.assertNotIn("native_depth", streams)

    def test_project_and_official_resize_use_same_pixel_centers(self):
        from jar.data_transforms import resize_frame
        from official_methods.data import resize_sample

        inputs, targets, _ = fixture()
        inputs["rgb"] = torch.arange(4 * 3 * 4 * 6).float().reshape(1, 4, 3, 4, 6)
        targets["depth"] += torch.arange(6)[None, None, None]
        sample = {
            "inputs": {key: value[0] for key, value in inputs.items()},
            "targets": {key: value[0] for key, value in targets.items()},
        }
        project = resize_frame(sample, (7, 11), depth_resize_mode="nearest-exact")
        official = resize_sample(sample, (7, 11))
        for group, keys in (
            ("inputs", ("rgb", "K", "T_cam_from_rig", "image_mask", "view_mask")),
            ("targets", ("depth", "depth_valid")),
        ):
            for key in keys:
                torch.testing.assert_close(
                    project[group][key], official[group][key], atol=0, rtol=0
                )

    def test_depth_mapping_distinguishes_stride_and_resized_lattices(self):
        from official_methods.methods.niantic_common import depth_intrinsics
        from official_methods.postprocess import depth_on_image_grid

        inputs, _, _ = fixture()
        height, width = 12, 20
        inputs["rgb"] = torch.zeros(1, 4, 3, height, width)
        inputs["image_mask"] = torch.ones(1, 4, height, width, dtype=torch.bool)
        inputs["K"][:] = torch.tensor([[100.0, 0, 9.3], [0, 120.0, 5.1], [0, 0, 1]])
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        expected = 3 + 0.03 * x + 0.02 * y
        low_y, low_x = y[::2, ::2], x[::2, ::2]
        for resized in (False, True):
            if resized:
                k = depth_intrinsics(inputs["K"][:, [2]], 0.5)
                offset = 0.5
            else:
                k = inputs["K"][:, [2]].clone()
                k[..., :2, :] /= 2
                offset = 0.0
            depth = (3 + 0.03 * (low_x + offset) + 0.02 * (low_y + offset))[None, None]
            mapped = depth_on_image_grid(
                {"depth": depth, "depth_K": k, "reference_indices": [2]}, inputs
            )
            # 去掉复制填充的边界，仅检查真实网格内部的解析坐标映射。
            torch.testing.assert_close(
                mapped["depth"][0, 0, 2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-6, rtol=0
            )

    def test_manifest_temporal_gap_and_camera_order(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "configs/carla_comparison/common.json").read_text())
        manifests = cfg["data"]["manifests"]
        rows = {
            name: [json.loads(line) for line in Path(manifests[name]).read_text().splitlines()]
            for name in ("train_complete", "dev_complete", "test_complete")
        }
        self.assertEqual([len(rows[name]) for name in rows], [66, 25, 200])
        self.assertLessEqual(max(r["frame_id"] for r in rows["train_complete"]), 15892)
        self.assertGreaterEqual(min(r["frame_id"] for r in rows["dev_complete"]), 15933)
        for values in rows.values():
            for row in values:
                self.assertEqual(
                    [c["camera_id"] for c in row["cameras"]], cfg["data"]["camera_ids"]
                )

    def test_cas_native_batch_keeps_reference_and_original_loss_backward(self):
        from tools.train_carla_casmvs import native_batch, lr_multiplier
        from third_party.cas_mvsnet.module import cas_mvsnet_loss

        inputs, targets, settings = fixture()
        inputs["rgb"][:, 2] = 0.75
        images, matrices, labels, masks = native_batch(inputs, targets, settings)
        self.assertTrue((images[:, 0] == 0.75).all())
        self.assertEqual(matrices["stage1"].shape, (1, 4, 2, 4, 4))
        predictions = {
            name: {"depth": (value + 0.5).requires_grad_()} for name, value in labels.items()
        }
        loss, _ = cas_mvsnet_loss(predictions, labels, masks, dlossw=[0.5, 1, 2])
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 0.4375)
        for value in predictions.values():
            self.assertTrue(torch.isfinite(value["depth"].grad).all())
            self.assertTrue(value["depth"].grad.abs().sum() > 0)
        schedule = {
            "warmup_steps": 500,
            "warmup_factor": 1 / 3,
            "decay_epochs": [10, 12, 14],
            "gamma": 0.5,
        }
        self.assertAlmostEqual(lr_multiplier(0, schedule, 66), 1 / 3)
        self.assertEqual(lr_multiplier(500, schedule, 66), 1)
        self.assertEqual(lr_multiplier(660, schedule, 66), 0.5)

    def test_cas_undefined_projection_zero_fill_finite_backward(self):
        from third_party.cas_mvsnet.module import homo_warping

        feature = torch.randn(1, 2, 4, 6, requires_grad=True)
        reference = torch.eye(4)[None]
        source = reference.clone()
        source[:, 2, 3] = -1
        # depth=1 的整层恰好处于源相机焦平面；depth=2正常。
        output = homo_warping(feature, source, reference, torch.tensor([[1.0, 2.0]]))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(float(output[:, :, 0].detach().abs().sum()), 0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(float(feature.grad.abs().sum()), 0)

    def test_cas_adapter_reference_matches_direct_native_network(self):
        from jar.config import load_config
        from jar.official_casmvs import OfficialCasMVSNet
        from jar.synthetic import make_synthetic_batch
        from tools.train_carla_casmvs import native_batch

        torch.manual_seed(37)
        torch.set_num_threads(2)
        cfg = load_config("configs/unit_tiny.json")
        cfg["model"]["far"] = 24.0
        model = OfficialCasMVSNet(cfg).eval()
        batch = make_synthetic_batch(N=4, H=32, W=32)
        settings = {
            "reference_index": 2,
            "depth_range": [2, 24],
            "roi": [[-100, 100], [-100, 100], [-100, 100]],
        }
        images, matrices, _, _ = native_batch(batch["inputs"], batch["targets"], settings)
        with torch.inference_mode():
            direct = model.net(images, matrices, model.depth_values[None])["depth"]
            adapted = model(batch["inputs"])["dense_depth"][:, 2]
        torch.testing.assert_close(adapted, direct, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
