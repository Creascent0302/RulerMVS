import copy
import unittest
import torch
from jar import build_model, load_config
from jar.synthetic import make_synthetic_batch, SyntheticRigDataset
from jar.data import collate_frames
from jar.losses import ReconstructionLoss


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from tensors(v)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_final_model_variable_cameras_backward_optimizer(self):
        for name in ("mf_et_jar",):
            for n in [2, 3, 4]:
                with self.subTest(model=name, cameras=n):
                    torch.manual_seed(71)
                    cfg = load_config("configs/unit_tiny.json")
                    cfg["model"]["name"] = name
                    batch = make_synthetic_batch(N=n, H=32, W=48)
                    model = build_model(cfg).train()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    prediction = model(batch["inputs"])
                    self.assertTrue(all(torch.isfinite(t).all() for t in tensors(prediction)))
                    loss = ReconstructionLoss(cfg)(prediction, batch["inputs"], batch["targets"])
                    self.assertTrue(torch.isfinite(loss["total"]))
                    loss["total"].backward()
                    for module in [model.encoder, model.coarse, model.decoder]:
                        gradients = [p.grad for p in module.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
                    previous = next(model.encoder.parameters()).detach().clone()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    self.assertFalse(
                        torch.equal(previous, next(model.encoder.parameters()).detach())
                    )
                    scene = prediction["scenes"][0]
                    self.assertEqual(scene["points"].shape, scene["normals"].shape)
                    self.assertTrue((scene["scales"] > 0).all())
                    self.assertTrue(((scene["state"] >= 0) & (scene["state"] <= 2)).all())

    def test_target_independence_and_eval_teacher_forbidden(self):
        batch = make_synthetic_batch(N=3, H=32, W=48)
        for name in ("mf_et_jar",):
            with self.subTest(model=name):
                cfg = load_config("configs/unit_tiny.json")
                cfg["model"]["name"] = name
                model = build_model(cfg).eval()
                original_inputs = {k: v.clone() for k, v in batch["inputs"].items()}
                with torch.no_grad():
                    before = model(batch["inputs"])
                changed = copy.deepcopy(batch["targets"])
                changed["depth"].fill_(67.0)
                changed["depth_valid"].logical_not_()
                changed["semantic"].fill_(0)
                with torch.no_grad():
                    after = model(batch["inputs"])
                for a, b in zip(tensors(before), tensors(after)):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                for key, value in original_inputs.items():
                    torch.testing.assert_close(batch["inputs"][key], value, atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    model(batch["inputs"], teacher=object())

    def test_variable_camera_batch_and_all_missing_unknown(self):
        a = SyntheticRigDataset(length=1, image_size=(32, 48), num_cameras=2)[0]
        b = SyntheticRigDataset(length=1, image_size=(32, 48), num_cameras=4)[0]
        batch = collate_frames([a, b])
        self.assertEqual(batch["inputs"]["rgb"].shape[:2], (2, 4))
        self.assertFalse(batch["inputs"]["view_mask"][0, 2:].any())
        for name in ("mf_et_jar",):
            cfg = load_config("configs/unit_tiny.json")
            cfg["model"]["name"] = name
            model = build_model(cfg).eval()
            with torch.no_grad():
                output = model(batch["inputs"])
            self.assertEqual(len(output["scenes"]), 2)
            missing = {k: v.clone() for k, v in batch["inputs"].items()}
            missing["view_mask"].fill_(False)
            with torch.no_grad():
                output = model(missing)
            for scene in output["scenes"]:
                self.assertTrue((scene["state"] == 2).all())
                self.assertFalse(scene["valid"].any())
                self.assertFalse(scene["support"].any())
                self.assertTrue(torch.isfinite(scene["points"]).all())

    def test_masked_views_cannot_supply_semantic_supervision(self):
        cfg = load_config("configs/unit_tiny.json")
        cfg["model"]["semantic_classes"] = 5
        batch = make_synthetic_batch(N=2, H=32, W=48)
        batch["inputs"]["view_mask"].fill_(False)
        model = build_model(cfg)
        output = model(batch["inputs"])
        only_labels = {"semantic": batch["targets"]["semantic"]}
        loss = ReconstructionLoss(cfg)(output, batch["inputs"], only_labels)
        self.assertEqual(float(loss["total"].detach()), 0.0)
        loss["total"].backward()
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )


if __name__ == "__main__":
    unittest.main()
