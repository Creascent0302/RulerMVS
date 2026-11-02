"""全训薄接口的增强顺序和配置不隐式更换原数据/目标。"""

import copy
from unittest.mock import patch

from research.v7_domain_20261102.full_training_round2 import V7Round2Hooks
from research.v7_domain_20261102.prepare_full_round2 import make_config
from research.v7_domain_20261102.test_augmentation import sample


def test_prepare_order_and_original_targets():
    config = {"input_mode": "native", "datasets": [], "domain_augmentation": True,
              "synthetic_pair": {"enabled": True, "probability": 1.}}
    hooks = V7Round2Hooks(config)
    inputs = sample()
    targets = {"original": True}
    calls = []

    def synthetic(images, labels, domain, **options):
        assert labels is targets
        calls.append("synthetic")
        return dict(images, tag="generated"), {"generated": True}, {
            "active": images["view_mask"][:, 0],
            "source_coverage": images["view_mask"][:, 0].float(),
        }

    def gray(images, **options):
        assert images["tag"] == "generated"
        calls.append("grayscale_subset")
        return images, {"grayscale": images["view_mask"][:, 0]}

    with patch("research.v7_domain_20261102.full_training_round2.augment_training_batch", synthetic), \
            patch("research.v7_domain_20261102.full_training_round2.augment_inputs", gray):
        batch, domain = hooks.prepare_batch({"inputs": inputs, "targets": targets}, {})
    assert calls == ["synthetic", "grayscale_subset"]
    assert targets == {"original": True} and batch["targets"] == {"generated": True}
    assert "tag" not in inputs


def test_config_preserves_full_coverage_and_independent_v6_starts():
    recipe = {"prior_kind": "coordinate", "output": "/data/example", "phases": {}}
    for phase in ("public", "carla"):
        recipe["phases"][phase] = {
            "v6_checkpoint": f"/data/v6_{phase}.pt", "stages": [{
                "name": "new_only", "epochs": 2, "trainable_prefixes": ["decoder.context_prior."],
                "prior_lr": 3e-4,
            }],
        }
    before = copy.deepcopy(recipe)
    public, carla = make_config(recipe, "public"), make_config(recipe, "carla")
    assert recipe == before
    assert len(public["datasets"]) == 7 and len(carla["datasets"]) == 1
    assert public["retention_checkpoint"] != carla["retention_checkpoint"]
    assert public["initial_model_checkpoint"] != carla["initial_model_checkpoint"]
    assert public["architecture"] == "rulermvs_v7_coordinate"
    assert public["stages"][0]["decoder_lr"] == 0
    assert public["stages"][0]["prior_lr"] == 3e-4
    assert not public["synthetic_pair"]["enabled"]
    assert not public["evaluation"]["test_after_training"]
