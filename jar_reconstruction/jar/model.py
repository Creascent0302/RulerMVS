import torch
from torch import nn
from .encoder import ImageFPN
from .coarse import PlaneSweep
from .geometry import validate_inputs, image_texture
from .multifocal.data import mask_camera_inputs


class ReconstructionModel(nn.Module):
    def __init__(self, cfg, checkpoint_state=None):
        super().__init__()
        self.cfg = cfg
        self.matching_evidence_enabled = (
            cfg["model"].get("matching_evidence", {}).get("enabled", False)
        )
        exchange_enabled = cfg["model"].get("feature_exchange", {}).get("enabled", False)
        dense_settings = cfg["model"].get("dense_decoder", {})
        self.raw_feature_reader_enabled = dense_settings.get("raw_feature_reader", {}).get(
            "enabled", False
        )
        if self.raw_feature_reader_enabled:
            foundation = cfg["model"].get("deep_encoder", {}).get("foundation", {})
            if not (
                cfg["model"].get("deep_encoder", {}).get("enabled", False)
                and foundation.get("enabled", False)
                and dense_settings.get("enabled", False)
            ):
                raise ValueError("原始特征读取需要 deep_encoder.foundation 和稠密解码器")
        refinement = dense_settings.get("calibrated_refinement", {})
        pretrained_refinement = (
            dense_settings.get("enabled", False)
            and refinement.get("enabled", True)
            and refinement.get("feature_source", "local") == "pretrained"
        )
        self.matching_features_required = (
            self.matching_evidence_enabled or exchange_enabled or pretrained_refinement
        )
        if cfg["model"].get("feature_enrichment", {}).get("enabled", False) and not cfg[
            "model"
        ].get("deep_encoder", {}).get("enabled", False):
            raise ValueError("feature_enrichment 需要开启 deep_encoder 和 foundation")
        if cfg["model"].get("deep_encoder", {}).get("enabled", False):
            from .models.deep_encoder import DeepImageFPN

            self.encoder = DeepImageFPN(cfg, checkpoint_state=checkpoint_state)
        else:
            self.encoder = ImageFPN(cfg)
        self.coarse = PlaneSweep(cfg)
        if cfg["model"]["name"] != "mf_et_jar":
            raise ValueError("当前自研架构仅支持 mf_et_jar")
        from .models.token_hierarchy import EncoderTokenHierarchyDecoder

        mf = cfg["model"].get("multifocal", {})
        self.multifocal_active = cfg.get("camera", {}).get(
            "mode", "wide4"
        ) == "wide4_tele2" and mf.get("enabled", True)
        if self.multifocal_active and cfg["model"].get("deep_encoder", {}).get("enabled", False):
            raise ValueError("深编码实验目前验证广角阵列；远焦模式请使用原编码配置")
        if self.multifocal_active:
            from .multifocal.frontend import MultiFocalFrontend
            from .multifocal.decoder import MultiFocalDecoder

            self.multifocal_frontend = MultiFocalFrontend(cfg)
            self.decoder = MultiFocalDecoder(cfg)
        else:
            self.decoder = EncoderTokenHierarchyDecoder(cfg)
        self.sparse_enabled = cfg["model"].get("sparse_decoder", {}).get("enabled", True)
        dense_settings = cfg["model"].get("dense_decoder", {})
        if not self.sparse_enabled and not dense_settings.get("enabled", False):
            raise ValueError("稀疏和稠密解码头至少开启一个")
        self.reference_indices = cfg["model"].get("execution", {}).get("reference_indices")
        if self.reference_indices is not None:
            if (
                not isinstance(self.reference_indices, (list, tuple))
                or not self.reference_indices
                or any(type(index) is not int or index < 0 for index in self.reference_indices)
                or len(set(self.reference_indices)) != len(self.reference_indices)
            ):
                raise ValueError("execution.reference_indices须为不重复的非负相机索引列表")
            foundation = cfg["model"].get("deep_encoder", {}).get("foundation", {})
            if (
                self.multifocal_active
                or self.sparse_enabled
                or not dense_settings.get("enabled", False)
                or exchange_enabled
                or cfg["model"].get("image_context", {}).get("enabled", False)
                or cfg["model"].get("feature_enrichment", {}).get("enabled", False)
                or foundation.get("fuse_matching", False)
            ):
                raise ValueError(
                    "参考视图快速路径仅支持广角稠密头，须关闭稀疏头、feature_exchange、"
                    "image_context、feature_enrichment及foundation.fuse_matching"
                )
            self.reference_indices = tuple(self.reference_indices)
        self.matching_image_size = cfg.get("data", {}).get("matching_image_size")
        if self.matching_image_size is not None:
            if len(self.matching_image_size) != 2 or min(self.matching_image_size) < 1:
                raise ValueError("data.matching_image_size须为正的[H,W]")
            if (
                cfg.get("camera", {}).get("mode") != "wide4"
                or self.sparse_enabled
                or not dense_settings.get("enabled", False)
                or exchange_enabled
            ):
                raise ValueError(
                    "matching_image_size当前仅支持wide4稠密头，须关闭sparse_decoder和feature_exchange"
                )
            if not self.matching_features_required:
                raise ValueError(
                    "matching_image_size需要实际启用matching_evidence或pretrained稠密细化"
                )
        if not self.sparse_enabled:
            self.decoder.requires_grad_(False)
        if dense_settings.get("enabled", False):
            if self.multifocal_active:
                raise ValueError("稠密解码头当前支持广角阵列；长焦分支暂不启用")
            from .models.dense_decoder import DenseDepthDecoder

            with torch.random.fork_rng(devices=[]):
                self.dense_decoder = DenseDepthDecoder(cfg, checkpoint_state=checkpoint_state)
        field_settings = cfg["model"].get("candidate_field", {})
        recovery_enabled = dense_settings.get("surface_recovery", {}).get("enabled", False)
        if recovery_enabled and not field_settings.get("enabled", False):
            raise ValueError("初始表面重选需要候选场")
        if recovery_enabled and int(field_settings.get("stride", 8)) != 4:
            raise ValueError("初始表面重选需要候选场与稠密初值共用 stride 4 像素网格")
        if field_settings.get("enabled", False):
            refinement = dense_settings.get("calibrated_refinement", {})
            if not dense_settings.get("enabled", False) or not refinement.get("enabled", False):
                raise ValueError("候选残差场需要稠密解码和细化器")
            if refinement.get("feature_source") != "pretrained":
                raise ValueError("候选残差场当前只与预训练描述子细化器配合")
            from .models.candidate_field import PersistentCandidateField

            field_state = {
                key[len("candidate_field."):]: value
                for key, value in (checkpoint_state or {}).items()
                if key.startswith("candidate_field.")
            }
            with torch.random.fork_rng(devices=[]):
                self.candidate_field = PersistentCandidateField(
                    {**field_settings, "capture_recovery": recovery_enabled}, field_state
                )
        if self.decoder.pretrained_matching_weight > 0 or self.matching_features_required:
            from .pretrained_matching import FrozenMVSMatching

            settings = cfg["model"].get("pretrained_matching", {})
            prefix = "pretrained_matching."
            matching_state = {
                key[len(prefix) :]: value
                for key, value in (checkpoint_state or {}).items()
                if key.startswith(prefix)
            }
            self.pretrained_matching = FrozenMVSMatching(
                settings.get("weights"),
                settings.get("prefix", "matching_model."),
                state=matching_state or None,
            )
        if exchange_enabled:
            if self.multifocal_active:
                raise ValueError("feature_exchange 当前只支持广角阵列")
            from .models.feature_exchange import CalibratedFeatureExchange

            with torch.random.fork_rng(devices=[]):
                self.feature_exchange = CalibratedFeatureExchange(cfg)
        if cfg["model"].get("image_context", {}).get("enabled", False):
            if self.multifocal_active:
                raise ValueError("image_context 本轮仅验证广角阵列，远焦模式请关闭该实验模块")
            from .models.image_context import CalibratedImageContext

            with torch.random.fork_rng(devices=[]):
                self.image_context = CalibratedImageContext(cfg)

    def forward(self, inputs, teacher=None):
        from .multifocal.view_order import canonicalize_wide_inputs, restore_wide_outputs

        ordered, restore_order = canonicalize_wide_inputs(inputs, self.cfg)
        prediction = self._forward(ordered, teacher=teacher)
        return restore_wide_outputs(prediction, restore_order)

    def _forward(self, inputs, teacher=None):
        if not self.training and teacher is not None:
            raise ValueError("GT teacher forcing is forbidden in eval")
        inputs = mask_camera_inputs(inputs, self.cfg.get("camera", {}))
        if self.cfg["model"].get("validate_inputs", True):
            validate_inputs(inputs)
        work = dict(inputs)
        if self.matching_image_size is not None:
            matching = inputs.get("matching_rgb")
            if matching is None or tuple(matching.shape[-2:]) != tuple(self.matching_image_size):
                raise ValueError(
                    "须从同一原图提供配置尺寸的matching_rgb，不能由低分辨率场景特征代替"
                )
            if matching.shape[:3] != inputs["rgb"].shape[:3]:
                raise ValueError("matching_rgb必须与场景RGB具有相同batch/相机/通道")
            low_h, low_w = inputs["rgb"].shape[-2:]
            high_h, high_w = matching.shape[-2:]
            work["_pretrained_matching_scale"] = matching.new_tensor(
                [high_w / low_w, high_h / low_h], dtype=torch.float32
            ).expand(inputs["rgb"].shape[0], -1)
        elif "matching_rgb" in inputs:
            raise ValueError("使用matching_rgb须显式配置data.matching_image_size")
        B, N = inputs["rgb"].shape[:2]
        if self.reference_indices is not None:
            if max(self.reference_indices) >= N:
                raise ValueError("execution.reference_indices超出实际相机数")
            work["_reference_indices"] = self.reference_indices
        work["_texture"] = image_texture(
            inputs["rgb"].reshape(B * N, *inputs["rgb"].shape[2:])
        ).reshape(B, N, 1, *inputs["rgb"].shape[-2:])
        if "camera" in self.cfg:
            from .multifocal.data import validate_camera_inputs

            validate_camera_inputs(work, self.cfg)
        if hasattr(self, "pretrained_matching") and (
            self.sparse_enabled or self.matching_features_required
        ):
            # 每帧各相机只编码一次；四级解码共用，不携带任何监督标签。
            work["_pretrained_matching_features"] = self.pretrained_matching(work)
        if getattr(self, "multifocal_active", False):
            from .multifocal.frontend import encode_native_sizes

            pyramid = encode_native_sizes(self.encoder, work)
            coarse = self.multifocal_frontend(work, pyramid, self.coarse)
        else:
            # Exact original path for wide4 (including MF with tele disabled).
            # Mixed native sizes need grouped encoding only when explicitly supplied.
            native_sizes = work.get("image_sizes")
            needs_native_encoding = False
            if native_sizes is not None:
                valid_sizes = native_sizes[work["view_mask"]]
                canvas_size = native_sizes.new_tensor(work["rgb"].shape[-2:])
                needs_native_encoding = bool((valid_sizes != canvas_size).any())
            if needs_native_encoding:
                if hasattr(self.encoder, "forward_features"):
                    raise ValueError("深编码实验目前要求相同图像尺寸；请先统一缩放并更新内参")
                from .multifocal.frontend import encode_native_sizes

                pyramid = encode_native_sizes(self.encoder, work)
            else:
                if hasattr(self.encoder, "forward_features"):
                    if self.raw_feature_reader_enabled:
                        pyramid, context, raw_maps = self.encoder.forward_features(
                            work, return_raw=True
                        )
                        work["_foundation_maps"] = raw_maps
                    else:
                        pyramid, context = self.encoder.forward_features(work)
                    if context:
                        foundation = self.cfg["model"]["deep_encoder"].get("foundation", {})
                        if foundation.get("fuse_matching", False):
                            pyramid = {
                                stride: feature + context[stride]
                                for stride, feature in pyramid.items()
                            }
                        else:
                            work["_image_context"] = context
                else:
                    pyramid = self.encoder(work)
            if hasattr(self, "feature_exchange"):
                pyramid = self.feature_exchange(work, pyramid, context=work.get("_image_context"))
            coarse = self.coarse(work, pyramid)
        if hasattr(self, "image_context"):
            # 上下文只注入查询；不能混入用于确认多相机匹配的原始特征。
            extra_context = self.image_context(work, pyramid)
            previous_context = work.get("_image_context")
            work["_image_context"] = (
                {
                    stride: value + previous_context[stride]
                    for stride, value in extra_context.items()
                }
                if previous_context
                else extra_context
            )
        coarse["evidence_threshold"] = self.cfg["model"].get("evidence_threshold", 0.15)
        coarse["texture_threshold"] = self.cfg["model"].get("texture_threshold", 0.002)
        coarse["min_support"] = self.cfg["model"].get("min_support", 2)
        decoded = (
            self.decoder(work, pyramid, coarse, teacher=teacher)
            if self.sparse_enabled
            else {"scenes": []}
        )
        if isinstance(decoded, list):
            decoded = {"scenes": decoded}
        decoded["coarse"] = coarse
        if self.reference_indices is not None:
            decoded["reference_indices"] = self.reference_indices
            decoded["execution"] = {
                "reference_indices": self.reference_indices,
                "source_views": N,
                "uncomputed_references": "invalid/unknown",
                "cross_reference_depth_evidence": "available_references_only",
            }
        if hasattr(self, "dense_decoder"):
            if hasattr(self, "candidate_field"):
                work["_candidate_fields"] = self.candidate_field(work, pyramid, self.coarse)
            # 稠密头可在更细格点重读全深度范围；共享匹配参数，原稀疏头仍读取stride16。
            dense_stride = self.cfg["model"]["dense_decoder"].get("full_range_stride", 16)
            dense_coarse = (
                coarse if dense_stride == 16 else self.coarse(work, pyramid, stride=dense_stride)
            )
            dense = self.dense_decoder(
                work, pyramid, dense_coarse, context=work.get("_image_context")
            )
            evidence_settings = self.cfg["model"].get("dense_evidence", {})
            if not self.training and evidence_settings.get("enabled", True):
                from .dense_evidence import verify_dense_depth

                evidence = verify_dense_depth(work, dense["depth"], self.cfg)
            else:
                valid = dense["valid"]
                evidence = {
                    "observed": torch.zeros_like(valid),
                    "support": torch.zeros_like(valid, dtype=torch.long),
                    "state": torch.full_like(valid, 2, dtype=torch.long),
                }
            dense.update(evidence)
            decoded.update(
                dense=dense,
                dense_depth=dense["depth"],
                dense_valid=dense["valid"],
                dense_confidence=dense["confidence"],
                dense_observed=dense["observed"],
                dense_support=dense["support"],
                dense_state=dense["state"],
                depth_stride=1,
            )
        return decoded
