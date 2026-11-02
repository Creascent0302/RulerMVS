"""复用四层 ET 解码块，只扩展出生、深度范围、匹配、验收与融合钩子。"""

import torch
from jar.models.token_hierarchy import EncoderTokenHierarchyDecoder, _reference_sample
from jar.query_selection import select_query_candidates
from jar.geometry import project, sample_maps
from .frontend import segmented_depths
from .proposals import TeleBirthScorer, tele_births


def _nearest_modes(maps, uv, refs, image_sizes=None):
    # 深度是多峰几何标签；跨像素双线性插值会混入前后两个表面。
    if image_sizes is not None:
        largest = ((image_sizes[refs][:, [1, 0]] - 1).clamp_min(0) // 16) * 16
        uv = uv.minimum(largest).clamp_min(0)
    values = sample_maps(
        maps, uv[None].expand(len(maps), -1, -1), 16, mode="nearest", padding_mode="border"
    )
    return values[refs, torch.arange(len(refs), device=refs.device)]


class MultiFocalDecoder(EncoderTokenHierarchyDecoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        from .matching import MultiFocalSampler
        from .polish import MFOrientedPolisher
        from .evidence import ObservabilityVerifier

        self.mf = cfg["model"].get("multifocal", {})
        if self.polish_stages:
            raise ValueError(
                "MF使用按焦距分组的有向精修，不能额外启用et_polish_stages旧广角深度细化器"
            )
        self.tele_near = float(self.mf.get("tele_near", 10.0))
        self.tele_far = float(self.mf.get("tele_far", 150.0))
        self.sampler = MultiFocalSampler(cfg)
        self.birth_scorer = (
            TeleBirthScorer(cfg["model"]["feature_dim"])
            if self.mf.get("tele_birth", True)
            else None
        )
        self.verifier = ObservabilityVerifier(cfg)
        self.register_buffer(
            "tele_global_depths",
            segmented_depths(self.tele_near, self.tele_far, self.global_count),
            persistent=False,
        )
        if self.oriented_polisher is not None:
            self.oriented_polisher = MFOrientedPolisher(cfg)
        self.free_space_source = "verified_queries"
        self.birth_shares = tuple(self.mf.get("birth_shares", [0.5, 1 / 3, 1 / 6]))
        if (
            len(self.birth_shares) != 3
            or min(self.birth_shares) < 0
            or abs(sum(self.birth_shares) - 1) > 1e-6
        ):
            raise ValueError("birth_shares 是 wide/tele/explore 三个非负比例且总和为1")

    def _stage_queries(self, inputs, pyramid, coarse, b, stage, previous):
        rgb = inputs["rgb"][b]
        device = rgb.device
        budget = max(1, int(self.query_budget * self.stage_fractions[stage]))
        old_count = 0 if previous is None else len(previous["uv"])
        count = max(0, budget - old_count) if previous is None or self.detail_births else 0
        old_refs = None if previous is None else previous["refs"]
        old_uv = None if previous is None else previous["uv"]
        wide_count = int(count * self.birth_shares[0])
        tele_count = int(count * self.birth_shares[1])
        explore_count = count - wide_count - tele_count
        if not self.mf.get("tele_birth", True):
            wide_count += tele_count
            tele_count = 0
        wide_inputs = dict(inputs)
        wide_inputs["view_mask"] = inputs["view_mask"] & (inputs["camera_roles"] == 0)
        edge_level = inputs.get("_edge_levels", [{}] * len(inputs["rgb"]))[b].get(
            self.strides[stage]
        )
        children = None
        if self.parent_children and previous is not None:
            from jar.hierarchy_queries import child_queries

            # 子查询只使用广角名额，独立探索与长焦预算均保留。
            child_budget = min(wide_count, int((wide_count + explore_count) * self.parent_fraction))
            children = child_queries(
                wide_inputs,
                b,
                previous,
                child_budget,
                self.strides[stage],
                edge_level if self.edge_queries else None,
                self.roi if self.roi_proposals else None,
                self.near,
                self.far,
            )
            wide_count -= len(children["uv"])
            old_refs = torch.cat((old_refs, children["refs"]))
            old_uv = torch.cat((old_uv, children["uv"]))
        policy = select_query_candidates(
            wide_inputs,
            b,
            wide_count + explore_count,
            old_refs,
            old_uv,
            policy="coverage_edge_pairs" if self.edge_queries else "phase_grid",
            stride=self.strides[stage],
            coverage_fraction=self.exploration,
            roi=self.roi if self.roi_proposals else None,
            near=self.near,
            far=self.far,
            edge_level=edge_level if self.edge_queries else None,
            edge_cell_size=max(4, self.strides[stage]),
        )
        uv, refs = policy["ref_uv"], policy["ref_index"]
        child_parent = None
        if children is not None:
            child_parent = torch.cat((children["parent_id"], torch.full_like(refs, -1)))
            uv = torch.cat((children["uv"], uv))
            refs = torch.cat((children["refs"], refs))
        kind = torch.ones(len(uv), dtype=torch.long, device=device)
        if explore_count:
            kind[-min(len(uv), explore_count) :] = 3
        if tele_count:
            tr, tu, tk = tele_births(
                inputs,
                b,
                tele_count,
                inputs["_mf_birth_logits"],
                old_refs,
                old_uv,
                exploration=float(self.mf.get("tele_exploration", 0.25)),
                cell=int(self.mf.get("tele_coverage_cell", 16)),
            )
            uv = torch.cat((uv, tu))
            refs = torch.cat((refs, tr))
            kind = torch.cat((kind, tk))
            if child_parent is not None:
                child_parent = torch.cat((child_parent, torch.full_like(tr, -1)))
        if previous is None and not len(uv):
            uv = rgb.new_zeros(1, 2)
            refs = torch.zeros(1, device=device, dtype=torch.long)
            kind = torch.full_like(refs, 4)
        depth = (
            _nearest_modes(coarse["depth"][b, :, None].float(), uv, refs, inputs["image_sizes"][b])[
                :, 0
            ]
            if len(uv)
            else rgb.new_empty(0)
        )
        depth = self._clamp_depth(depth, refs, inputs, b)
        peaks = (
            _nearest_modes(coarse["peaks"][b].float(), uv, refs, inputs["image_sizes"][b])
            if len(uv)
            else rgb.new_empty(0, 2)
        )
        peaks = self._clamp_depth(peaks, refs, inputs, b)
        parent = rgb.new_zeros(len(uv), self.hidden)
        pid = torch.full_like(refs, -1)
        if previous is not None and len(uv):
            pid = (
                child_parent
                if child_parent is not None
                else self._parents(uv, refs, previous, self.strides[stage - 1], rgb.shape[-2:])
            )
            # 独立长焦出生不继承任何旧射线深度，包括自己的长焦父格。
            pid = torch.where(inputs["camera_roles"][b, refs] == 1, -1, pid)
            has = pid >= 0
            parent = torch.where(has[:, None], previous["features"][pid.clamp_min(0)], parent)
            depth = torch.where(has, previous["depth"][pid.clamp_min(0)], depth)
            if self.parent_children:
                from jar.hierarchy_queries import inherit_plane_depth

                inherited = inherit_plane_depth(
                    inputs, b, uv, refs, pid.clamp_min(0), previous, self.near, self.far
                )
                depth = torch.where(has, inherited, depth)
        parent_valid = torch.zeros(len(refs), device=device, dtype=torch.bool)
        if previous is not None and (self.parent_children or self.protect_parent):
            parent_valid = (pid >= 0) & previous["valid"][pid.clamp_min(0)]
        born = torch.ones(len(refs), device=device, dtype=torch.bool)
        if previous is not None:
            uv = torch.cat((previous["uv"], uv))
            refs = torch.cat((previous["refs"], refs))
            depth = torch.cat((previous["depth"], depth))
            parent = torch.cat((previous["features"], parent))
            pid = torch.cat((torch.arange(old_count, device=device), pid))
            born = torch.cat((torch.zeros(old_count, device=device, dtype=torch.bool), born))
            if self.parent_children or self.protect_parent:
                parent_valid = torch.cat((previous["valid"], parent_valid))
            kind = torch.cat((torch.zeros(old_count, device=device, dtype=torch.long), kind))
            old_peaks = _nearest_modes(
                coarse["peaks"][b].float(),
                previous["uv"],
                previous["refs"],
                inputs["image_sizes"][b],
            )
            peaks = torch.cat((self._clamp_depth(old_peaks, previous["refs"], inputs, b), peaks))
        queries = dict(
            uv=uv,
            refs=refs,
            depth=depth,
            parent=parent,
            born=born,
            parent_id=pid,
            source_kind=kind,
            depth_modes=peaks,
        )
        if self.parent_children or self.protect_parent:
            queries["parent_valid"] = parent_valid
        return queries

    def _clamp_depth(self, depth, refs, inputs, b):
        tele = inputs["camera_roles"][b, refs] == 1
        near = torch.where(tele, self.tele_near, self.near)
        far = torch.where(tele, self.tele_far, self.far)
        while near.ndim < depth.ndim:
            near = near[:, None]
            far = far[:, None]
        return depth.maximum(near).minimum(far)

    def _source_sampling_stride(self, inputs, b, points, refs, stride, jacobian):
        if not self.mf.get("scale_routing", True):
            return stride
        from .matching import _projection_scales

        with torch.autocast(device_type=points.device.type, enabled=False):
            normals = inputs["T_cam_from_rig"][b, refs, 2, :3].float()[:, None]
            rms, _, _ = _projection_scales(jacobian.float()[:, :, None], refs, normals)
            return (rms[:, :, 0, 0] * stride).clamp(4.0, 32.0)

    def _depth_candidates(self, inputs, b, query, step):
        refs = query["refs"]
        depth = query["depth"].float()
        local = self._clamp_depth(
            depth[:, None] + self.offsets[None] * step[:, None], refs, inputs, b
        )
        global_depth = torch.where(
            (inputs["camera_roles"][b, refs] == 1)[:, None],
            self.tele_global_depths[None],
            self.global_inverse_depth.reciprocal()[None],
        )
        # 第二个模式只成为另一个明确候选；不平均前景/背景。
        second = query["depth_modes"][:, 1:2]
        return torch.cat((local, global_depth, second), -1)

    def _candidate_observations(self, inputs, pyramid, b, stride, points, refs, uv, normals=None):
        return self.sampler.sample(inputs, pyramid, b, stride, points, refs, uv, normals)

    def _finish_chunk(self, scene, inputs, b, stage, query, routed):
        aux = scene["aux"]
        refs = query["refs"]
        aux["source_kind"] = query["source_kind"]
        aux["camera_role"] = inputs["camera_roles"][b, refs]
        aux["initial_depth_modes"] = query["depth_modes"]
        tele = inputs["camera_roles"][b, refs] == 1
        aux["depth_bounds_m"] = torch.stack(
            (
                torch.where(tele, self.tele_near, self.near),
                torch.where(tele, self.tele_far, self.far),
            ),
            -1,
        )
        # 覆盖半径来自参考图像物理足迹，与depth_sigma单独记录。
        _, _, _, jac = project(
            scene["points"],
            inputs["K"][b],
            inputs["T_cam_from_rig"][b],
            inputs["rgb"].shape[-2:],
            inputs["view_mask"][b],
            inputs["image_mask"][b],
        )
        row = torch.arange(len(refs), device=refs.device)
        footprint = (
            self.strides[stage] / jac[refs, row].square().sum((-1, -2)).sqrt().clamp_min(1e-6)
        ).clamp(0.001, 20.0)
        scene["scales"] = footprint[:, None].expand(-1, 3)
        scene["sizes"] = scene["scales"] * 2
        aux["coverage_radius_m"] = footprint
        if routed is not None:
            debug = routed.get("scale_debug", {})
            # Q轴张量才能由原有chunk拼接正确合并。
            for key, value in debug.items():
                if torch.is_tensor(value) and value.ndim and len(value) == len(refs):
                    aux["mf_" + key] = value
        return scene

    def _finish_stage(self, output, inputs, pyramid, b, stage):
        output["aux"]["architecture"] = "mf_encoder_token_hierarchy"
        if stage != 3 or not self.mf.get("observability", True):
            return output
        from .matching import photometric_observations

        aux = output["aux"]
        results = []

        def score_fn(work, batch, refs, uv, depths, normals, stride=4):
            return photometric_observations(
                work, batch, refs, uv, depths, normals, cfg=self.sampler.cfg, stride=stride
            )

        for start in range(0, len(output["points"]), self.chunk):
            sl = slice(start, start + self.chunk)
            results.append(
                self.verifier(
                    inputs,
                    b,
                    aux["ref_index"][sl],
                    aux["ref_uv"][sl],
                    aux["reference_depth"][sl],
                    aux.get("oriented_patch_normal", output["normals"])[sl],
                    score_fn,
                    candidate_depths=aux["candidate_depths"][sl],
                    footprint_m=output["scales"][sl],
                    stride=self.strides[stage],
                )
            )
        result = {
            k: torch.cat([r[k] for r in results])
            for k, v in results[0].items()
            if torch.is_tensor(v)
        }
        # 先有独立几何与state头确认，才把这些预测命中用于遮挡/矛盾复核。
        from jar.ray_evidence import build_ray_evidence

        trusted = dict(output)
        trusted["valid"] = result["valid"] & (output["state_logits"].argmax(-1) == 1)
        trusted["state"] = torch.where(trusted["valid"], 1, 2)
        trusted["support"] = result["support"]
        trusted["confidence"] = result["confidence"] * trusted["valid"]
        hits = build_ray_evidence(trusted, inputs, b, self.roi)
        result = self.verifier.recheck_visibility(
            result, inputs, b, aux["ref_index"], aux["reference_depth"], output["points"], hits
        )
        accepted = result["valid"]
        # 保留原有可学习state头；几何验收不能靠学习权重置零逃避。
        output["valid"] = accepted & (output["state_logits"].argmax(-1) == 1)
        output["state"] = torch.where(output["valid"], 1, 2)
        output["support"] = result["support"]
        output["confidence"] = (
            result["confidence"]
            * output["state_logits"].float().softmax(-1)[:, 1]
            * output["valid"]
        )
        aux.update({"mf_" + k: v for k, v in result.items()})
        aux["depth_sigma_m"] = result["depth_sigma"]
        aux["depth_sigma"] = result["depth_sigma"]
        aux["evidence"] = accepted
        return output

    def _fuse_scene(self, stage, inputs, b):
        if not self.mf.get("resolution_fusion", True):
            return super()._fuse_scene(stage, inputs, b)
        from .fusion import resolution_aware_fuse

        return resolution_aware_fuse(dict(stage), cell_size=self.fusion_cell_size)

    def _ray_evidence(self, stage, inputs, b):
        from .fusion import build_multifocal_ray_evidence

        return build_multifocal_ray_evidence(stage, inputs, b, self.roi)

    def forward(self, inputs, pyramid, coarse, teacher=None):
        work = dict(inputs)
        if self.birth_scorer is not None:
            work["_mf_birth_logits"] = self.birth_scorer(pyramid)
        out = super().forward(work, pyramid, coarse, teacher)
        if self.birth_scorer is not None:
            out["mf_birth_logits"] = work["_mf_birth_logits"]
        for scene in out["scenes"]:
            scene["aux"]["architecture"] = "mf_encoder_token_hierarchy"
        return out
