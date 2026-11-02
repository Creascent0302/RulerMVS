# MF-ET-JAR 的相机数据、命令行与合成场景

这份说明对应 `jar/multifocal/data.py`、`options.py` 和 `synthetic.py`。完整架构见 [架构说明](history/architecture.md)。没有 `camera` 配置的原项目仍使用原数据加载器；所有新增真实数据根路径和 manifest 路径默认为空，缺失时明确报错。仅显式 `--synthetic` 才渲染解析场景。

## 1. 用身份区分相机

配置中将架构和相机模式分开。下面的 ID 是示例名字，采集系统可以改名，但一个实验中的身份必须稳定。

```json
"camera": {
  "mode": "wide4_tele2",
  "wide_ids": ["wide_left", "wide_right", "wide_up", "wide_down"],
  "tele_ids": ["tele_left", "tele_right"],
  "allow_missing": false,
  "disabled_ids": []
}
```

`wide4` 只选择四个广角 ID；`wide4_tele2` 再加入两个长焦 ID。加载器按配置顺序排列，manifest 的数组顺序可以不同。四个广角的次序不因增加长焦改变，以保留原有查询探索相位。模型入口也验证身份顺序，不能省略 ID 后靠数组位置猜角色。

每个 JSONL 行仍是一帧，与原接口一致：`sequence`、`frame_id`、`timestamp` 和 `cameras`。每个相机项目最少包含：

```json
{
  "camera_id": "wide_left",
  "role": "wide",
  "frame_id": 3,
  "timestamp": 1.0,
  "rgb": "sequence_01/000003/wide_left.png",
  "K": [[640, 0, 639.5], [0, 640, 359.5], [0, 0, 1]],
  "T_cam_from_rig": [[0, -1, 0, -0.6], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
  "targets": {
    "frame_id": 3,
    "depth": "sequence_01/000003/wide_left.depth.npy",
    "depth_type": "optical_z"
  }
}
```

这是单台相机条目的格式示例，实际 `cameras` 应记录所选模式需要的所有相机，各自使用实际测得的 K 和 T。长焦的 `role` 为 `tele`；省略 role 时以配置中的稳定 ID 分组为准。不要把该示例标定照搬到真实采集数据。

缺失相机默认报错。只有 `camera.allow_missing=true` 才允许降级：缺失 ID 或 `valid=false` 的条目变成无效槽，RGB 为零，掩码为 false，K/T 为安全单位阵，原尺寸记为 `[0,0]`。这些槽不能作为匹配证据，不复制其他相机的图片。整帧全部缺失时数据加载报错，因为没有任何真实图片可供重建。

## 2. 尺寸、坐标与标签

阵列坐标使用右手系 **x 前、y 左、z 上**，相机坐标使用 **x 右、y 下、z 前**。`T_cam_from_rig` 把阵列坐标点变到相机坐标，平移和深度单位均为米。CARLA 原坐标转换沿用 `jar.data.carla_transform_to_cv`。

广角和长焦可以有不同图像尺寸。加载器仅在右侧和下侧补零，因此原 K 不需要改变；`image_sizes` 保存补零前的真实尺寸。`image_mask` 标识有效像素，可额外用相机条目的 `image_valid_mask` 指向同尺寸布尔 `.npy`。编码器按真实尺寸分组运行，避免补零面积改变归一化统计。后续模块必须继续检查掩码。

| 批量输入字段 | 形状/含义 |
|---|---|
| `rgb` | `[B,N,3,Hpad,Wpad]`，0–1 浮点 RGB |
| `K` | `[B,N,3,3]`，原图像素坐标内参 |
| `T_cam_from_rig` | `[B,N,4,4]`，米制刚体变换 |
| `camera_ids` | B 行字符串列表，每行 N 个稳定 ID |
| `camera_roles` | `long[B,N]`，0 广角、1 长焦 |
| `camera_valid`、`view_mask` | `bool[B,N]`，两者必须相同 |
| `image_sizes` | `long[B,N,2]`，顺序为 H、W |
| `image_mask` | `bool[B,N,Hpad,Wpad]`，补零和缺失区域必须 false |

监督仍在独立 `targets` 中，包括 `depth`、`depth_valid`，可选 `semantic`、`instance`；合成数据还提供阵列坐标法线。标签与 RGB 使用同一原始像素网格，补零标签没有有效深度。深度必须声明 `optical_z`（相机光轴深度）或 `ray_range`（沿射线距离），后一种读取时转换为光轴深度。原始 CARLA 24 位编码仍使用原项目的显式格式标记与解码函数，不能使用可视化图。

推理调用 `load_targets=False`，不会打开 GT 深度/语义文件。相机帧号和时间戳仍接受同步校验。缩放和裁剪应使用正确的内参变换；设置 `data.image_size=[H,W]` 时会显式缩放并同步更新 K；未设置时保留原始尺寸。

## 3. 一个架构的独立开关

在 JSON 中设置 `camera.mode="wide4"` 选择四目，设置 `"wide4_tele2"` 选择四广角加双长焦。相机模式与多焦距算法独立：六目下可设 `model.multifocal.enabled=false` 测试基础解码器，或只关闭其中一个模块。完整字段表见 [README](../README.md#配置文件与独立消融开关)。

主动禁用使用 `camera.disabled_ids=["tele_left"]`，可列出多个实际 ID。加载器不读取这些相机的 RGB/标签，固定槽位保持无效；API 输入也在编码前屏蔽。禁用左长焦不会自动允许右相机缺失；真实采集缺失由 `camera.allow_missing` 独立控制。数量由模式和 ID 确定，无需另设相机数量。

```bash
# 先在完整 JSON 配置中修改所需开关，再启动。
python infer.py --config configs/mf_et_jar_tiny.json \
  --synthetic --device cpu --limit 1 --output runs/mf/ablation
```

不再使用命令行覆盖模型参数。最终配置写入 `config.json`；checkpoint 的 `model_mode` 记录开关；续训检查相机身份、顺序、禁用列表和缺失策略。新实验用 `--init-checkpoint` 初始化可兼容权重，继续同一实验用 `--resume`，二者不能同时使用。

`configs/mf_et_jar.json` 广角深度范围为 2–24 米，长焦为 10–150 米。`configs/mf_et_jar_tiny.json` 降低通道、候选和查询数量，保留远距 ROI。小模型直接选择该文件，没有 `--tiny` 覆盖规则。

## 4. 无真实数据时能测试什么

解析场景默认包括 120 米远墙、45 米细杆、24 米遮挡箱和地面。所有相机对同一个三维场景做首命中射线求交，表面使用共享世界坐标纹理；不是各自生成随机图片。默认广角 128×192、长焦 256×384，长焦水平视场更窄。tiny 中广角 32×48、长焦 48×64。

可配置 `synthetic.scene` 的墙、地面和 `boxes`，`tele_height/tele_width`、`tele_focal_multiplier`、`baseline_scale`、`texture`、`missing_camera_ids`。尺寸变更会重新计算相机内参。`baseline_scale=0` 用于退化测试，不能据此声称有米制深度证据。相同设置下四目模式与六目模式前四张图片严格一致。

数据测试命令：

```bash
python -m unittest tests.test_multifocal_data tests.test_data_losses tests.test_inference -v
```

这些测试检查接口、身份、标签隔离和解析投影；合成短测试不证明远距精度超过四目基线。实际执行结果与限制见主 README。
