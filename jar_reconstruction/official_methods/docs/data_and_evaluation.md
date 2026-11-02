# 四目数据与评价契约

本集成复用项目的 `jar.data.CarlaFrameDataset` 和 `jar.synthetic`，不会另造一套互不兼容的 CARLA 格式。所有配置的真实根目录、训练/验证/测试清单默认是空字符串。普通命令会报错；只有明确写 `--synthetic` 才渲染小场景。

## 输入与坐标

一个 batch 分为 `inputs`、`targets`、`metadata`。网络推理调用只有 `forward(inputs)`。

| 项目字段 | 形状 | 含义 |
|---|---|---|
| `inputs.rgb` | B×4×3×H×W | 同时刻 RGB，浮点 0–1 |
| `inputs.K` | B×4×3×3 | 原图整数像素中心内参 |
| `inputs.T_cam_from_rig` | B×4×4×4 | rig 点变换到 CV 相机；最后两个轴是矩阵行、列 |
| `inputs.view_mask` | B×4 | 四个视角是否存在；共同协议要求全部有效 |
| `inputs.image_mask` | B×4×H×W | RGB 有效区域，不是深度标签有效性 |
| `metadata.sample_id` | 每帧一个字符串 | `sequence/frame_id` |

也提供 `canonical_inputs` 转换用户指定的 `images/intrinsics/T_cam_to_rig` 格式，其中外参只在入口求逆一次。

内部 rig 为右手系：x 向前、y 向左、z 向上。CV 相机为 x 向右、y 向下、z 向前。坐标和深度以米计。CARLA 的 y 向右，因此不能直接把 CARLA transform 当项目外参；使用已有 `jar.data.carla_transform_to_cv`。每台相机都保存实际 K/T，不根据建议安装位置重新猜测标定。

深度标签是相机光轴方向的 Z-depth。若采集结果是沿射线的距离，必须先根据 K 换成 Z-depth。原始 CARLA RGB24 深度字节与彩色可视化图不是同一回事；复用的解码器要求明确 `raw_sensor_bytes=true`、通道顺序及 `depth_type`，不能凭显示效果判断。

缩放使用半像素中心对应关系，裁剪同步移动主点。`data.py` 还提供纯像素翻转的几何函数并有重投影测试；默认公共流水线不翻转。Niantic 方法训练时单独沿用官方的图像/相机反射增强，不叠加公共翻转。三个深度模型的官方像素中心为 0.5，本项目为 0；适配器显式换算，返回 `depth_K` 后再转换回公共约定。

## JSONL 清单

每行一帧；四台相机必须具有相同 `frame_id` 和时间戳。以下展示一台相机字段，实际 `cameras` 数组恰好包含四项：

```json
{
  "sequence": "map01_route03",
  "frame_id": "000120",
  "timestamp": 6.0,
  "cameras": [{
    "frame_id": "000120", "timestamp": 6.0,
    "rgb": "rgb/cam0/000120.png",
    "K": [[640,0,639.5],[0,640,359.5],[0,0,1]],
    "T_cam_from_rig": [[0,-1,0,0],[0,0,-1,1.6],[1,0,0,-1.5],[0,0,0,1]],
    "valid_mask": "masks/cam0/000120.npy",
    "targets": {
      "frame_id": "000120",
      "depth": "depth/cam0/000120.npy",
      "depth_encoding": "metric_npy", "depth_type": "optical_z",
      "depth_valid": "depth_valid/cam0/000120.npy",
      "semantic": "semantic/cam0/000120.npy"
    }
  }],
  "occupancy_targets": {"frame_id": "000120", "path": "occupancy/000120.npz"}
}
```

`valid_mask`、语义和占据标签是可选项。推理模式不读取 `targets` 或 `occupancy_targets` 文件，标签不存在也不能影响推理。当前官方深度网络没有带孔源图的原生掩码聚合机制，因此拒绝非完整有效矩形；可先用 `image_crop: [top,left,height,width]` 裁去无效边缘，随后同步缩放。不能用零图补缺失相机。

MVSAnywhere 可在每台相机的 `targets.sky` 指定 `.npy` 天空标签：与原始 RGB 同尺寸，浮点 0=非天空、1=天空、NaN=未知。缺失视角或缺失样本填 NaN，只跳过相应监督，不把未知当作非天空。该标签随图像裁剪和缩放，推理不读取。

占据 npz：`state[X,Y,Z]` 取 0=已观测 free、1=可见表面命中 occupied、2=unknown；`grid_origin[3]` 是最小边界，`voxel_size[3]` 为米制边长；可选 `semantics[X,Y,Z]`，未知语义为 -1。**这套可见表面占据标签与 nuScenes 完整占据协议不同**。若未来获得完整实体标注，必须另建清单/协议，不能把两类指标混用。

采集时按方法准备以下标签：

| 方法 | 必需监督 | 可选监督或额外要求 |
|---|---|---|
| Fast-MVSNet | 各相机米制 Z-depth 与有效掩码 | 不需要语义 |
| SimpleRecon | 各相机 Z-depth；法线由深度和标定生成 | 多视角几何损失同帧使用其余相机深度 |
| MVSAnywhere | 各相机 Z-depth | 天空标签可选；缺失时对应项跳过 |
| GaussianFormer-2 | 同一固定网格的观测状态与命中语义 | 17 个非空语义槽位加 free；必须保存 CARLA→模型类别映射，不能直接使用原始 CARLA ID |
| OctreeOcc | 公开原流程还依赖语义结构初始化文件 | 初始化生成链缺失，现阶段暂停整模型 |

GaussianFormer-2 的 unknown 是**监督有效性**，不是新增的网络预测类别；原头在全 ROI 作 free/occupied 预测，未知 GT 上的输出保留并单独统计。若只有深度，可显式设 `data.derive_observed_occupancy=true` 派生观测状态，但这不会凭空提供语义类别。正式语义 IoU 必须配置统一的 `evaluation.semantic_classes` 并确保类别映射一致。

`occupancy_labels.observed_occupancy` 可由深度生成保守标签：表面命中体素优先保留；同一相机中体素八角均在首个命中前且可见，才判为 free；遮挡后方继续 unknown。它不会把物体包围盒填成实心占据。语义冲突保留几何命中但忽略语义。

按地图/路线/序列划分数据，不随机混合相邻帧。入口检查同一序列不能同时出现在训练、验证或测试；同一地图不同序列是否相近仍需采集清单作者控制。

## 方法能力与后处理

三种深度方法返回原生结果 `native`、深度 `depth`、深度网格内参 `depth_K` 和 `reference_indices`。Fast-MVSNet 只有粗尺度概率，不把它伪装成最终深度置信度。SimpleRecon/MVSAnywhere 没有通用置信度、占据和语义占据头。GaussianFormer-2 的原生输出为高斯表示与占据，不强行换成共同的深度预测头。OctreeOcc 受公开源码缺失阻塞，不能声称有可用整模型输出。

后处理文件 `postprocess.py` 独立保存：按 K 把原生深度映射到 RGB 网格，然后反投影成点云。双线性重采样可能在深度断边混合前后表面，这属于评价转换局限。这里没有冒充官方 TSDF/fusibile 融合，也没有把占据中心直接当作精确表面。

## 指标定义

- 深度：统一数据有效区域和配置范围，AbsRel、米制 RMSE/MAE 在有限正深度预测上计算，并强制同时报告覆盖率；越出搜索范围的有限正预测仍计误差。阈值准确率的分母是全部有效 GT，缺失预测算失败。
- 表面：先在同一可评价观测像素区域反投影预测和 GT，再限制共同 rig ROI。固定像素顺序等距采样，默认各最多 8192 点。Accuracy 是预测到 GT 的最近欧氏距离均值；Completeness 反向；Chamfer-L1 取两者均值，单位米，**不是平方距离之和**。F-score 默认阈值 0.05/0.1/0.2 米；有 GT 但没有预测时记零分，没有可评价 GT 或距离均值无定义时写 null。
- 占据：仅相同网格、类别映射和标签定义才能算 IoU/mIoU；GT unknown 不当 free。预测 unknown 在 GT 已知处不会被剔除。另报错误 free、unknown 比例和对应计数。类别并集为空时不虚构满分，mIoU 只平均有并集的类别并保留各类值。

表面 GT 掩码仅用于评价，不会反馈网络选点、深度范围或推理。原生预测文件和评价派生点云分开保存。

## 时间、显存与预训练公平性

评价配置的 `reference_indices=[0,1,2,3]` 强制轮换四个参考图，各自只使用其余三张源图。原生网络 hook 累加所有调用时间；完整管线另计输入搬运、适配和后处理。预热后做 GPU 同步。显存记录原生阶段峰值、原生计算相对进入时的新增峰值、管线峰值；所有值是 PyTorch allocated memory，非 nvidia-smi 的整个进程占用。磁盘 I/O 和使用 GT 的指标计算不计入计算管线，范围在 JSON 明示。

不能把单参考图时间和四目完整场景时间混用；batch>1 时也不能把 batch 时间直接称为单场景时间。合成小图只做可运行性验收，不作为真实吞吐榜。

正式实验要另记录预训练数据与权重 SHA256、每卡 batch、总图像数、更新次数、总 GPU 小时。MVSAnywhere 在 CARLA 微调后必须标作微调结果；没有加载原预训练权重的测试不是原模型零样本评测。
