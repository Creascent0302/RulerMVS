# 数据契约与 CARLA 接入说明

本项目不会启动 CARLA 进程。所有真实数据路径默认均为空字符串。真实模式要求填写 `data.root`，以及所选的 `train_manifest`、`val_manifest` 或 `test_manifest`；合成模式必须显式传入 `--synthetic`。合成数据在固定平面和实体方体上改变共用的世界坐标纹理，不是为每台相机独立生成随机图像。它用于几何和训练管线检查，不能代表真实基准性能。

## 输入与标签分离

每个数据样本包含 `inputs`、`targets` 和 `metadata`。推理只接收 `inputs`，内容如下：

| 字段 | 每帧形状 | 含义 |
|---|---|---|
| `rgb` | N,3,H,W | [0,1] 浮点 RGB，不进行逐帧自动曝光归一化 |
| `K` | N,3,3 | 当前图像分辨率对应的真实针孔内参 |
| `T_cam_from_rig` | N,4,4 | 固定的米制 rig 到 CV 相机坐标变换 |
| `view_mask` | N | 相机是否存在；false 项被忽略 |
| `image_mask` | N,H,W | 预处理或填充后仍属于真实图像的像素 |

`collate_frames` 添加批次维，并用 false 掩码、零图像和安全的单位矩阵填充缺失相机。同一批次的所有帧必须具有相同图像尺寸。标签拥有独立掩码：相机可以提供 RGB，但没有深度标签。`targets.depth` 始终是**以米为单位的光轴深度**，形状为 N,H,W；`depth_valid` 为布尔值。可选的 rig 坐标法线 `normals` 为 N,3,H,W，`semantic` 类别 ID 为 N,H,W，-1 表示忽略。深度、语义字段均不得放进 inputs，模型不接收 targets 字典。

## 逐相机深度与直接解码器的多参考训练

新采集的每台相机都可以有自己的逐帧深度文件，旧数据只有部分相机带深度也可以继续使用。每台相机在自己的 `targets` 中声明 `depth`、`depth_type`、`depth_encoding`、可选 `depth_scale` 和 `depth_valid`，**这些字段不是全局设置**。例如，左相机可以声明 `optical_z`，上相机声明 `ray_range`；加载器分别用各自原始 `K` 转换，最后统一为米制光轴深度。必须先核实实际编码，不应靠文件扩展名猜测。所有深度及有效掩码必须与对应相机原始 RGB 像素格严格一致。

缺深度时省略该相机的 `targets.depth`，RGB 仍参与匹配；缺 RGB 时从清单省略该相机，或明确标为 `valid: false`，并设置 `camera.allow_missing: true`。批处理后的 `targets.depth/depth_valid` 为 `[B,N,H,W]`，缺标签位置的 `depth_valid=false`，不会复制上相机深度填充其他相机。这里仍是 RGB 与标定输入的重建，新增深度仅用于监督与离线评测，推理不要求深度传感器。

使用 [ruler_direct_multicamera.json](../configs/ruler_direct_multicamera.json) 作为独立配置模板，填写自己的四个稳定 `camera.wide_ids` 和数据清单。真实路径默认全部为空。`direct.reference_indices` 按配置中的相机顺序选择输出：默认模板 `[0,1,2,3]` 输出四目；`[2]` 只输出上相机；`[3]` 只输出下相机。它们共用编码器与解码器参数，不会用其他相机代替缺失的指定参考图；缺失输出保持无效。多参考共享匹配特征，单图 ViT 一次批量编码选中的有效相机，解码逐参考执行，所以四参考耗时必须单独测量。

10月24日新增的候选保留与多尺度空间版本使用 [train_multicamera.json](../configs/multiview_20261024/train_multicamera.json)。它与上述旧空间卷积模板分别保留，便于同预算比较；该配置的真实路径也全部为空。新版本的初始化文件为 `runs/active/multiview_20261024/full_rgb_final/scale_transport_complete.pt`。本轮上相机 TEST MAE 比同预算旧结构有所改善，但仍未追平 MVSAnywhere；其他相机的真实精度尚未验证，不能因为入口支持就称泛化达标。

训练时，每个微批次对有监督的“样本—相机”分别计算损失。完全缺标签的项跳过；缺标签的微批次返回可微零损失，整个优化窗口的所有进程均无可用监督时不更新参数。DDP会汇总各进程有效监督相机数，再补偿其梯度平均，因此同一微批次跨进程的有效“样本—相机”等权；没有标签的进程不会稀释有效梯度。梯度累积仍按微批次样本数加权，**不同微批次有效相机数变化时，不等于整个累积窗口所有相机严格等权**。比较实验须固定批大小、进程数及缺失标签策略，并记录有效监督相机数。

在项目根目录执行以下命令。初始化文件必须是含两个冻结编码器和解码器的完整 `ruler_direct` checkpoint，不能直接传只保存解码器的 `best.pt`；模板与下面已有完整权重的解码结构相同，仅增加参考调度。结构不同的初始化会明确报错，`--resume` 还要求架构、参考调度、数据和损失合同一致。

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction
official_methods/.venv/bin/python train.py \
  --config configs/ruler_direct_multicamera.json \
  --init-checkpoint runs/active/ruler_candidate_20261023/spatial_full_rgb_dev25/spatial_volume_complete.pt \
  --device cuda:0 --output runs/active/ruler_multicamera_train
```

```bash
official_methods/.venv/bin/python train.py \
  --config configs/ruler_direct_multicamera.json \
  --resume runs/active/ruler_multicamera_train/last.pt \
  --device cuda:0 --output runs/active/ruler_multicamera_train

official_methods/.venv/bin/python infer.py \
  --config configs/ruler_direct_multicamera.json \
  --checkpoint runs/active/ruler_multicamera_train/last.pt \
  --device cuda:0 --output runs/active/ruler_multicamera_infer
```

`runs/active` 已指向数据盘；新训练输出应继续放在该软链接下。正式训练前必须填写真实数据路径；只有额外加 `--synthetic` 才运行合成场景。真实数据尚未到位，以上新数据训练没有执行；已执行的单卡、双卡与恢复测试使用几何合成小场景。

新结构完成数据清单后，可用以下命令；GPU数量属于运行资源设置，模型及训练参数仍由配置文件控制。已实测双卡冒烟，不把八卡命令当作已经完成八卡验证。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
official_methods/.venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 train.py \
  --config configs/multiview_20261024/train_multicamera.json \
  --init-checkpoint runs/active/multiview_20261024/full_rgb_final/scale_transport_complete.pt \
  --device cuda --output runs/active/ruler_scale_multicamera_train
```

采集前请同时保存深度导出脚本和一个已知距离平墙的小样例，并保留原始传感器编码或明确的解码元数据。当前旧数据的解码来源仍未最终确认，不宜让新数据延续这个歧义。新增四目深度增加的是不同视角的监督，不能替代新地图、路线、天气和遮挡条件的覆盖；同帧四张图必须一起划入同一数据划分。

原生输出 `direct_by_reference` 按规范相机索引存放各参考结果，`direct_camera_indices[B,R]` 显式给出它们在调用者原始输入中的槽位，避免相机重排后监督错位。公共 `dense_depth`、`dense_valid`、`dense_confidence`、`dense_support` 等张量恢复到调用者输入顺序。`dense_support` 是所选候选在几何上可投影、可采样的源相机数，**不是通过遮挡校验的表面支持数**；`dense_observed` 仍为 false，`dense_state` 仍为 unknown。新增其他相机监督的可运行性不等于这些相机已经获得经实测确认的泛化精度。

## JSONL 清单

每行表示一次同步相机阵列曝光。以下路径是格式示例，不是已配置的真实路径：

```json
{"sequence":"TownRoute01","frame_id":108,"timestamp":5.4,"cameras":[{"id":"front_left","frame_id":108,"timestamp":5.4,"rgb":"rgb/108_left.png","K":[[640,0,639.5],[0,640,359.5],[0,0,1]],"T_cam_from_rig":[[0,-1,0,0.75],[0,0,-1,1.6],[1,0,0,-1.5],[0,0,0,1]],"targets":{"frame_id":108,"depth":"depth/108_left.npy","depth_encoding":"metric_npy","depth_type":"optical_z","max_valid_depth":80.0,"semantic":"semantic/108_left.npy"}}]}
```

各相机 frame ID、时间戳必须与所在帧一致，深度标签的 frame ID 也必须一致。同步必须依据 CARLA frame ID，不能依据传感器消息的实际到达顺序。缺失的可选标签直接省略，不得填入猜测的 GT。图像由 Pillow 读取，深度 `.npy` 使用 `allow_pickle=False` 加载。米制深度必须来自原始测量，不能使用归一化可视化结果。

原始编码 PNG 深度可声明 `depth_encoding: "carla_rgb24"`、`raw_sensor_bytes: true`、`channel_order: "RGB"`、`far: 1000.0`，并明确指定 `depth_type`。可选布尔 `.npy` 字段 `depth_valid` 用于标注无效传感器像素。语义文件保存原始整数 ID，不使用调色板 RGB 预览图。可选 `instance` 路径读取原始整数 `.npy` 实例 ID 并放入 targets；实例 ID 会经过批处理传递，但当前基线没有实现实例专用损失。

在编写清单之前，先按地图、路线、序列划分训练、验证和测试集。同一序列的相邻帧或不同相机子集不得跨集合出现。每次采集还应保存模拟器版本、地图、天气、曝光设置、传感器蓝图、实际标定和同步策略。

## 坐标转换与深度核验

内部 rig 坐标为右手系：**x 向前、y 向左、z 向上**。CV 相机坐标为 **x 向右、y 向下、z 向前**。CARLA 坐标为 **x 向前、y 向右、z 向上**。令 M 表示 CARLA 相机到车体的齐次变换，则转换为：

```
T_cam_from_rig = C @ inverse(M) @ S
S = diag(1,-1,1,1)
C = [[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]]
```

`jar.data.carla_transform_to_cv` 实现这一转换，并正确处理相机平移。不能只翻转平移，也不能只修改旋转的某一个轴。矩阵测试验证输出是合法旋转，并检查车体前向射线转换后指向相机 +z。

初始 CARLA 相机位置可采用 `(1.5,-0.75,1.60)`、`(1.5,0.75,1.60)`、`(1.5,0,2.15)`、`(1.5,0,1.60)`；传感器姿态为零，图像为 1280×720，水平 FOV 为 90°。这些位置对应内部 y 坐标 +0.75、-0.75、0、0。采集时必须核验实际车体遮挡、畸变设置以及保存的 K/T。

该初始方案尚未被认定为最优。相机布局应根据目标距离、遮挡与实际重叠视场选择。旋转后必须保存实际新T；不要只改姿态角而沿用旧外参。

示例主点使用整数像素中心约定 `(W-1)/2,(H-1)/2`；应保存实际渲染器标定，不能事后凭记忆中的 FOV 重建内参。必须禁用镜头畸变，或明确去畸变至保存的 K 对应图像。

原始 24 位深度解码公式为
`meters = 1000 * (R + 256*G + 65536*B) / (256**3 - 1)`。
CARLA 内存中的原始图像是 BGRA；保存为 RGB PNG 后，通道顺序不同。`decode_carla_depth` 要求显式指定通道顺序。仅凭解码公式，无法判定某条导出管线保存的是光轴深度还是射线距离。每份清单必须明确声明 `optical_z` 或 `ray_range`；后者在训练前除以 `norm(inv(K) @ [u,v,1])`，转换为光轴深度。

在实际安装的模拟器版本上，采集一个已知的正面平墙：光轴深度在墙面上应保持恒定，射线距离则朝图像角落增大。大规模采集前，再核验一个已知偏轴点以及相机到世界坐标的往返转换。不能根据灰度预览推断深度类型。当前尚未执行针对某个 CARLA 版本的端到端采集检查。

编码、BGRA 存储及坐标约定参考已核查的官方来源：[CARLA 传感器文档](https://carla.readthedocs.io/en/latest/ref_sensors/)与[CARLA ROS bridge 相机解码实现](https://github.com/carla-simulator/ros-bridge/blob/master/carla_ros_bridge/camera.py)。

## 缩放、裁剪与特征格点约定

图像与标签必须执行一致的几何变换。使用 `align_corners=False` 缩放图像时，焦距乘以缩放倍数，主点变为 `scale*(c+0.5)-0.5`；随后裁剪时，减去左、上裁剪偏移。使用 `jar.geometry.resize_crop_intrinsics` 更新内参。无效掩码与类别 ID 适合最近邻插值；深度边缘不能跨表面平均。数据集会拒绝不兼容的图像与标签格点，不会静默缩放。

奇数卷积核带填充的 CNN，其 stride s 特征中心对应原图坐标 `s*j`；这与直接缩放图像是不同的约定。FPN 采样先使用 `u/s`，再执行 `grid_sample(align_corners=False)` 坐标归一化。图像裁剪不得修改外参 T。

## 监督与评估语义

`target_cloud` 反投影已观测到的首次深度命中。`label_query_points` 将首碰之前的点标为 free，将命中附近窄带标为 surface，将命中之后标为 unknown。缺失视图和像素不提供标签。轮廓附近，如果另一条射线把同一小邻域判为空闲，则表面证据优先。

`NodeTeacher` 将**全部**深度命中分配到每个已有树分辨率，因此单个已观测细表面也能保留粗单元的 surface 标签。它不会平均表面/空闲比例，也不会填满物体包围盒。free 单元要求中心和全部角点都有已观测空闲证据，并且盒内没有命中。这是有限射线监督，不保证采样点之间未观测体积的状态。

节点平面几何使用盒内全部命中的一阶、二阶矩：平面残差平方的期望等于均值点处残差的平方加上沿法向投影的协方差。因此，仅穿过粗单元平均命中点不能满足平面损失。语义投票使用真实盒内命中，而不是可能落在两层表面间空隙中的均值点。

Teacher forcing 是可选且仅限训练的功能，提供状态和 split 标签，不改变标定，eval 时禁止使用，默认关闭。没有深度标签时，A 的预测头仍可使用像素对齐的语义标签；其他依赖深度的项返回保持计算图连接的零。B/C 的语义投票需要深度来验证哪些视图看到了预测单元的 GT 表面。几何、split、状态和可见对应关系标签均由深度推导；法线和语义损失使用各自可用的标签。表面训练以标签可用性作为掩码，不受当前模型预测有效性的限制，使初始不确定的查询也能学习。

缺少显式法线标签时，通过米制深度反投影点的有限差分，推导朝向相机的 rig 法线。缺少邻点、图像边界和深度不连续区域均被剔除，避免跨遮挡平滑标签；这样在 CARLA 仅有深度的情况下，表面法线头仍可训练。

无匹配监督同样遵守候选的结构掩码，包括图像边界、纹理和射线导数。恒定图像或纯旋转输入不会被要求选择一个已被屏蔽、不可能成立的深度类别。缺少深度标签不等于无匹配。如果一帧中存在完全没有深度标签的相机，已知 surface 和 free 标签仍可使用，但省略 unknown 负样本，因为其原因可能只是缺少标注。置信度头对停止梯度的米制表面准确性接受监督，无法通过降低自身置信度减小几何损失。表面细化同时接受累计及逐步米制更新监督，以及经过深度可见性核验的逐视图对应偏移监督。

指标保留 unknown 预测在已观测区域状态准确率和表面 IoU 的分母中。稀疏点深度 MAE/RMSE 必须**同时报告像素覆盖率**：无预测时深度误差为 null、覆盖率为零，不能记成零误差的完美结果。F-score 使用有界、确定性的点采样，并报告样本数量和阈值；可选 ROI 裁剪对预测表面和 GT 表面一致应用。

`output_element_*` 状态指标依赖输出表示形式；用于论文比较时，应在相同的预定义射线/ROI 探针上，以相同表面采样密度评估全部模型。分别报告真实表面上错误 free，以及 unknown 区域中的 free 预测。物体包围盒和不可见背面不作为已观测占用真值。
