# TRAIN 短基线再生成：已实现与适用边界

日期：2026-11-02。此原型没有接入正在运行的训练、模型或正式评测，不改变V6。

`synthetic_pair.py`提供三个独立函数：

- `forward_splat`：从参考RGB和光轴深度构造点，精确投影至同K同R的新相机，再最近栅格投放；`scatter_reduce(amin)`完成深度z-buffer，冲突保留近表面。空洞RGB/深度为0，`image_mask=false`，不是空闲空间真值。
- `synthesize_pair`：每样本从预设`LogUniform(.03,.3)×domain.depth_scale`选择基线，符号独立随机，不根据当前GT范围选基线。将一个source槽替换为新相机，可关闭其他source得到真实的一个参考+一个虚拟源。两套编码输入来自同一主图的equal-pixels缩放；重算匹配掩码，保持K/T方向与单位一致。不原地修改inputs/targets。
- `correspondence_targets`：显式将虚拟source深度和有效性映射回目标槽，保留reference原GT，清除虚拟source的旧语义/实例/法线等标签。调用者不能继续沿用该槽原相机的GT。

模型推理不得调用这些函数。使用GT再生成仅用于TRAIN；返回的source监督和对应索引也不进入网络输入。`reference_pixel_index`说明每个生成source像素来自哪个reference点，`projected_uv`保存栅格化前的实际投影位置。

## 实际执行

```bash
official_methods/.venv/bin/python -m pytest -q \
  research/v7_domain_20261102/test_synthetic_pair.py

# 指定物理GPU7；输出只进入已有数据盘软链接。
official_methods/.venv/bin/python \
  research/v7_domain_20261102/smoke_synthetic_pair.py
```

CPU **4 passed，1.38秒**。纹理背景平面与近前景遮挡测试实际验证：碰撞选近点、被遮挡后新露出的区域保持洞、源标签显式替换、不改变原数据、改变长度单位后投影相同、基线不依赖GT、亚像素投影与最近栅格误差、关闭增强时输入不变。

GPU7为RTX5090，真实B=2分别处理TartanAir TRAIN前两组和新CARLA TRAIN前两组，命令退出0。没有读取VAL/TEST，没有模型前向或新精度结论。

| TRAIN样本 | 数据单位基线 | 生成源有效覆盖 | 连续投影往返最大误差 | 最近栅格最大误差 |
|---|---:|---:|---:|---:|
| TartanAir 0 | 0.126101 m | 70.94% | 0.0000588 px | 0.49995 px |
| TartanAir 1 | 0.099754 m | 71.89% | 0.0000626 px | 0.49997 px |
| CARLA 0 | 0.037838 m | 90.42% | 0.0000675 px | 0.50000 px |
| CARLA 1 | 0.107446 m | 90.00% | 0.0000736 px | 0.50000 px |

完整记录和四张图位于`runs/active/v7_domain_20261102/synthetic_pair_smoke/`，`completed.json`记录TRAIN样本ID、实际清单哈希、源码哈希、碰撞数和相机变换。森林和CARLA图已实际打开检查。

## 当前不宜直接作为亚像素精度训练的数据

精确几何投影通过，并不表示最终栅格图像是完美渲染。nearest点投放存在最多半像素的水平量化，恰可能与ETH弱视差的有效信号同量级：真实小于半像素的位移可能被量化为相同颜色格点。因此本实现适合验证几何契约、源缺失/遮挡的取舍，却不宜直接用来证明亚像素匹配提升。

森林图也出现明显洞；这些洞有原GT无效、视角新露出与离散点投放三种来源，不能全部叫真实遮挡。填补它们并赋予有效mask会制造伪对应，本轮没有这样做。

若后续将其用于弱视差精细学习，先实现遮挡受控的连续面片渲染或经验证的双线性重建，并用已知解析纹理平面验证不同亚像素位移的光度关系。双线性颜色混合也必须避免跨前后表面；不能仅因图片好看便标作真实一致几何。这是本轮原型已发现的不足，不应在正式全量训练中未经验证默认启用。
