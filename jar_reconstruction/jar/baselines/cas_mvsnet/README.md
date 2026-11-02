# CasMVSNet

`adapter.py` 保存三阶段作者网络的项目适配器和旧合成测试损失，`upstream/` 保存作者代码实体。网络仍是多尺度特征、三阶段方差代价体、三维卷积和深度回归，当前整理没有更换任何模块。

来源、提交和兼容修改见 [upstream/provenance.json](upstream/provenance.json)，许可证见 [upstream/LICENSE](upstream/LICENSE)。这是作者网络接入 CARLA，不等于复现官方 DTU 论文精度。

```python
from jar.baselines.cas_mvsnet import OfficialCasMVSNet, OfficialCasMVSLoss
```

旧 `jar.official_casmvs` 导入仍然有效；旧 `third_party.cas_mvsnet` 原生导入通过软链继续工作，已保存的 `state_dict` 键未改变。

## 配置与运行

`configs/model.json` 是模型参数，`configs/train.json` 是现有有限投影修复版本的训练设置，`configs/dev_complete.json` 和 `configs/test_complete.json` 是共同评价设置。它们软链接到 `configs/carla_comparison/` 的现有文件，保留原训练数据、划分和权重路径；复跑前检查并修改配置中的数据与权重位置，使用新的输出目录。

在项目根目录执行，输出目录应先链接到数据盘：

```bash
official_methods/.venv/bin/python tools/train_carla_casmvs.py --config jar/baselines/cas_mvsnet/configs/train.json --device cuda:0 --output runs/casmvs_new_train
official_methods/.venv/bin/python tools/carla_compare.py --config jar/baselines/cas_mvsnet/configs/dev_complete.json --device cuda:0 --output runs/casmvs_new_eval
```

真实 CARLA 训练入口调用 `upstream/module.py` 中的原生 `cas_mvsnet_loss`，采用配置记录的原生优化设置；`OfficialCasMVSLoss` 是保留旧合成基准兼容的包装损失，二者不要混称。适配器点云过滤和 unknown 包装属于独立后处理，不是原论文网络预测头。

CPU 几何、前向反向与标签隔离测试：

```bash
official_methods/.venv/bin/python -m pytest tests/test_official_casmvs.py -q
```
