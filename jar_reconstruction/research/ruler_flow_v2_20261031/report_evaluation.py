"""从已完成的正式 TEST 和独占测速生成报告，不运行模型或选择权重。"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/active/ruler_flow_v2_20261031"
GROUPS = ("public", "carla", "carla_other_refs")
NAMES = {
    "TartanAir_forest": "TartanAir",
    "MatrixCity_street": "MatrixCity",
    "Hypersim_subset": "Hypersim",
    "MVS_Synth": "MVS-Synth",
    "VKITTI2": "VKITTI2",
    "BlendedMVS": "BlendedMVS",
    "DynamicReplica_valid": "DynamicReplica",
    "CARLA_new_upper": "CARLA 上",
    "CARLA_new_stereo_left": "CARLA 左",
    "CARLA_new_stereo_right": "CARLA 右",
    "CARLA_new_stereo_down": "CARLA 下",
}


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_results(directory):
    datasets = {}
    for group in GROUPS:
        result = read(directory / group / "full_test/summary.json")
        assert result["completed"] and result["split"] == "test"
        datasets.update(result["datasets"])
    return datasets


def export_csv(native, equal):
    rows = []
    for protocol, datasets in (("native", native), ("equal_pixels", equal)):
        for name, dataset in datasets.items():
            for method, output in dataset["methods"].items():
                for scope in ("depth", "stages_stride8", "surface"):
                    sections = output.get(scope, {})
                    if scope == "surface":
                        sections = {"visible_surface": sections}
                    for region, metrics in sections.items():
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                rows.append([protocol, name, dataset["units"], method,
                                             output["frames"], scope, region, metric, value])
    path = RUN / "verification/metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protocol", "dataset", "units", "method", "frames",
                         "scope", "region", "metric", "value"])
        writer.writerows(rows)


def main():
    assert read(RUN / "verification/training_audit.json")["completed"]
    assert read(RUN / "verification/equal_pixels/completed_audit.json")["completed"]
    native = load_results(RUN / "evaluation")
    equal = load_results(RUN / "verification/equal_pixels")
    speed = read(RUN / "verification/speed_carla_all.json")["methods"]
    export_csv(native, equal)
    sections = ["""# V4 完整训练与测试：2026-10-31

这里的 **V4** 指独立 Flow 分支 `research/ruler_flow_v2_20261031`，架构标记仍为 `ruler_flow_v2`。对照 V2 是2026-10-29独立训练版本；不是早期 CARLA TEST200 或完整 Hero 通路适配版。

**结论：训练已完成；V4 有局部精度收益，但没有全面优于旧 V2，更没有全面超过 MVSAnywhere。暂不替换原推荐。** 公共原生输入下七域中六域 MAE 改善，近距七域改善；CARLA 上参考退化，单参考新增约1.1ms。原模型、权重和默认入口保留。本次只测试，没有重训或按 TEST 调参。

## 训练是否完成

北京时间公共训练12:40:45–15:07:02，CARLA适配15:07:02–15:26:35，合计 **2小时45分50秒**；随后自动测试15:26:35–15:52:08。训练、自动测试和本次补测退出状态均核验。

- 公共完整1+2+8轮，18,370更新、587,840次曝光；CARLA完整6+18轮，384更新、12,288次曝光。
- 正式比较采用完整VAL选出的best：公共stage3 epoch6/step15030，CARLA stage2 epoch18/step384。公共final与best不同，附加保存final结果但不按TEST择优；CARLA二者参数相同。
- 新增18,339参数确实更新，旧权重未覆盖。V4从V2热启动且多训练了一轮完整流程，收益不能全部归因为新增Flow结构。小预算同结构对照见原README，不等同于本次全预算架构消融。

## 评测范围与公平边界

两套输入协议各覆盖公共七域11,970组、CARLA四参考731组，共12,701参考样本。CARLA四参考来自187个唯一帧，不是731个独立场景。每套四方法，共50,804次预测；逐帧检查索引、共同GT分母、权重与清单，覆盖率均为1。

1. **原生输入：** 自研主图256×448、匹配图512×896；MVSAnywhere256×448。用于比较实际部署配置，不能称等像素信息量。
2. **同等图像信息：** 自研匹配图从共同256×448图上采样，计算尺寸仍512×896；基线仍用共同主图。固定原权重，是输入信息干预，不是低分辨率重训。
3. MVSAnywhere使用官方Hero架构/权重，未在本轮公共/CARLA数据继续训练；自研使用本地训练及CARLA适配权重。首遍只预测一次；细化为首遍后再做一次官方范围细化。双方使用同一TRAIN确定的深度搜索范围，不用TEST标签调范围。这是给定权重的共同测试，不是同等训练预算的纯架构对照，也不是官方原生范围或论文数值复现。
4. FP32、TF32关闭、相同位置编码缓存。VKITTI2/DynamicReplica实际双目，其余公共组四图；部分公共组为静态场景多位置图，不伪装为同步车载四目。缺失相机不复制填充，推理只用RGB和标定。
5. TRAIN/VAL/TEST序列ID、帧键及全部相机RGB路径无交叉；未证明视觉内容或官方预训练数据无交叉。Tartan仍为同一森林环境不同轨迹，Matrix为同城市官方划分；不称跨地图盲测。历史TEST已被研发查看。

MAE/RMSE/AbsRel均在固定有效GT像素上汇总，δ1为深度比小于1.25的比例；不按预测置信度删难例。米制近/中/远为<10、10–40、≥40m（CARLA总范围2–80m）。MVS-Synth、BlendedMVS、DynamicReplica保留scene_unit，不能标米或混合求米制平均；近距阈值分别133.125、18.212797、1.024414 scene_unit，来自固定TRAIN尺度。

## 原生配置：全范围深度

下表为全分辨率最终深度，不是共同stride8阶段抽样。V4均为VAL-best；MVS-A细化为首遍+1次细化。"""]
    rows = []
    for name, dataset in native.items():
        for key, label in (("original_v2", "V2"), ("flow_best", "V4"),
                           ("mvsanywhere_refined", "MVS-A细化")):
            output = dataset["methods"][key]
            full = output["depth"]["full"]
            rows.append([NAMES[name], dataset["units"], output["frames"], label,
                         f'{full["MAE"]:.6f}', f'{full["RMSE"]:.6f}',
                         f'{full["AbsRel"]:.6f}', f'{full["delta_1"]:.6f}',
                         f'{output["depth"]["near"]["MAE"]:.6f}'])
    sections.append(table(["数据", "单位", "组数", "模型", "MAE↓", "RMSE↓",
                           "AbsRel↓", "δ1↑", "近距MAE↓"], rows))
    sections.append("## 改善是否只来自少量帧\n\n以下按同帧配对，负差值表示V4更好。逐帧胜率不是像素加权总体指标，也不是统计显著性。")
    rows = []
    paired = read(RUN / "verification/per_frame_comparison.json")
    for name, dataset in paired["datasets"].items():
        full = dataset["full"]
        delta = full["delta_new_minus_old"]
        rows.append([NAMES[name], dataset["units"],
                     f'{full["wins"]}/{full["frames"]} ({100 * full["win_fraction"]:.2f}%)',
                     f'{full["improved_sequences"]}/{full["sequence_count"]}',
                     f'{delta["median"]:+.6f}', f'{delta["p90"]:+.6f}'])
    sections.append(table(["数据", "单位", "改善帧", "改善序列", "每帧MAE差中位", "差值P90"], rows))
    sections.append("CARLA上参考175/181帧退化，并非少量离群帧。Blended/Dynamic改善较广，Hyp变化接近持平。VK约一半帧改善，但像素加权全范围MAE仍小幅变差；其10种变体都来自同一TEST基础场景，不能看作10个独立城市。")
    sections.append("## 同等图像信息：完整TEST补测\n\n所有数值为全范围MAE，四方法同GT。不能与上表交叉择优。")
    rows = []
    for name, dataset in equal.items():
        keys = ("v2_equal_pixels", "v4_equal_pixels", "mvsanywhere_firstpass", "mvsanywhere_refined")
        values = [dataset["methods"][key]["depth"]["full"]["MAE"] for key in keys]
        rows.append([NAMES[name], dataset["units"], *[f"{value:.6f}" for value in values]])
    sections.append(table(["数据", "单位", "V2", "V4", "MVS-A首遍", "MVS-A细化"], rows))
    sections.append("同信息条件下，V4公共full MAE仍在6/7域优于V2，near MAE为7/7；但相对MVS首遍只有4/7域full MAE较低，相对细化为5/7，RMSE分别只有2/7和3/7。Blended原生优势0.75152对0.79966在同信息下缩为0.79771对0.79966，接近持平，说明较高匹配图信息确实影响优势大小。Tartan相对首遍2.51029对2.50911也接近持平；Hyp/Dynamic仍有明显差距。不能按每个域分别挑选原生/同信息结果作为一套模型成绩。")
    sections.append("CARLA同信息条件下，V4相对V2仍为左右下改善、上退化；四参考full MAE全部未超过MVS首遍，只有下参考优于细化版，且四参考near MAE都较差。MVS上参考首遍/细化MAE为1.88594/1.93503m，首遍更快且略准；细化并不必然有益。")
    sections.append("## 阶段定位\n\n以下粗/细/最终均在同一stride8有效像素采样；不得与全分辨率MAE混用。")
    rows = []
    for name, dataset in native.items():
        values = []
        for stage in ("coarse", "fine", "final"):
            old = dataset["methods"]["original_v2"]["stages_stride8"][stage]["MAE"]
            new = dataset["methods"]["flow_best"]["stages_stride8"][stage]["MAE"]
            values.append(f"{old:.5f} → {new:.5f}")
        rows.append([NAMES[name], dataset["units"], *values])
    sections.append(table(["数据", "单位", "粗层 V2→V4", "细层 V2→V4", "最终 V2→V4"], rows))
    sections.append("""公共粗层MAE仅Hypersim改善，细层六域改善；不能说初始粗候选问题已解决。Flow保留已有选面，主要改变区间内连续位置。小幅均值变化未经多随机种子重复确认，不宣称统计显著。

CARLA上参考的全分辨率MAE增加0.19123m，其中近距贡献0.18172m，约95%。粗层已由2.97574变为3.14604m，说明退化不只发生在末端Flow；联合训练的粗判断或域适配发生漂移是待验证原因。其他三参考有所改善，不能据此掩盖上参考退化。需要下一轮在TRAIN/VAL检查粗层保持约束、四参考训练取舍与分布变化；本次不使用TEST选择补救参数。

## 可见表面指标

把单参考有效GT区域中的预测/真值深度反投影成点集。Accuracy为预测到GT最近邻平均欧氏距离，Completeness反向，Chamfer_L1取二者平均；不是平方距离。每帧最多1024个确定性均匀查询点，对完整对方点集查询，按查询数汇总。阈值0.05/0.1/0.2/0.5m；scene_unit按TRAIN固定尺度换算，不称公制。

这是可见区域的单参考表面一致性，未做全场景融合、未知空间认证或完整占据IoU。全套阈值和统计保留在CSV/JSON；下表只展示Chamfer及第二个阈值的F-score。""")
    rows = []
    for name, dataset in native.items():
        for key, label in (("original_v2", "V2"), ("flow_best", "V4"),
                           ("mvsanywhere_refined", "MVS-A细化")):
            surface = dataset["methods"][key]["surface"]
            threshold = surface["thresholds"][1]
            fscore = surface[f"Fscore_{threshold:g}"]
            rows.append([NAMES[name], dataset["units"], label,
                         f'{surface["Chamfer_L1"]:.6f}', f"{threshold:.6g}", f"{fscore:.6f}"])
    sections.append(table(["数据", "单位", "方法", "Chamfer↓", "F阈值", "F-score↑"], rows))
    sections.append("""## 同卡网络耗时

独占RTX5090、FP32/TF32关闭，三个完整CARLA VAL输入；每帧每方法预热10次、交替计时40次，报告120次中位数。真实单参考/四参考调用，含编码与原生适配，不含读盘、CPU预处理、H2D或三维融合。各方法采用原生输入预算，因此既要看这张部署表，也要看上面的同信息精度表。""")
    rows = []
    for key, label in (("original_v2", "V2"), ("flow_best", "V4"),
                       ("mvsanywhere_first", "MVS-A首遍"), ("mvsanywhere_refined", "MVS-A细化")):
        single = speed[f"{key}_single"]
        serial = speed[f"{key}_serial_four"]
        shared = speed.get(f"{key}_shared_four")
        rows.append([label, f'{single["median_ms"]:.3f}', f'{serial["median_ms"]:.3f}',
                     f'{serial["hz"]:.2f}', f'{shared["median_ms"]:.3f}' if shared else "未测",
                     f'{shared["hz"]:.2f}' if shared else "未测"])
    sections.append(table(["方法", "单参考ms", "四参考串行ms", "串行Hz", "共享匹配四参考ms", "共享Hz"], rows))
    sections.append("""V4单参考比V2慢3.5%，比MVS-A细化快12.2%，比首遍慢48.8%。V4共享四参考9.02Hz，尚未到10Hz，且不是车端完整管线频率。共享与串行的深度/有效性逐元素相同；V2同样支持共享，不能把这个收益归于Flow。官方MVS接受匹配特征缓存，但本次未实现/计时其共享入口，不宣称战胜优化后的MVS。

另行隔离显存测量：V4/V2/MVS-A细化约1.023/1.023/1.771GiB峰值allocated，参数98.365/98.347/201.670M；包含单模型权重、输入与前向，不含CUDA驱动上下文。公共逐域单参考耗时与显存也保存在原始latency JSON，不与三帧重复测量混称完整TEST时延分布。

## 可复查文件与运行方式

- [完整指标CSV](../runs/active/ruler_flow_v2_20261031/verification/metrics.csv)：两个协议、best/final/对照、全范围/距离段/阶段/表面；`scope`明确区分统计口径。
- [训练审计](../runs/active/ruler_flow_v2_20261031/verification/training_audit.json)、[原生测试审计](../runs/active/ruler_flow_v2_20261031/verification/completed_audit.json)、[同信息测试审计](../runs/active/ruler_flow_v2_20261031/verification/equal_pixels/completed_audit.json)。
- [原始测速](../runs/active/ruler_flow_v2_20261031/verification/speed_carla_all.json)、[测速输入核验](../runs/active/ruler_flow_v2_20261031/verification/speed_input_audit.json)。
- [逐帧与逐序列比较](../runs/active/ruler_flow_v2_20261031/verification/per_frame_comparison.json)：只读已保存预测指标，没有重新选择测试帧。
- 原生逐帧结果：`runs/active/ruler_flow_v2_20261031/evaluation/{public,carla,carla_other_refs}/full_test/`。
- 同信息逐帧结果：`runs/active/ruler_flow_v2_20261031/verification/equal_pixels/{public,carla,carla_other_refs}/full_test/`。
- 本次大文件实际位于`/data/liuzonghao_data/3Dscene_runs/active_research/ruler_flow_v2_20261031`，通过项目软链接访问；没有网络下载。

在项目根目录`jar_reconstruction/`运行以下命令；报告命令只读取已完成结果，不重新推理或改权重：

```bash
official_methods/.venv/bin/python research/ruler_flow_v2_20261031/audit_training_v4.py
official_methods/.venv/bin/python research/ruler_flow_v2_20261031/audit_completed.py \
  --output runs/active/ruler_flow_v2_20261031/verification/completed_audit.json
official_methods/.venv/bin/python research/ruler_flow_v2_20261031/audit_equal_pixels.py
official_methods/.venv/bin/python research/ruler_flow_v2_20261031/report_evaluation.py
```

补测实际由`run_verification.sh`在tmux `v4_equal_pixels_20261031`执行，先全部域VAL首/中/末冒烟，再完整TEST。具体配置在`configs/verification/`，均固定权重。历史启动日志留存，不把旧“未完成”记录改写成当时已完成。
""")
    destination = ROOT / "docs/v4_evaluation_20261031.md"
    destination.write_text("\n\n".join(sections) + "\n")
    print(destination)
    print(RUN / "verification/metrics.csv")


if __name__ == "__main__":
    main()
