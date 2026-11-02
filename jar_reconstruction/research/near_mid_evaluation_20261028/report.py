"""从实际完整测试及独立测速结果生成中文报告和完整CSV。"""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import write_json

NAMES = {
    "own_original": "RulerMVS-V2·原推荐",
    "own_control_valbest": "RulerMVS-V2·读出VAL最佳",
    "own_control_last": "RulerMVS-V2·续训最终",
    "own_surface_last": "RulerMVS-V3·最终",
    "mvsanywhere_refined": "MVSAnywhere细化",
}


def number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows]) + "\n"


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    output = ROOT / config["output"]
    tested = json.loads((output / "full_test/summary.json").read_text())
    speed = json.loads((output / "latency/latency.json").read_text())
    assert tested["completed"] and speed["completed"] and tested["split"] == "test"
    depth_rows, stage_rows, surface_rows, joint_rows = [], [], [], []
    lines = ["# 最新中近距离训练：完整测试与速度对照（2026-10-28）", "",
             "本报告只评价固定检查点，不根据TEST结果选权重或修改模型。具体解释和结论见下方实测对照。", "",
             "## 训练与选择", "",
             "新面片结构和同预算旧结构均已完成1+6+6共13轮、17836次更新、570752组曝光。两组最终权重均未通过预先设定的VAL升级门槛；原推荐不覆盖。旧结构阶段1通过门槛，另列其VAL最佳权重，不能把最后一轮等同于最佳模型。", "",
             "完整7957组VAL近中宏归一风险：新结构最终+39.23%，旧结构最终+14.11%，旧结构阶段1−0.0333%。这是逐域近/中MAE、RMSE、AbsRel相对原推荐的平均变化，不是米制平均误差。", "",
             "‘旧结构续训’保留原凸上采样；与新结构一样采用独立细层评分/偏移头（从旧共享头复制，增加130参数）。因此它控制的是本轮共同训练设置下的旧读出，不是原推荐每个参数绑定完全不变的续训。原推荐零更新对照另列。", "",
             "## 本次结论", "",
             "**新面片结构未达到升级目标。** 七个公共域的最终MAE全部差于原推荐；旧结构的VAL最佳检查点七域MAE均小幅降低，但Tartan/Hypersim/VKITTI2/Blended的AbsRel略升，且Tartan/VKITTI2近距MAE变差，仍不能称为满足‘近中改善且其他不退化’。本次不替换原推荐配置。", "",
             "**末端读出会损失前端收益，但不是唯一问题。** 在同stride8位置，MVS-Synth新结构细候选MAE102.847优于旧版108.460，最终却113.516劣于旧版104.854；MatrixCity新结构细候选23.235、最终50.294，而旧版13.701→9.682。说明新像素选面/平面读出未把候选优势转成输出，同时Matrix前端也已退化。仅靠继续增强候选供给不足以解释和修复全部差距。", "",
             "**联合继续训练存在迁移退化。** MatrixCity旧结构最终MAE34.509、新结构47.081，而原推荐10.326；旧结构VAL曾改善，TEST却明显恶化。该现象与过拟合/跨场景失准相容，但本次评测没有隔离表示、损失和编码器更新的因果贡献，不能断言单一原因。不能仅凭完整训练loss或VAL某域下降认定模型更好。", "",
             "下一步应先隔离末端路由与联合特征更新的影响，并保留原平滑读出作为受控对照；有更好的候选oracle并不保证硬选面更稳。本次没有继续调参或启动新训练，避免根据TEST反复选型。", "",
             "## 共同协议与适用范围", "",
             "- 公共TEST七域11970组；额外旧CARLA完整200和缺相机150组。最新公共权重没有CARLA微调，CARLA表只代表迁移测试；不能替代历史CARLA微调结果。",
             "- 相同真实视图、参考相机、256×448主图及有效GT，FP32、TF32关闭。自研原生匹配图512×896，MVSAnywhere保留原生设置，属于系统比较，不是同像素计算预算。",
             "- MVSAnywhere官方Hero权重、1次范围细化、共享各域TRAIN范围；不使用TEST深度决定输入范围。训练数据暴露和预算不同，不将净差距全归因于架构。",
             "- 不按置信度、未知状态或候选范围删除GT。MAE/RMSE等对有效预测计算，同时保留全部GT分母和coverage；预测缺失时full_gt_MAE为null。",
             "- meters域近<10m、中[10,40)m、远≥40m；scene_unit域按TRAIN中位深度的0.5和2倍分段。不同单位绝不合并MAE。",
             "- DynamicReplica是原官方validation子集内的项目TRAIN/VAL/TEST拆分，不是官方完整独立test。官方baseline可能见过部分公共序列，非对等零样本认证。",
             "- 这些项目TEST已在历史研究中多次查看，不是全新盲测；本轮预先固定五组权重再评测，不用本轮TEST重选检查点。",
             "- CARLA沿用旧数据2–80m全有效区域及上相机参考，tele不输入；缺相机保留实际2/3/4目。旧采集深度的来源仍有历史限制，不混入新四轨迹数据。", "",
             "## 深度精度、速度与显存", "",
             "MAE为绝对误差均值；RMSE是平方误差均值开根号，更惩罚少数大错误；AbsRel为绝对误差/真值深度的均值。δ1/2/3分别统计max(预测/真值,真值/预测)<1.25¹/²/³，完整CSV提供。", "",
             "网络耗时：单张RTX5090、batch1、同卡交替、每样本预热8次/计时30次，各域3个无标签VAL样本共90次。表中为墙钟中位；不含读盘、CPU→GPU、点云/网格融合。Hz只代表单参考深度，不是全部相机重建频率。独占显存包含输入、权重和前向激活，排除CUDA驱动上下文。", ""]
    for domain, result in tested["datasets"].items():
        timings = speed["datasets"][domain]
        lines += [f"### {domain}：{result['units']}，{result['num_views']}个相机槽位", ""]
        rows = []
        for name in NAMES:
            method = result["methods"][name]
            metrics = method["depth"]["full"]
            latency = timings["summary"][name]["wall_network_ms"]
            resource = timings["isolated_resources"][name]
            rows.append([NAMES[name], method["frames"], number(metrics["MAE"]), number(metrics["RMSE"]),
                         number(metrics["AbsRel"]), number(latency["median_ms"], 2),
                         number(latency["single_reference_hz"], 1), number(resource["peak_allocated_gib"], 3)])
            common = {"dataset": domain, "method": name, "label": NAMES[name], "units": result["units"]}
            for region, values in method["depth"].items():
                depth_rows.append(dict(common, region=region, frames=method["frames"], **values))
            for stage, values in method["stages_stride8"].items():
                stage_rows.append(dict(common, stage=stage, **values))
            surface_rows.append(dict(common, **method["surface"]))
            joint_rows.append(dict(common, **metrics, network_ms=latency["median_ms"],
                                   network_p90_ms=latency["p90_ms"], single_reference_hz=latency["single_reference_hz"],
                                   **resource))
        lines += [table(["方法", "TEST组数", "MAE↓", "RMSE↓", "AbsRel↓", "网络ms↓", "单参考Hz↑", "峰值GiB↓"], rows), ""]
    lines += ["## 近、中、远与深度突变区", "",
              "以下均为MAE；单位沿用数据集。深度突变带由相邻有效GT的|ΔlogZ|>0.1并向外扩2像素得到，仅是边界代理，不等同于实例边缘或小障碍物识别率。", ""]
    for domain, result in tested["datasets"].items():
        rows = [[NAMES[name]] + [number(result["methods"][name]["depth"][region]["MAE"])
                                 for region in ("near", "mid", "far", "gradient_band", "smooth")] for name in NAMES]
        lines += [f"### {domain}", "", table(["方法", "近", "中", "远", "突变带", "平滑区"], rows), ""]
    lines += ["## 共同stride8的阶段诊断", "",
              "粗层原格点、细层每隔2点和最终每隔8点都对应原图(8x,8y)，不插值或平移GT。该表不能与上方全分辨率主指标混用。MVSAnywhere这里只列最终值，不把其内部不可比中间张量命名成自研粗层。", ""]
    for domain, result in tested["datasets"].items():
        rows = []
        for name in NAMES:
            stages = result["methods"][name]["stages_stride8"]
            rows.append([NAMES[name]] + [number(stages.get(stage, {}).get("MAE")) for stage in ("coarse", "fine", "final")])
        lines += [f"### {domain}", "", table(["方法", "粗候选MAE", "细候选MAE", "最终MAE"], rows), ""]
    lines += ["## 单参考可见表面指标", "",
              "使用同一参考相机内参反投影GT与预测。每帧每侧等间隔抽最多1024个查询点，但最近邻参考是另一侧全部有效像素点；不能双侧抽稀后计算距离。Accuracy/Completeness分别为预测→GT、GT→预测平均最近邻欧氏距离，Chamfer_L1为两方向均值。只评价共同GT可见区域，不代表完整场景覆盖或占据。", "",
              "meters域F-score阈值0.05/0.1/0.2/0.5m；scene_unit域阈值为TRAIN中位深度的0.005/0.01/0.02/0.05倍，不称米制阈值。方向距离按查询点合并、F-score由合并precision/recall计算；不是逐帧F-score平均。", ""]
    for domain, result in tested["datasets"].items():
        rows = []
        example = next(iter(result["methods"].values()))["surface"]
        fscores = [k for k in example if k.startswith("Fscore_")]
        for name in NAMES:
            values = result["methods"][name]["surface"]
            rows.append([NAMES[name]] + [number(values[key]) for key in ("Accuracy", "Completeness", "Chamfer_L1", *fscores)])
        lines += [f"### {domain}", "", table(["方法", "Accuracy↓", "Completeness↓", "Chamfer↓", *fscores], rows), ""]
    lines += ["## 证据与复现", "",
              "- `research/near_mid_evaluation_20261028/config.json`：固定五组权重、数据范围、相机协议。",
              "- `runs/active/near_mid_evaluation_20261028/full_test/summary.json`：完整测试结果；各域part文件包含逐帧统计和源/权重/清单hash。",
              "- 同目录`latency/latency.json`及逐样本文件：交替计时原始90次样本和独占显存。",
              "- `depth_metrics.csv`包含全部区域和δ/log/coverage，`stage_metrics.csv`为阶段结果，`surface_metrics.csv`为可见表面，`accuracy_speed.csv`合并深度精度及速度。",
              "- 未测或不适用：全场景多参考融合、mesh、占据/语义IoU、车端硬件、训练显存/吞吐；本次网络没有经认证的occupied/free输出，不能虚构相应指标。", "",
              "```bash", "official_methods/.venv/bin/python -m unittest research.near_mid_evaluation_20261028.test_metrics -v",
              "official_methods/.venv/bin/python research/near_mid_evaluation_20261028/evaluate.py --config research/near_mid_evaluation_20261028/config.json --smoke",
              "bash research/near_mid_evaluation_20261028/run.sh", "```", "",
              "计时入口拒绝覆盖既有输出；重跑需修改输出目录。完整测试分片仅在协议hash完全一致时复用。所有大结果落在数据盘既有软链接。", ""]
    for name, rows in (("depth_metrics", depth_rows), ("stage_metrics", stage_rows),
                       ("surface_metrics", surface_rows), ("accuracy_speed", joint_rows)):
        save_csv(output / f"{name}.csv", rows)
    (ROOT / "docs/latest_training_evaluation_20261028.md").write_text("\n".join(lines))
    write_json(output / "report_status.json", {"complete": True, "domains": len(tested["datasets"]),
               "tested_groups": sum(next(iter(d["methods"].values()))["frames"] for d in tested["datasets"].values()),
               "methods": list(NAMES), "documentation": "docs/latest_training_evaluation_20261028.md"})


if __name__ == "__main__":
    main()
