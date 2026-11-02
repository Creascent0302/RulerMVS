"""汇总已完成的共同对比，公共迁移未完成时明确标记。"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/active/baseline_full_comparison_20261101"
LABELS = {"original_v2": "V2", "v5_best": "V5", "mvsanywhere_refined": "MVSAnywhere一次细化",
          "patchmatchnet_best": "PatchmatchNet CARLA best", "diffmvs_best": "DiffMVS CARLA best",
          "patchmatchnet_final": "PatchmatchNet final", "diffmvs_final": "DiffMVS final"}


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    carla = read(OUT / "carla_four_references/full_test/summary.json")
    validation = read(OUT / "validation/full_val/summary.json")["datasets"]["CARLA_new_upper"]["methods"]
    single = read(OUT / "single_latency/latency.json")["datasets"]["CARLA_new_upper"]
    four = read(OUT / "four_reference_latency.json")["methods"]
    training = read(OUT / "training_audit.json")
    names = ("original_v2", "v5_best", "mvsanywhere_refined", "patchmatchnet_best", "diffmvs_best")
    upper = carla["datasets"]["CARLA_new_upper"]["methods"]
    for metric in ("MAE", "RMSE", "AbsRel"):
        a = upper["patchmatchnet_best"]["depth"]["full"][metric]
        b = upper["patchmatchnet_final"]["depth"]["full"][metric]
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7), "同权重同随机流的误差超出浮点容差"
    assert len({record["depth"]["full"]["gt_pixels"] for record in upper.values()}) == 1
    rows = []
    for name in names:
        d = upper[name]["depth"]["full"]
        speed = single["summary"][name]
        resource = single["isolated_resources"][name]
        rows.append([LABELS[name], f"{d['MAE']:.5f}", f"{d['RMSE']:.5f}", f"{d['AbsRel']:.5f}",
                     f"{100*d['coverage']:.5f}%", f"{upper[name]['surface']['Fscore_0.1']:.5f}",
                     f"{speed['wall_network_ms']['median_ms']:.2f}",
                     f"{speed['wall_with_resampling_ms']['median_ms']:.2f}", f"{resource['peak_allocated_gib']:.3f}"])
    text = ["# 新baseline训练完成与完整对比（2026-11-01）", "",
            "两种新baseline均完成384次更新、11940次实际样本曝光和自动best/final TEST。原训练结束时间均约为北京时间11月1日09:50。本次不再训练、不依据TEST改模型，而是补齐统一输入、四参考、随机重复及速度/显存对比。", "",
            "## 上参考共同主表", "",
            "174帧相同完整四目，256×448共同RGB信息，2–80米，同一GT掩码、FP32/TF32关闭；V2/V5匹配张量仍为512×896，但只能上采样共同主图。三个新随机种子只是推理重复，不能替代多次独立训练。", "",
            table(["方法", "MAE/m", "RMSE/m", "AbsRel", "覆盖率", "F-score@0.1m", "网络+适配/ms", "含格点转换/ms", "独占峰值/GiB"], rows), "",
            "误差是有效预测条件指标，覆盖率不足时不能称全GT MAE；缺失预测仍计入覆盖率/阈值失败，不静默删帧。表面F-score仅评价参考视图可见表面，不是全场景占据。", "",
            "速度在单卡5090无其他GPU任务时测量，固定VAL首中末3帧、10次预热/40次交替计时；单模型显存另行独立驻留测量。GPU常驻输入，不含IO/H2D/点云融合；不能当作车端完整系统延迟。", "",
            "## 实际四参考速度", ""]
    speed_rows = []
    for name in names:
        serial = four[name + "_serial_four"]
        shared = four.get(name + "_shared_four", serial)
        speed_rows.append([LABELS[name], f"{serial['median_ms']:.2f}", f"{shared['median_ms']:.2f}", f"{shared['hz']:.2f}"])
    text += [table(["方法", "逐参考四次/ms", "当前可用四参考入口/ms", "四参考Hz"], speed_rows), "",
             "V2/V5共享入口与逐参考depth/valid逐位一致；其余使用原生逐参考调用。四参考时间为实际执行而非单参考乘四，此列未包含格点转换。没有给baseline添加新的共享编码优化，不把当前实现速度当其理论最快速度。", "",
             "## 每个相机与距离段", ""]
    rows = []
    records = []
    for domain, result in carla["datasets"].items():
        for name, record in result["methods"].items():
            for region, d in record["depth"].items():
                records.append(dict(dataset=domain, method=name, frames=record["frames"], region=region, **d))
            if name in names:
                rows.append([domain, LABELS[name], record["frames"], *[f"{record['depth'][region]['MAE']:.5f}"
                             for region in ("full", "near", "mid", "far")]])
    with (OUT / "depth_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    text += [table(["参考", "方法", "帧数", "全范围MAE", "2–10m", "10–40m", "40–80m"], rows), "",
             "两新增baseline此次仅以上相机作为训练参考；左/右/下属于额外的跨参考迁移。V2/V5有参考轮换训练，因此这些副表不能作为相同训练任务下的架构能力结论。", "",
             "## 随机重复与best/final", ""]
    repeats = read(OUT / "stochastic_repeats/full_test/summary.json")["datasets"]["CARLA_new_upper"]["methods"]
    repeat_rows = []
    for method in ("patchmatchnet", "diffmvs"):
        samples = [upper[method + "_best"]["depth"]["full"]["MAE"]]
        samples += [repeats[f"{method}_seed{seed}"]["depth"]["full"]["MAE"] for seed in (2701, 3701)]
        repeat_rows.append([method, *[f"{value:.5f}" for value in samples],
                            f"{statistics.mean(samples):.5f} ± {statistics.stdev(samples):.5f}"])
    text += [table(["方法", "seed1701 MAE", "seed2701", "seed3701", "均值±样本标准差/m"], repeat_rows), "",
             "保留官方随机候选/扩散噪声；每帧用RGB+K+T+参考号+seed确定随机流，既不读GT，也不依赖模型加载顺序或分片GPU。PatchmatchNet best/final参数相同，同随机流下MAE/RMSE在float32容差内一致，不宣称GPU数值末位逐位相同；先前约0.0015米差别是随机性，不是权重更优。", ""]
    rows = []
    for name in ("patchmatchnet_best", "patchmatchnet_final", "diffmvs_best", "diffmvs_final"):
        v, t = validation[name]["depth"]["full"], upper[name]["depth"]["full"]
        score = (v["MAE"] * v["predicted_pixels"] + 78 * (v["gt_pixels"] - v["predicted_pixels"])) / v["gt_pixels"]
        rows.append([LABELS[name], f"{v['MAE']:.5f}", f"{score:.5f}", f"{t['MAE']:.5f}", f"{t['RMSE']:.5f}"])
    text += [table(["保存权重", "完整VAL MAE/m", "缺失惩罚VAL分数", "TEST MAE/m", "TEST RMSE/m"], rows), "",
             "复核发现旧四卡验证混用了各rank的BN运行统计，而实际checkpoint保存rank0统计。因此历史best只能称旧规则选出的保存点，不保证是所有历史训练迭代在部署模式下的最佳点。已修复后续验证：临时广播rank0缓冲量，验证后恢复各rank训练状态；现表逐个实际checkpoint独立验证，best/final全部保留，不用TEST选择。该问题影响验证选模，不会把GT泄漏进推理，也不使此前固定checkpoint的TEST输出失效。", ""]
    if (OUT / "bn_audit.json").exists():
        bn = read(OUT / "bn_audit.json")
        text += ["固定原来的采样/RNG、真实四卡单独隔离BN影响：", "",
                 table(["方法", "旧混合BN VAL分数", "实际rank0模型 VAL分数"],
                       [[name, f"{v['historical_mixed_buffers']['selection_score']:.5f}",
                         f"{v['single_saved_checkpoint']['selection_score']:.5f}"] for name, v in bn.items()]), ""]
    text += ["## 公平性与训练成熟度", "",
             "PatchmatchNet从官方DTU权重迁移；DiffMVS无官方任务权重，从头只训练当前403帧CARLA。V2/V5还经过七域公共预训练，MVSAnywhere使用官方多域预训练。当前是相同测试输入与指标下的模型状态比较，不是等数据、等预训练预算的架构上限对比。尤其不能把DiffMVS先导训练较差当作论文方法无效。完整训练预算和原生单位/损失差异沿用194条记录；本次没有改任何网络结构。", "",
             "## 公共跨域测试", ""]
    public_path = OUT / "public_carla_transfer/full_test/summary.json"
    if public_path.exists():
        public = read(public_path)
        rows = []
        for domain, result in public["datasets"].items():
            for name, record in result["methods"].items():
                d = record["depth"]["full"]
                rows.append([domain, result["units"], LABELS[name], record["frames"],
                             f"{d['MAE']:.5f}", f"{d['RMSE']:.5f}", f"{d['coverage']:.6f}"])
        text += [table(["数据集", "单位", "CARLA适配权重", "组数", "MAE", "RMSE", "覆盖率"], rows), "",
                 "七域11970组全部完成，使用真实N=2/4，不复制视图；scene_unit不写成米。这里测CARLA适配后的域外迁移，不能冒充这些baseline已完成七域共同预训练。", ""]
    else:
        text += ["七域11970组CARLA适配权重跨域TEST已排在同一tmux队列，此报告快照尚未完成，不填未完成指标。", ""]
    text += ["## 文件与复现", "",
             "配置/入口：`research/baseline_full_comparison_20261101/`；队列为`baseline_full_compare_20261101`。完整结果、逐帧分片、权重/源码/清单hash、时间分布、深度CSV与验证审计均在`runs/active/baseline_full_comparison_20261101/`数据盘软链接下。训练权重保留在`runs/active/v5_comparison_20261101/{patchmatchnet_adapt,diffmvs_adapt}/training/`。", "",
             "新增回归包含随机流不受加载顺序影响、不同seed保留随机变化、双进程Gloo验证使用同一BN并恢复本地状态；连同既有训练保存/恢复测试共7项通过。初次Gloo受沙箱回环限制，已在允许本机通信环境实测；另修正随机初始化网络必定100%覆盖的错误测试前提，未因此改变网络。", ""]
    (ROOT / "docs/baseline_full_comparison_20261101.md").write_text("\n".join(text))
    marker = OUT / ("public_recorded.json" if public_path.exists() else "carla_recorded.json")
    if not marker.exists():
        message = "\n\n### 实验 195 自动完成补充：" + ("公共跨域TEST" if public_path.exists() else "共同四参考TEST与随机重复") + "\n\n"
        message += "- 时间UTC：" + datetime.now(timezone.utc).isoformat() + "。配置固定、无TEST选模，结果写入docs/baseline_full_comparison_20261101.md。\n"
        if public_path.exists():
            message += "- 新两方法CARLA适配权重的七域11970组跨域测试完成；真实N=2/4、scene_unit明确。它不是七域充分预训练后的同预算架构对照。证据public_carla_transfer/full_test/summary.json。\n"
        else:
            message += "- 同174帧上参考MAE：" + "；".join(f"{LABELS[name]} {upper[name]['depth']['full']['MAE']:.5f}m" for name in names) + "。其他三参考各174帧均完成，全部方法GT分母一致。Patchmatch同权重同seed的best/final MAE/RMSE在浮点容差内一致，原随机性保留，不宣称GPU数值逐位相同。\n"
        message += "- 保留可复现的固定权重结果，不把当前有限训练baseline当作架构上限。后续须完善共同公共预训练预算并使用修正后的单模型VAL选优，不能依据此次TEST再选超参数。完整证据在runs/active/baseline_full_comparison_20261101。\n"
        with (ROOT / "docs/research_memory.md").open("a") as stream:
            stream.write(message)
        marker.write_text(json.dumps({"completed": True}))


if __name__ == "__main__":
    main()
