"""合并已核验协议的首遍补充与主TEST；速度只使用本轮独占实测。"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from research.paper_analysis_20261101 import analyze as plots


plots.LABELS["mvsanywhere_firstpass"] = "MVSAnywhere (first pass)"
plots.COLORS["mvsanywhere_firstpass"] = "#7f86c2"
plots.LABELS["v6_fast"] = "RulerMVS-V6 (fast)"
plots.COLORS["v6_fast"] = "#42b4aa"
METHODS = ("v6_best", "mvsanywhere_firstpass", "mvsanywhere_refined")


def combine(main_folder, first_folder):
    main = plots.read(main_folder / "summary.json")
    first = plots.read(first_folder / "summary.json")
    if not main.get("completed") or not first.get("completed") or main["split"] != "test" or first["split"] != "test":
        raise ValueError("只能合并完整TEST")
    audit = plots.read(main_folder / "audit.json")
    addition = plots.read(first_folder / "audit.json")
    for key in ("datasets", "image_size", "matching_image_size", "tf32", "surface_queries", "seed", "position_cache", "fast_cost_volume"):
        if audit["config"][key] != addition["config"][key]:
            raise ValueError("首遍与主TEST协议不一致：" + key)
    if audit["manifest_sha256"] != addition["manifest_sha256"]:
        raise ValueError("首遍与主TEST清单不一致")
    if audit["source_sha256"] != addition["source_sha256"]:
        raise ValueError("首遍与主TEST模型或指标实现版本不一致")
    refined = copy.deepcopy(audit["config"]["methods"]["mvsanywhere_refined"])
    refined["refinements"] = 0
    if refined != addition["config"]["methods"]["mvsanywhere_firstpass"]:
        raise ValueError("首遍不能改变细化次数以外的模型配置")
    if audit["weights_sha256"]["mvsanywhere_refined"] != addition["weights_sha256"]["mvsanywhere_firstpass"]:
        raise ValueError("首遍与细化版必须同一任务权重")
    combined = copy.deepcopy(main)
    for name, domain in combined["datasets"].items():
        values = first["datasets"][name]
        if domain["units"] != values["units"] or domain["num_views"] != values["num_views"]:
            raise ValueError("数据单位或视图数不一致")
        entry = values["methods"]["mvsanywhere_firstpass"]
        for existing in domain["methods"].values():
            if existing["frames"] != entry["frames"] or existing["depth"]["full"]["gt_pixels"] != entry["depth"]["full"]["gt_pixels"]:
                raise ValueError("样本或GT分母不一致")
        domain["methods"]["mvsanywhere_firstpass"] = entry
    return combined, audit, addition


def pooled_camera_metrics(summary, method):
    values = [domain["methods"][method]["depth"]["full"] for domain in summary["datasets"].values()]
    predicted = sum(value["predicted_pixels"] for value in values)
    gt = sum(value["gt_pixels"] for value in values)
    return {"MAE": sum(value["MAE"] * value["predicted_pixels"] for value in values) / predicted,
            "RMSE": float(np.sqrt(sum(value["RMSE"] ** 2 * value["predicted_pixels"] for value in values) / predicted)),
            "AbsRel": sum(value["AbsRel"] * value["predicted_pixels"] for value in values) / predicted,
            "coverage": predicted / gt, "gt_pixels": gt, "predicted_pixels": predicted}


def surface_comparison(summary, output):
    """同一参考可见GT范围的点集距离，不扩展为全场景重建结论。"""
    names = list(summary["datasets"])
    if len(names) != 4 or any(value["units"] != "meters" for value in summary["datasets"].values()):
        raise ValueError("此紧凑表面图专用于四参考CARLA公制协议")
    fig = plots.plt.figure(figsize=(12, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 12)
    geometry_axes = [fig.add_subplot(grid[0, index * 4:(index + 1) * 4]) for index in range(3)]
    fields = ("Accuracy", "Completeness", "Chamfer_L1")
    titles = ("Accuracy: prediction to GT", "Completeness: GT to prediction", "Symmetric Chamfer-L1")
    for axis, field, title in zip(geometry_axes, fields, titles):
        for method in METHODS:
            values = [summary["datasets"][name]["methods"][method]["surface"][field] for name in names]
            axis.plot(range(4), values, "o-", color=plots.COLORS[method], label=plots.LABELS[method], lw=1.5, ms=4)
        axis.set(title=title, xticks=range(4), xticklabels=[plots.SHORT[name].removeprefix("CARLA ") for name in names],
                 ylabel="Distance (m), lower is better")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    for index, name in enumerate(names):
        axis = fig.add_subplot(grid[1, index * 3:(index + 1) * 3])
        for method in METHODS:
            metric = summary["datasets"][name]["methods"][method]["surface"]
            thresholds = metric["thresholds"]
            if thresholds != [.05, .1, .2, .5]:
                raise ValueError("表面阈值不是预登记的5/10/20/50cm")
            scores = [metric[f"Fscore_{threshold:g}"] for threshold in thresholds]
            axis.plot(thresholds, scores, "o-", color=plots.COLORS[method], ms=4, lw=1.4)
        axis.set(xscale="log", xticks=thresholds, xticklabels=["5", "10", "20", "50"],
                 xlabel="Distance threshold (cm)", ylabel="F-score (higher is better)",
                 title=plots.SHORT[name], ylim=(0, 1))
        axis.minorticks_off()
        axis.grid(alpha=.2)
    handles, labels = geometry_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    plots.save(fig, output, "visible_surface_comparison", "CARLA TEST: single-reference backprojected visible surfaces; 1,024 evenly spaced queries per side against the full opposite set. Same GT-visible region; no ICP, scene fusion, mesh, or occupancy evaluation.")
    lines = ["% Visible single-reference point sets; F-scores in percent; distances in meters.",
             "\\begin{tabular}{llrrrrrrr}", "\\toprule",
             "Camera & Method & Acc.$\\downarrow$ & Comp.$\\downarrow$ & Chamfer$\\downarrow$ & F5 & F10 & F20 & F50 \\\\", "\\midrule"]
    for name in names:
        for method in METHODS:
            metric = summary["datasets"][name]["methods"][method]["surface"]
            row = [f"{metric[field]:.4f}" for field in fields]
            row += [f"{100 * metric[f'Fscore_{threshold:g}']:.2f}" for threshold in metric["thresholds"]]
            lines.append(plots.SHORT[name] + " & " + plots.LABELS[method] + " & " + " & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output / "surface_table.tex").write_text("\n".join(lines) + "\n")


def paired_surface_frames(main_folder, first_folder, output):
    """检查表面均值优势是否来自多数帧；这里只是固定轨迹描述，不做独立帧显著性。"""
    rows, summary = [], {}
    for directory in sorted(first_folder.iterdir()):
        if not directory.is_dir():
            continue
        frames = {}
        for source, method in ((main_folder / directory.name, "v6_best"), (directory, "mvsanywhere_firstpass")):
            for filename in source.glob("part_*.json"):
                part = plots.read(filename)
                for item in part["frames"]:
                    values = item["methods"][method]
                    surface = values["surface"]
                    depth = values["depth"]["full"]
                    accuracy = surface["accuracy_sum"] / surface["pred_queries"]
                    completeness = surface["completeness_sum"] / surface["gt_queries"]
                    frames.setdefault(item["index"], {})[method] = {
                        "depth_mae": depth["abs_sum"] / depth["predicted_pixels"],
                        "surface_accuracy": accuracy, "surface_completeness": completeness,
                        "surface_chamfer": .5 * (accuracy + completeness)}
        if not frames:
            continue
        for index, methods in sorted(frames.items()):
            if set(methods) != {"v6_best", "mvsanywhere_firstpass"}:
                raise ValueError("逐帧表面比较没有完整配对")
            rows.append({"dataset": directory.name, "index": index,
                         **{method + "_" + metric: value for method, metrics in methods.items() for metric, value in metrics.items()}})
        metrics = {}
        for metric in ("depth_mae", "surface_accuracy", "surface_completeness", "surface_chamfer"):
            own = np.array([values["v6_best"][metric] for values in frames.values()])
            first = np.array([values["mvsanywhere_firstpass"][metric] for values in frames.values()])
            metrics[metric] = {"v6_equal_frame_mean": float(own.mean()), "firstpass_equal_frame_mean": float(first.mean()),
                               "v6_frame_win_fraction": float((own < first).mean()),
                               "paired_difference_median": float(np.median(own - first))}
        summary[directory.name] = {"frames": len(frames), "metrics": metrics}
    plots.csv_write(output / "paired_surface_depth_frames.csv", rows)
    plots.write_json(output / "paired_surface_depth_summary.json", {"datasets": summary,
                     "note": "固定同轨迹逐帧描述；未将连续帧当独立样本。深度equal-frame mean与主表pixel-weighted MAE不是同一聚合口径。"})


def plot_speed(summary, accuracy_audit, report, output, fast_audit=None):
    if not report.get("completed") or report.get("actual_views") != 4:
        raise ValueError("需要本轮完整四目独占测速")
    for method in METHODS:
        source = "mvsanywhere_refined" if method == "mvsanywhere_firstpass" else method
        if report["checkpoint_sha256"][method] != accuracy_audit["weights_sha256"][source]:
            raise ValueError("速度和精度权重不相同")
    for key in ("image_size", "matching_image_size", "position_cache", "fast_cost_volume"):
        if report["config"][key] != accuracy_audit["config"][key]:
            raise ValueError("速度和精度输入设置不相同")
    own = accuracy_audit["config"]["methods"]["v6_best"]
    if own != report["config"]["methods"]["v6_best"]:
        raise ValueError("V6精度/速度实现设置不同")
    comparison_methods = list(METHODS)
    if fast_audit is not None:
        if fast_audit["config"]["fast"] != report["settings"]["fast"]:
            raise ValueError("fast完整TEST与测速部署选项不相同")
        if fast_audit["weights_sha256"]["v6_fast"] != report["checkpoint_sha256"]["v6_best"]:
            raise ValueError("fast完整TEST与测速权重不相同")
        comparison_methods.append("v6_fast")
    frame_sets = []
    for domain in accuracy_audit["config"]["datasets"]:
        samples = [json.loads(line) for line in plots.path(domain["manifests"]["test"]).read_text().splitlines()]
        frame_sets.append({(sample["sequence"], sample["frame_id"]) for sample in samples})
    if len(frame_sets) != 4 or any(value != frame_sets[0] for value in frame_sets):
        raise ValueError("四参考池化比较必须使用同一组同步帧")
    speed = report["methods_with_resampling"]
    rows = []
    fig, axes = plots.plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for axis, scope in zip(axes, ("single", "four")):
        for method in comparison_methods:
            metric = (summary["datasets"]["CARLA_new_upper"]["methods"][method]["depth"]["full"]
                      if scope == "single" else pooled_camera_metrics(summary, method))
            key = method + ("_single" if scope == "single" else "_shared_four" if method.startswith("v6") else "_serial_four")
            milliseconds = speed[key]["median_ms"]
            row = {"method": method, "scope": scope, "latency_ms": milliseconds,
                   "latency_p95_ms": speed[key]["p95_ms"], "hz": speed[key]["hz"], **metric}
            rows.append(row)
            axis.scatter(milliseconds, metric["MAE"], s=90, color=plots.COLORS[method],
                         marker="*" if method.startswith("v6") else "o", label=plots.LABELS[method])
        axis.set(xlabel="Network + common output resampling (ms)", ylabel="TEST MAE (m)",
                 title="One upper reference" if scope == "single" else "All four reference depths")
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    plots.save(fig, output, "firstpass_accuracy_latency", "Same 5090, complete four-view VAL timing and fixed CARLA TEST accuracy. Four-reference MAE pools camera-ray pixels, not a fused point cloud. Fast is plotted only after its own full TEST completes.")
    plots.csv_write(output / "accuracy_latency.csv", rows)

    # 快速版只绘制真实速度，不用eager的完整TEST精度给fast填点。
    names = ["v6_best", "v6_fast", "mvsanywhere_firstpass", "mvsanywhere_refined"]
    labels = ["V6 eager", "V6 fast", "MVSAnywhere first", "MVSAnywhere refined"]
    fig, axes = plots.plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, scope in zip(axes, ("single", "four")):
        times = []
        for method in names:
            key = method + ("_single" if scope == "single" else "_shared_four" if method.startswith("v6") else "_serial_four")
            times.append(speed[key]["median_ms"])
        bars = axis.barh(labels, times, color=["#067a83", "#42b4aa", "#7f86c2", "#4351a3"])
        axis.bar_label(bars, fmt="%.2f ms", padding=4, fontsize=8)
        axis.set(xlabel="Network + common output resampling (ms)",
                 title="One upper reference" if scope == "single" else "All four reference depths")
        axis.invert_yaxis()
        axis.set_xlim(0, max(times) * 1.22)
        axis.grid(axis="x", alpha=.2)
    plots.save(fig, output, "firstpass_latency_modes", "Actual complete-four-view measurements; four-reference calls are timed, not single latency times four. V6 uses shared matching for four outputs; baseline uses its existing serial path. Cold compile excluded.")


def add_fast_depth(summary, accuracy_audit, filename, output, main_folder):
    report = plots.read(filename)
    audit = plots.read(filename.with_name("audit.json"))
    if not report.get("completed") or report["split"] != "test":
        raise ValueError("fast只接入完整TEST")
    for key in ("datasets", "image_size", "matching_image_size", "tf32", "seed", "position_cache"):
        if audit["config"][key] != accuracy_audit["config"][key]:
            raise ValueError("fast与eager共同TEST配置不符：" + key)
    if audit["manifest_sha256"] != accuracy_audit["manifest_sha256"]:
        raise ValueError("fast与eager TEST清单不同")
    if audit["config"]["methods"]["v6_fast"] != accuracy_audit["config"]["methods"]["v6_best"]:
        raise ValueError("fast不能换权重或输入模式")
    for source, expected in accuracy_audit["source_sha256"].items():
        if audit["source_sha256"][source] != expected:
            raise ValueError("fast的共同数据或指标源码不一致")
    rows = []
    for name, domain in summary["datasets"].items():
        value = report["datasets"][name]["methods"]["v6_fast"]
        eager = domain["methods"]["v6_best"]
        if value["frames"] != eager["frames"]:
            raise ValueError("fast测试帧数不同")
        fingerprints = []
        for folder in (main_folder, filename.parent):
            observed = {}
            for part in (folder / name).glob("part_*.json"):
                for frame in plots.read(part)["frames"]:
                    if "input_sha256" in frame:
                        observed[frame["index"]] = frame["input_sha256"]
            fingerprints.append(observed)
        if not fingerprints[0] or fingerprints[0] != fingerprints[1]:
            raise ValueError("fast与eager的首中末输入张量指纹不同")
        for region, metrics in value["depth"].items():
            if metrics["gt_pixels"] != eager["depth"][region]["gt_pixels"]:
                raise ValueError("fast的GT分母不相同")
            rows.append({"dataset": name, "units": domain["units"], "method": "v6_fast", "region": region,
                         "frames": value["frames"], **metrics,
                         **{key + "_delta_vs_eager": None if metrics[key] is None else metrics[key] - eager["depth"][region][key]
                            for key in ("MAE", "RMSE", "AbsRel", "coverage")}})
        domain["methods"]["v6_fast"] = value
    plots.csv_write(output / "fast_complete_test_depth.csv", rows)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("firstpass.json")))
    args = parser.parse_args()
    settings = plots.read(plots.path(args.config))
    plan = plots.read(plots.path(settings["evaluation_plan"]))
    output = plots.path(settings["output"]) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"completed_groups": [], "missing": [], "source_sha256": {str(Path(__file__)): plots.sha(__file__)}}
    notes = ["# MVSAnywhere 首遍、细化与 V6 的共同协议对比", "",
             "只汇报已完成的固定TEST。首遍与细化同权重、同TRAIN范围，仅细化次数0/1不同。",
             "计时必须来自本轮独占同卡完整四目；快速V6没有借用eager完整TEST精度。", ""]
    for phase in ("public", "carla_new"):
        main_folder = plots.path(plan["groups"][phase + "_main"]["output"]) / "full_test"
        first_folder = plots.path(settings["output"]) / phase / "full_test"
        files = [folder / filename for folder in (main_folder, first_folder) for filename in ("summary.json", "audit.json")]
        if not all(filename.exists() for filename in files):
            evidence["missing"].append(phase)
            continue
        combined, audit, _ = combine(main_folder, first_folder)
        folder = output / phase
        folder.mkdir(exist_ok=True)
        depths, stages, surfaces = plots.flatten(combined, phase, "test")
        plots.csv_write(folder / "depth_metrics.csv", depths)
        plots.csv_write(folder / "surface_metrics.csv", surfaces)
        plots.table_latex(depths, folder / "depth_table.tex")
        plot_settings = {"primary": "v6_best", "comparators": ["v5_best", "mvsanywhere_firstpass", "mvsanywhere_refined"]}
        plots.relative_heatmap(combined, plot_settings, folder, "firstpass_comparison")
        plots.relative_heatmap(combined, plot_settings, folder, "firstpass_distance", ("near", "mid", "far"))
        if phase == "carla_new":
            surface_comparison(combined, folder)
            paired_surface_frames(main_folder, first_folder, folder)
        fast_file = plots.path(settings["output"]).parent / "paper_fast_test" / phase / "full_test/summary.json"
        fast_audit = None
        if fast_file.exists() and plots.read(fast_file).get("completed"):
            fast_audit = add_fast_depth(combined, audit, fast_file, folder, main_folder)
            files.extend([fast_file, fast_file.with_name("audit.json")])
        if phase == "carla_new":
            speedfile = plots.path(settings["output"]) / "speed/results.json"
            if speedfile.exists() and plots.read(speedfile).get("completed"):
                plot_speed(combined, audit, plots.read(speedfile), folder, fast_audit)
                files.append(speedfile)
            else:
                notes.append("首遍独占测速尚未完成，当前没有精度—速度联合图。")
        notes.extend([f"## {phase}", "", f"![相对精度]({phase}/firstpass_comparison.png)", "",
                      f"[全指标CSV]({phase}/depth_metrics.csv) · [LaTeX]({phase}/depth_table.tex)", ""])
        if phase == "carla_new":
            notes.extend([f"![可见表面完整对比]({phase}/visible_surface_comparison.png)", "",
                          f"[表面CSV]({phase}/surface_metrics.csv) · [表面LaTeX]({phase}/surface_table.tex)", "",
                          "表面仅为相同GT可见范围的单参考深度反投影；每侧最多1024等间隔查询点对完整对侧点集。没有ICP、时序/多参考融合或官方mesh/occupancy评价。", ""])
        evidence["completed_groups"].append(phase)
        evidence["source_sha256"].update({str(filename): plots.sha(filename) for filename in files})
    notes.extend(["", "公共域原始单位不同，不合并MAE；CARLA四参考池化指标是多相机像素加权，不等于场景融合或独立场景统计。",
                  "部分方法预训练预算不同，性能主表不能单独归因架构。缺失组：" + str(evidence["missing"])])
    (output / "说明.md").write_text("\n".join(notes) + "\n")
    plots.write_json(output / "audit.json", evidence)
    print(output)


if __name__ == "__main__":
    main()
