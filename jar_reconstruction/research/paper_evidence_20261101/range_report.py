"""从已完成的范围诊断生成表格、独立图及解释，不再次推理。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate(result_path):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    if not result["completed"] or result["split"] != "val" or result["test_accessed"]:
        raise ValueError("只接受已完成的VAL范围诊断")
    output = result_path.parent / "analysis"
    output.mkdir(exist_ok=True)
    rows, range_details = [], {}
    for name, data in result["datasets"].items():
        low, high = data["train_range"]
        far_start = 40. if data["units"] == "meters" else 20 * data["depth_scale"]
        detail = {"expanded_upper_frames": 0, "expanded_lower_frames": 0,
                  "narrowed_upper_frames": 0, "narrowed_lower_frames": 0,
                  "far_pixels_definitely_above_final_range": 0,
                  "frames_final_upper_below_far_start": 0,
                  "frames_final_upper_below_far_start_with_far_gt": 0, "far_start": far_start}
        for frame in data["frames"]:
            item = frame["methods"]["mvsanywhere_refined"]
            calls = item["native_cost_volume_ranges"]
            if not calls:
                continue
            lower, upper = calls[-1]["min_depth"][0], calls[-1]["max_depth"][0]
            detail["expanded_upper_frames"] += upper > high * (1 + 1e-5)
            detail["expanded_lower_frames"] += lower < low * (1 - 1e-5)
            detail["narrowed_upper_frames"] += upper < high * (1 - 1e-5)
            detail["narrowed_lower_frames"] += lower > low * (1 + 1e-5)
            far_pixels = item["depth"]["far"]["raw"]["gt_pixels"]
            detail["frames_final_upper_below_far_start"] += upper < far_start
            if upper < far_start and far_pixels:
                detail["frames_final_upper_below_far_start_with_far_gt"] += 1
                detail["far_pixels_definitely_above_final_range"] += far_pixels
        count = data["methods"]["mvsanywhere_refined"]["raw"]["far"]["gt_pixels"]
        detail["far_gt_pixels"] = count
        detail["far_pixels_outside_final_range_lower_bound_fraction"] = (
            detail["far_pixels_definitely_above_final_range"] / count if count else None)
        range_details[name] = detail
        for method, values in data["methods"].items():
            for region, raw in values["raw"].items():
                clipped, audit = values["clipped"][region], values["range"][region]
                rows.append({"dataset": name, "units": data["units"], "method": method,
                             "region": region, "raw_MAE": raw["MAE"], "clipped_MAE": clipped["MAE"],
                             "raw_RMSE": raw["RMSE"], "clipped_RMSE": clipped["RMSE"],
                             "raw_AbsRel": raw["AbsRel"], "clipped_AbsRel": clipped["AbsRel"],
                             "MAE_reduction": audit["clip_MAE_reduction"],
                             "prediction_outside_fraction": audit["pred_outside_train_fraction"],
                             "gt_outside_fraction": audit["gt_outside_train_fraction"],
                             "outside_abs_error_share": audit["outside_abs_error_share"],
                             "gt_pixels": raw["gt_pixels"], "predicted_pixels": raw["predicted_pixels"]})
    with (output / "range_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"split": "val", "frames": 176, "test_accessed": False,
               "datasets": range_details,
               "v6_all_clipping_metrics_unchanged": all(r["raw_MAE"] == r["clipped_MAE"]
                    and r["raw_RMSE"] == r["clipped_RMSE"] and r["raw_AbsRel"] == r["clipped_AbsRel"]
                    for r in rows if r["method"] == "v6_best"),
               "mvs_all_far_clipping_metrics_unchanged": all(r["raw_MAE"] == r["clipped_MAE"]
                    and r["raw_RMSE"] == r["clipped_RMSE"] and r["raw_AbsRel"] == r["clipped_AbsRel"]
                    for r in rows if r["method"] == "mvsanywhere_refined" and r["region"] == "far")}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    plot_tartan(result["datasets"]["TartanAir_forest"], range_details["TartanAir_forest"], output)
    text = """# 176组固定VAL范围诊断

两方法都具有输出范围约束。实际差异是V6固定于TRAIN范围，而MVSAnywhere每遍限制于当前范围、官方细化可缩放该范围；不是DPT无界输出。

- V6在11域评分有效像素上无TRAIN越界，裁剪前后所有统计指标完全不变。
- MVSAnywhere在10域无越界；仅CARLA下参考67个像素低于TRAIN下界，占有效像素0.00672%。裁剪后该域full MAE仅改善4.54e-7米。
- 所有域的远距MAE、RMSE、AbsRel在裁剪前后完全不变。因此这176VAL不支持“V6远距优势来自抑制baseline范围越界”的解释。
- 另一项范围作用仍须区分：Tartan的16帧均收紧上界，其中15帧最终上界低于40米，14帧同时具有远距GT。这14帧合计49871个远距GT点，占该子集55761个远距GT点的89.44%；它们必然落在MVS最终范围之外。这只是可由日志推出的覆盖率下界，不是完整GT与范围逐点覆盖统计。
- Tartan上MVS远距MAE为102.82684米、signed bias为−102.82684米，显示远距整体严重低估；V6为93.36982米，同样误差很大。现有证据支持进一步比较首遍/细化与GT最终范围覆盖，不能据此宣称关闭细化必定更好。
- Matrix/MVS-Synth/VKITTI2的范围内远距误差仍有明显差异，但该裁剪实验不能把差异直接归因于标尺或候选交接，也不能排除范围收缩对内部搜索的影响。

上述均为预登记16帧/域、共176组VAL诊断，不是完整TEST，不能替换正式主表。原始结果未改写，源码版本事项见上级目录audit_note.json。

图tartan_range_diagnostic展示各帧MVS最终范围、40米远距边界、TRAIN边界与远距GT计数。它描述一个具体失败机制，不能代表所有数据域。
"""
    (output / "summary_zh.md").write_text(text)
    print(json.dumps(summary, ensure_ascii=False))


def plot_tartan(data, details, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    x = np.arange(1, len(data["frames"]) + 1)
    calls = [f["methods"]["mvsanywhere_refined"]["native_cost_volume_ranges"][-1] for f in data["frames"]]
    low = np.array([c["min_depth"][0] for c in calls])
    high = np.array([c["max_depth"][0] for c in calls])
    counts = np.array([f["methods"]["mvsanywhere_refined"]["depth"]["far"]["raw"]["gt_pixels"]
                       for f in data["frames"]])
    excluded = high < 40
    colors = np.where(excluded, "#D55E00", "#0072B2")
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    axes[0].axhspan(40, 400, color="#ECECEC", alpha=.7)
    axes[0].vlines(x, low, high, color=colors, linewidth=3, alpha=.85)
    axes[0].scatter(x, high, c=colors, s=22, zorder=3)
    axes[0].axhline(40, color="#555555", linestyle="--", linewidth=1, label="Far-depth boundary (40 m)")
    axes[0].axhline(data["train_range"][1], color="#009E73", linestyle=":", linewidth=1.2,
                   label="Shared initial TRAIN upper bound")
    axes[0].set_yscale("log")
    axes[0].set_ylim(.3, 450)
    axes[0].set_ylabel("Final native depth range (m)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_title("TartanAir: range contraction can exclude distant surfaces", loc="left", fontweight="bold")
    axes[1].bar(x, counts, color=colors, width=.65)
    axes[1].set_ylabel("Valid far-depth GT pixels")
    axes[1].set_xlabel("Pre-registered VAL sample order (16 frames)")
    axes[1].set_xticks(x)
    axes[1].text(.015, .94, "Orange: final upper bound < 40 m\nAt least 89.44% of far GT pixels exceed the final range",
                 transform=axes[1].transAxes, va="top", fontsize=8,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88})
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / ("tartan_range_diagnostic." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    generate(parser.parse_args().results)
