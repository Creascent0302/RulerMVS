"""从实际完成的评测生成论文图表；不训练、不选择权重、不补造缺失结果。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rulermvs_paper_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "v2_best": "RulerMVS-V2", "v3_public": "V3 (public transfer)",
    "v4_best": "RulerMVS-V4", "v5_best": "RulerMVS-V5",
    "v6_best": "RulerMVS-V6 (VAL-best)", "v6_final": "RulerMVS-V6 (last)",
    "mvsanywhere_refined": "MVSAnywhere (refined)",
    "patchmatchnet_best": "PatchmatchNet (adapted)", "diffmvs_best": "DiffMVS (adapted)",
}
SHORT = {
    "TartanAir_forest": "TartanAir", "MatrixCity_street": "MatrixCity",
    "Hypersim_subset": "Hypersim", "MVS_Synth": "MVS-Synth",
    "VKITTI2": "VKITTI2", "BlendedMVS": "BlendedMVS",
    "DynamicReplica_valid": "DynamicReplica",
    "CARLA_new_upper": "CARLA up",
    **{f"CARLA_new_stereo_{side}": f"CARLA {side}" for side in ("up", "left", "right", "down")},
}
COLORS = {"v2_best": "#8395a7", "v4_best": "#9b75ac", "v5_best": "#d28b32",
          "v6_best": "#067a83", "v6_final": "#42b4aa", "mvsanywhere_refined": "#4351a3",
          "patchmatchnet_best": "#be677d", "diffmvs_best": "#797979"}
REGIONS = ("full", "near", "mid", "far", "gradient_band", "smooth")
METRICS = ("MAE", "RMSE", "AbsRel")
CMAP = LinearSegmentedColormap.from_list("improvement", ["#b54b50", "#fafafa", "#087c87"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "semibold", "axes.labelcolor": "#263442",
                     "axes.edgecolor": "#9aa4af", "text.color": "#263442",
                     "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def read(filename):
    return json.loads(Path(filename).read_text())


def sha(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def write_json(filename, value):
    Path(filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def csv_write(filename, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(filename).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, output, name, caption):
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, .07, 1, .93))
    fig.text(0.01, 0.005, caption, fontsize=7, color="#596575")
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def training_curves(training, output):
    """只使用每轮完整VAL；小样本中间监控不可与完整VAL连成曲线。"""
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, phase in enumerate(("public", "carla_new")):
        source = training / phase / "events.jsonl"
        events = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        val = [event for event in events if event.get("event") == "validation"
               and event.get("full") is True and event.get("split") == "val"]
        if not val:
            continue
        x = np.arange(1, len(val) + 1)
        risks = np.array([event["selection_risk"] for event in val])
        best = int(np.argmin(risks))
        axes[0, column].plot(x, risks, "o-", color=COLORS["v6_best"], ms=3, lw=1.6)
        axes[0, column].scatter(x[best], risks[best], marker="*", s=150, color="#d28b32", zorder=5)
        axes[0, column].set(title=f"{phase.replace('_', ' ').title()}: full-VAL selection risk",
                            xlabel="Completed epoch", ylabel="Selection risk (lower is better)")
        for index in range(1, len(val)):
            if val[index]["stage"] != val[index - 1]["stage"]:
                for axis in axes[:, column]:
                    axis.axvline(index + .5, color="#acb7c3", ls="--", lw=.8)
        names = [item["dataset"] for item in val[0]["domains"]]
        for name in names:
            values = [next(item for item in event["domains"] if item["dataset"] == name)
                      for event in val]
            first = values[0]["full"]["mae"]
            relative = [100 * (item["full"]["mae"] / first - 1) for item in values]
            axes[1, column].plot(x, relative, lw=1.2, label=SHORT.get(name, name))
            for epoch, (event, item) in enumerate(zip(val, values), 1):
                for region in ("full", "near", "mid", "far"):
                    rows.append({"phase": phase, "completed_epoch": epoch, "stage": event["stage"],
                                 "stage_epoch": event["epoch"], "dataset": name, "units": item["units"],
                                 "region": region, "selection_risk": event["selection_risk"],
                                 **item[region]})
        axes[1, column].axhline(0, color="#acb7c3", lw=.8)
        axes[1, column].set(title="Per-domain MAE change from first full VAL", xlabel="Completed epoch",
                            ylabel="MAE change (%)")
        axes[1, column].legend(fontsize=7, ncol=2, frameon=False)
        for axis in axes[:, column]:
            axis.grid(alpha=.2)
    save(fig, output, "training_full_val", "TRAINING VALIDATION ONLY. Star: minimum validation risk; dashed lines: training stages. Not TEST results.")
    csv_write(output / "training_full_val.csv", rows)


def flatten(summary, group, split):
    depth, stages, surfaces = [], [], []
    for dataset, domain in summary["datasets"].items():
        for method, values in domain["methods"].items():
            common = {"group": group, "split": split, "dataset": dataset, "method": method,
                      "units": domain["units"], "num_views": domain["num_views"], "frames": values["frames"]}
            depth.extend({**common, "region": region, **metrics} for region, metrics in values["depth"].items())
            stages.extend({**common, "stage": stage, **metrics}
                          for stage, metrics in values["stages_stride8"].items())
            surfaces.append({**common, **values["surface"]})
    return depth, stages, surfaces


def relative_heatmap(summary, settings, output, tag, regions=("full",)):
    primary = settings["primary"]
    comparators = settings["comparators"]
    datasets = list(summary["datasets"])
    fig, axes = plt.subplots(len(regions), len(METRICS), figsize=(13, 2.6 * len(regions) + .8),
                             squeeze=False, constrained_layout=True)
    for row, region in enumerate(regions):
        for col, metric in enumerate(METRICS):
            matrix = np.full((len(comparators), len(datasets)), np.nan)
            for j, name in enumerate(datasets):
                methods = summary["datasets"][name]["methods"]
                own = methods[primary]["depth"][region].get(metric)
                for i, comparator in enumerate(comparators):
                    other = methods.get(comparator, {}).get("depth", {}).get(region, {}).get(metric)
                    if own is not None and other is not None and other > 0:
                        matrix[i, j] = 100 * (other - own) / other
            bound = max(5., float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 5.
            ax = axes[row, col]
            pic = ax.imshow(matrix, cmap=CMAP, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
            ax.set_xticks(range(len(datasets)), [SHORT.get(name, name) for name in datasets], rotation=40, ha="right")
            ax.set_yticks(range(len(comparators)), [LABELS.get(name, name) for name in comparators] if col == 0 else [])
            ax.set_title(f"{region.replace('_', ' ').title()} / {metric}")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, "—" if not np.isfinite(value) else f"{value:+.1f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(value) > .62 * bound else "#263442")
            fig.colorbar(pic, ax=ax, shrink=.65, pad=.02, label="Error reduction (%)")
    save(fig, output, tag, f"{summary['split'].upper()} — V6 relative to each row. Positive = V6 improves; negative = worse. Domains retain their native units; no raw-error averaging.")


def stage_plot(summary, settings, output):
    datasets = list(summary["datasets"])
    methods = ["v2_best", "v4_best", "v5_best", settings["primary"]]
    rows = (len(datasets) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3 * rows + .3), squeeze=False, constrained_layout=True)
    for ax, name in zip(axes.flat, datasets):
        for method in methods:
            values = summary["datasets"][name]["methods"].get(method, {})
            values = values.get("stages_stride8", {})
            error = [values.get(stage, {}).get("MAE", np.nan) for stage in ("coarse", "fine", "final")]
            ax.plot(range(3), error, "o-", color=COLORS[method], lw=1.6 if method == settings["primary"] else 1.1,
                    ms=4, label=LABELS[method])
        unit = "m" if summary["datasets"][name]["units"] == "meters" else "scene unit"
        ax.set(title=SHORT.get(name, name), xticks=range(3), xticklabels=["Coarse", "Fine", "Final"], ylabel=f"MAE ({unit})")
        ax.grid(axis="y", alpha=.2)
    for ax in list(axes.flat)[len(datasets):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    save(fig, output, "stages_stride8", f"{summary['split'].upper()} — all three outputs sampled on the SAME stride-8 GT lattice. These are not full-resolution final MAEs.")


def paired_bootstrap(records, primary, comparator, repetitions, seed):
    """按真实sequence重采样；保留整条序列全部帧及像素权重。"""
    clusters = {}
    differences = []
    for sequence, index, frame in records:
        own = frame["methods"][primary]["depth"]["full"]
        other = frame["methods"][comparator]["depth"]["full"]
        if own["gt_pixels"] != other["gt_pixels"]:
            raise ValueError("配对方法GT分母不一致")
        if own["predicted_pixels"] != own["gt_pixels"] or other["predicted_pixels"] != other["gt_pixels"]:
            return {"status": "skipped_incomplete_prediction_coverage"}, []
        counts = own["gt_pixels"]
        if counts == 0:
            continue
        values = clusters.setdefault(sequence, np.zeros(3))
        values += [own["abs_sum"], other["abs_sum"], counts]
        differences.append({"sequence": sequence, "index": index, "gt_pixels": counts,
                            "own_mae": own["abs_sum"] / counts, "reference_mae": other["abs_sum"] / counts,
                            "mae_difference": (own["abs_sum"] - other["abs_sum"]) / counts})
    if not clusters:
        return {"status": "no_valid_pixels"}, differences
    matrix = np.stack(list(clusters.values()))
    total = matrix.sum(0)
    result = {"status": "descriptive_one_sequence" if len(matrix) < 2 else "sequence_bootstrap",
              "sequences": len(matrix), "frames": len(differences),
              "own_mae": total[0] / total[2], "reference_mae": total[1] / total[2],
              "relative_reduction_percent": 100 * (1 - total[0] / total[1]),
              "frame_win_fraction": float(np.mean([item["mae_difference"] < 0 for item in differences])),
              "ci_low": None, "ci_high": None, "bootstrap_repetitions": 0,
              "few_clusters_caution": len(matrix) < 5}
    if len(matrix) >= 2:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(repetitions):
            sample = matrix[rng.integers(0, len(matrix), len(matrix))].sum(0)
            estimates.append(100 * (1 - sample[0] / sample[1]))
        low, high = np.quantile(estimates, [.025, .975])
        result.update(ci_low=float(low), ci_high=float(high), bootstrap_repetitions=repetitions)
    return result, differences


def scene_cluster(dataset, sequence):
    """同一场景的天气、视角或相机变体不能被视为独立场景。"""
    if dataset == "VKITTI2":
        return "/".join(sequence.split("/")[:2])
    if dataset == "Hypersim_subset":
        return sequence.rsplit("_cam_", 1)[0]
    return sequence


def paired_analysis(summary, audit, folder, settings, output):
    results, differences, source_hashes = [], [], {}
    domains = {item["name"]: item for item in audit["config"]["datasets"]}
    for name, dataset in summary["datasets"].items():
        manifest = path(domains[name]["manifests"][summary["split"]])
        samples = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        source_hashes[str(manifest)] = sha(manifest)
        if any(not sample.get("sequence") for sample in samples):
            results.append({"dataset": name, "status": "missing_sequence_metadata"})
            continue
        frames = {}
        for part in sorted((folder / name).glob("part_*.json")):
            data = read(part)
            if not data["completed"] or data["protocol"] != summary["protocol"]:
                raise ValueError(f"分片未完成或协议不一致：{part}")
            source_hashes[str(part)] = sha(part)
            for frame in data["frames"]:
                if frame["index"] in frames:
                    raise ValueError("重复帧不应进入统计")
                frames[frame["index"]] = frame
        if set(frames) != set(range(len(samples))):
            raise ValueError(f"{name}分片帧不完整")
        records = [(scene_cluster(name, samples[index]["sequence"]), index, frames[index])
                   for index in sorted(frames)]
        for comparator in settings["paired_comparators"]:
            if comparator not in dataset["methods"]:
                continue
            result, paired = paired_bootstrap(records, settings["primary"], comparator,
                                              settings["bootstrap_repetitions"], settings["seed"])
            result.update(dataset=name, comparator=comparator, units=dataset["units"])
            results.append(result)
            differences.extend({"dataset": name, "comparator": comparator, "units": dataset["units"], **item}
                               for item in paired)
    csv_write(output / "paired_sequence_bootstrap.csv", results)
    csv_write(output / "paired_frame_differences.csv", differences)
    if not results:
        return source_hashes
    names = list(summary["datasets"])
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for j, comparator in enumerate(settings["paired_comparators"]):
        for i, name in enumerate(names):
            match = next((item for item in results if item.get("dataset") == name
                          and item.get("comparator") == comparator and "relative_reduction_percent" in item), None)
            if match is None:
                continue
            value = match["relative_reduction_percent"]
            y = i + (j - .5) * .22
            ax.scatter(value, y, s=35, color=COLORS[comparator], marker="o" if match["sequences"] >= 2 else "x",
                       label=LABELS[comparator] if i == 0 else None)
            if match["ci_low"] is not None:
                ax.hlines(y, match["ci_low"], match["ci_high"], color=COLORS[comparator], lw=2)
            ax.annotate(f"n={match['sequences']}", (value, y), xytext=(5, 4), textcoords="offset points", fontsize=6)
    ax.axvline(0, color="#747f8d", ls="--", lw=1)
    ax.set(yticks=range(len(names)), yticklabels=[SHORT.get(name, name) for name in names],
           xlabel="V6 MAE reduction relative to comparator (%) — right is better",
           title=f"{summary['split'].upper()}: paired scene bootstrap (95% percentile interval)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.legend(frameon=False, loc="best")
    save(fig, output, "paired_sequence_intervals", "n = scenes / trajectories (VKITTI2 variants merged), not pixels. n=1: descriptive point only; n<5: fragile intervals. No pixel-independent confidence claims.")
    return source_hashes


def table_latex(rows, filename):
    lines = ["% Generated from actual completed evaluation; MAE uses predicted pixels. Report coverage.",
             "\\begin{tabular}{llrrrr}", "\\toprule",
             "Dataset / units & Method & MAE $\\downarrow$ & RMSE $\\downarrow$ & AbsRel $\\downarrow$ & Coverage \\\\",
             "\\midrule"]
    for row in rows:
        if row["region"] != "full":
            continue
        name = SHORT.get(row["dataset"], row["dataset"]).replace("_", "\\_")
        unit = "m" if row["units"] == "meters" else "scene unit"
        method = LABELS.get(row["method"], row["method"]).replace("_", "\\_")
        values = " & ".join(f"{row.get(metric, float('nan')):.4f}" for metric in ("MAE", "RMSE", "AbsRel"))
        values += f" & {row.get('coverage', float('nan')):.6f}"
        lines.append(f"{name} ({unit}) & {method} & {values} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    filename.write_text("\n".join(lines) + "\n")


def speed_pareto(summary, accuracy_audit, latency_path, output):
    """只读取这轮已完成的同组测速；不从历史或单参考乘四填入。"""
    if not latency_path.exists():
        return "本轮测速未完成，未生成速度或Pareto图。"
    speed = read(latency_path)
    if not speed.get("completed"):
        return "本轮测速尚未完成，未生成速度或Pareto图。"
    timing_audit = read(latency_path.parent / "audit.json")
    for key in ("image_size", "matching_image_size", "methods"):
        if timing_audit["config"][key] != accuracy_audit["config"][key]:
            raise ValueError(f"测速与精度的{key}不一致，拒绝合并")
    for method, expected in accuracy_audit["weights_sha256"].items():
        weight = accuracy_audit["config"]["methods"].get(method, {}).get("weights", method)
        if timing_audit["checkpoint_sha256"].get(weight) != expected:
            raise ValueError("测速与精度权重不一致，拒绝合并")
    names = list(summary["datasets"])
    rows = (len(names) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3 * rows + .5), squeeze=False, constrained_layout=True)
    table = []
    for ax, name in zip(axes.flat, names):
        timed = speed["datasets"].get(name)
        if timed is None:
            ax.axis("off")
            continue
        actual_views = sorted({read(filename)["valid_views"] for filename in
                               (latency_path.parent / name).glob("frame_*.json")})
        points = []
        for method, values in summary["datasets"][name]["methods"].items():
            if method not in timed["summary"]:
                continue
            stat = timed["summary"][method]["wall_network_ms"]
            milliseconds = stat["median_ms"]
            mae = values["depth"]["full"]["MAE"]
            table.append({"dataset": name, "method": method, "split": summary["split"],
                          "units": summary["datasets"][name]["units"], "MAE": mae,
                          "network_ms": milliseconds, "network_p90_ms": stat["p90_ms"],
                          "single_reference_hz": stat["single_reference_hz"],
                          "timing_actual_views": "/".join(map(str, actual_views)),
                          "coverage": values["depth"]["full"]["coverage"]})
            points.append((milliseconds, mae))
            ax.scatter(milliseconds, mae, c=COLORS.get(method, "#777777"), s=38,
                       marker="*" if method == "v6_best" else "o", label=LABELS.get(method, method))
        frontier, best = [], float("inf")
        for milliseconds, mae in sorted(points):
            if mae < best:
                frontier.append((milliseconds, mae))
                best = mae
        if frontier:
            ax.plot(*np.array(frontier).T, color="#9aa4af", ls="--", lw=1, zorder=0)
        unit = "m" if summary["datasets"][name]["units"] == "meters" else "scene unit"
        ax.set(title=f"{SHORT.get(name, name)} | timing N={'/'.join(map(str, actual_views))}",
               xlabel="Single-reference latency (ms)", ylabel=f"MAE ({unit})")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, fontsize=7)
    save(fig, output, "accuracy_latency", f"{summary['split'].upper()} accuracy + same-run eager timing. Lower-left is better. Native task pretraining/training budgets differ; network + adapter, not end-to-end vehicle latency.")
    csv_write(output / "accuracy_latency.csv", table)
    return "已使用同轮完成的eager网络+适配耗时；未混入fast部署耗时或历史测速。"


def mechanism_figures(filename, output):
    if not filename.exists() or not read(filename).get("completed"):
        return "机制探针尚未完成，未生成干预图。"
    data = read(filename)
    methods = ["v5_trained", "v6_trained", "v6_no_parent", "v6_flat_readout"]
    labels = ["V5", "V6", "V6: no parent evidence", "V6: flat readout"]
    colors = ["#d28b32", "#067a83", "#9b75ac", "#be677d"]
    fields = [("coarse_good_destroyed_fraction", "Correct coarse to wrong fine (%)", 100),
              ("coarse_bad_corrected_fraction", "Wrong coarse to correct fine (%)", 100),
              ("continuous_coverage_5pct", "Continuous candidate coverage (%)", 100),
              ("group_selection_regret", "Group selection regret / V5", 1)]

    def value_of(record, field):
        if field == "coarse_bad_corrected_fraction":
            wrong = 1 - record["coarse_good"]
            return record["coarse_bad_fine_good"] / wrong if wrong > 0 else None
        return record.get(field)

    rows = []
    for group in ("public", "carla"):
        names = [name for name in data["datasets"] if name.startswith("CARLA") == (group == "carla")]
        if not names:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
        for ax, (field, title, factor) in zip(axes, fields):
            for method, label, color in zip(methods, labels, colors):
                values = []
                for name in names:
                    domain = data["datasets"][name]
                    raw = value_of(domain["methods"][method]["candidates_stride8"]["full"], field)
                    base = value_of(domain["methods"]["v5_trained"]["candidates_stride8"]["full"], field)
                    value = np.nan if raw is None else raw * factor
                    if field == "group_selection_regret":
                        value = raw / base if raw is not None and base else np.nan
                    values.append(value)
                    rows.append({"group": group, "dataset": name, "method": method, "field": field,
                                 "raw_value": raw, "units": domain["units"] if field == "group_selection_regret" else "fraction",
                                 "plotted_value": value})
                ax.plot(range(len(names)), values, "o-", color=color, label=label, ms=4, lw=1.3)
            ax.set(xticks=range(len(names)), xticklabels=[SHORT.get(name, name) for name in names], ylabel=title)
            ax.grid(axis="y", alpha=.2)
            if field == "group_selection_regret":
                ax.axhline(1, color="#9aa4af", ls="--", lw=.8)
        handles, legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend, loc="outside upper center", ncol=4, frameon=False, fontsize=8)
        save(fig, output, f"mechanism_{group}", "Fixed VAL subset. V6 interventions share one trained checkpoint; NOT retrained ablations. V5/V6 training budgets differ; oracle coverage is not deployed accuracy.")
    csv_write(output / "mechanism_metrics.csv", rows)
    tradeoffs = []
    for name, domain in data["datasets"].items():
        methods = domain["methods"]
        parent = methods["v6_trained"]["candidates_stride8"]["full"]
        without = methods["v6_no_parent"]["candidates_stride8"]["full"]
        correction = [value_of(record, "coarse_bad_corrected_fraction") for record in (parent, without)]
        error = [methods[method]["depth"]["full"]["MAE"] for method in ("v6_trained", "v6_no_parent")]
        if any(value is None for value in correction):
            continue
        tradeoffs.append({"dataset": name,
                          "retention_gain_pp": 100 * (without["coarse_good_destroyed_fraction"] - parent["coarse_good_destroyed_fraction"]),
                          "correction_gain_pp": 100 * (correction[0] - correction[1]),
                          "full_mae_gain_percent": 100 * (1 - error[0] / error[1])})
    csv_write(output / "mechanism_retention_correction.csv", tradeoffs)
    if tradeoffs:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        bound = max(abs(row["full_mae_gain_percent"]) for row in tradeoffs) or 1.
        pic = ax.scatter([row["retention_gain_pp"] for row in tradeoffs],
                         [row["correction_gain_pp"] for row in tradeoffs],
                         c=[row["full_mae_gain_percent"] for row in tradeoffs], s=90, cmap=CMAP,
                         norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), edgecolor="white", linewidth=.8)
        for i, row in enumerate(tradeoffs):
            ax.annotate(SHORT.get(row["dataset"], row["dataset"]),
                        (row["retention_gain_pp"], row["correction_gain_pp"]), xytext=(5, 7 if i % 2 else -12),
                        textcoords="offset points", fontsize=8)
        ax.axhline(0, color="#9aa4af", ls="--", lw=.8)
        ax.axvline(0, color="#9aa4af", ls="--", lw=.8)
        ax.set(xlabel="Improved retention of correct coarse estimates (percentage points)",
               ylabel="Improved correction of wrong coarse estimates (percentage points)",
               title="Parent evidence: preservation versus correction")
        ax.grid(alpha=.2)
        fig.colorbar(pic, ax=ax, label="Full-resolution MAE reduction (%)", shrink=.75)
        save(fig, output, "mechanism_retention_vs_correction", "Same V6 weights; parent evidence on versus off. Right preserves more correct coarse depths; up corrects more wrong depths. Fixed 16-frame VAL subsets, not retrained ablations.")
    return "机制图来自固定VAL推理干预，不能替代同预算重新训练的模块消融。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = parser.parse_args()
    settings = read(args.config)
    output = path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    training_curves(path(settings["training_root"]), output)
    evaluation = path(settings["evaluation_root"])
    manifest = {"settings": settings, "groups": {}, "missing": [], "source_sha256": {str(Path(__file__)): sha(__file__)}}
    notes = ["# 论文图表的证据与使用边界", "", f"来源说明：{settings['provenance_label']}", "",
             "所有图直接读取已完成的原始统计；缺失评测不填数。训练曲线只取full=true的完整VAL。",
             "相对改善=(对照误差−V6误差)/对照误差；正值更好。不同单位的原始误差不做平均。",
             "MAE/RMSE统计使用预测有效像素，表中同时提供coverage、gt_pixels与predicted_pixels。",
             "主表方法预训练及追加训练预算不同，不能直接证明结构的因果收益。",
             "gradient_band来自GT深度变化，不等于小物体实例或语义障碍物；阶段图只比相同stride8采样点。",
             "按场景/轨迹整体重采样，不按像素独立bootstrap；VKITTI2同Scene天气/视角合并、Hypersim同场景相机合并。只有一个独立组时不提供置信区间，少于五组须谨慎解释。", ""]
    for group in settings["groups"]:
        for split in settings["splits"]:
            tag = f"{group}_{split}"
            folder = evaluation / group / f"full_{split}"
            summary_path = folder / "summary.json"
            if not summary_path.exists() or not read(summary_path).get("completed"):
                manifest["missing"].append(tag)
                continue
            summary, audit = read(summary_path), read(folder / "audit.json")
            if summary["split"] != split:
                raise ValueError("目录和实际split不一致")
            destination = output / tag
            destination.mkdir(exist_ok=True)
            depths, stages, surfaces = flatten(summary, group, split)
            csv_write(destination / "depth_metrics.csv", depths)
            csv_write(destination / "stage_metrics.csv", stages)
            csv_write(destination / "surface_metrics.csv", surfaces)
            table_latex(depths, destination / "depth_table.tex")
            if all(settings["primary"] in domain["methods"] for domain in summary["datasets"].values()):
                relative_heatmap(summary, settings, destination, "domain_relative_errors")
                relative_heatmap(summary, settings, destination, "distance_relative_errors", ("near", "mid", "far"))
                relative_heatmap(summary, settings, destination, "boundary_relative_errors", ("gradient_band", "smooth"))
                stage_plot(summary, settings, destination)
                sources = paired_analysis(summary, audit, folder, settings, destination)
                manifest["source_sha256"].update(sources)
            timing_note = speed_pareto(summary, audit, evaluation / group / "latency/latency.json", destination)
            manifest["source_sha256"].update({str(summary_path): sha(summary_path), str(folder / "audit.json"): sha(folder / "audit.json")})
            manifest["groups"][tag] = {"split": split, "domains": list(summary["datasets"]), "protocol": summary["protocol"], "timing": timing_note}
            notes.extend([f"## {tag}", "", f"完整{split.upper()}，{len(summary['datasets'])}个域/参考；协议hash：`{summary['protocol']}`。", "", timing_note, ""])
            if (destination / "domain_relative_errors.png").exists():
                notes.extend([f"![相对误差热图]({tag}/domain_relative_errors.png)", "",
                              f"[近中远分析]({tag}/distance_relative_errors.pdf) · "
                              f"[边界与平滑区]({tag}/boundary_relative_errors.pdf) · "
                              f"[共同格点阶段分析]({tag}/stages_stride8.pdf) · "
                              f"[场景配对统计]({tag}/paired_sequence_intervals.pdf)", ""])
            notes.extend([f"[完整深度表 CSV]({tag}/depth_metrics.csv) · "
                          f"[表面几何表 CSV]({tag}/surface_metrics.csv) · "
                          f"[LaTeX 表格]({tag}/depth_table.tex)", ""])
            print(f"完成 {tag}", flush=True)
    notes.extend(["## 当前尚缺", "", ", ".join(manifest["missing"]) or "配置中的完整评测均已生成图表。", "",
                  "缺失项是未完成或尚未生成结果，不表示方法误差为零。VAL结果用于开发，不能在论文中改名为TEST结果。"])
    if settings.get("mechanism_result"):
        filename = path(settings["mechanism_result"])
        notes.extend(["", mechanism_figures(filename, output)])
        if filename.exists():
            manifest["source_sha256"][str(filename)] = sha(filename)
    (output / "图表说明.md").write_text("\n".join(notes) + "\n")
    write_json(output / "analysis_manifest.json", manifest)
    print(output, flush=True)


if __name__ == "__main__":
    main()
