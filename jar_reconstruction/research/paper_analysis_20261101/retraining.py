"""只读取真实完成的论文重训消融；三seed与单seed、完整VAL与机制子集分开。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from research.paper_analysis_20261101 import analyze as plots


VARIANTS = ("v5", "marginal", "constant", "parent", "parent_oldpoint")
LABELS = ("V5", "Marginal", "Constant gate", "Adaptive gate", "Adaptive + old loss")
COLORS = ("#9ba7b4", "#7f86c2", "#cc963f", "#067a83", "#ae665e")


def heatmap(matrix, rows, columns, output, name, title, caption):
    values = np.asarray(matrix, dtype=float)
    bound = max(1., float(np.nanmax(np.abs(values))))
    fig, axis = plots.plt.subplots(figsize=(max(7, len(columns) * 1.35), len(rows) * .65 + 2.2), constrained_layout=True)
    picture = axis.imshow(values, cmap=plots.CMAP, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    axis.set(xticks=range(len(columns)), yticks=range(len(rows)), yticklabels=rows, title=title)
    axis.set_xticklabels(columns, rotation=30, ha="right")
    for (i, j), value in np.ndenumerate(values):
        axis.text(j, i, "—" if not np.isfinite(value) else f"{value:+.2f}", ha="center", va="center",
                  color="white" if abs(value) > .63 * bound else "#263442", fontsize=9)
    fig.colorbar(picture, ax=axis, shrink=.8, label="Relative MAE reduction (%)")
    plots.save(fig, output, name, caption)


def handoff(report, output):
    if not report.get("completed") or report.get("test_accessed") or len(report["seeds"]) != 3:
        raise ValueError("交接绘图需要三seed完整VAL，不能把机制子集当全量")
    rows = []
    for phase in ("public", "carla"):
        domains = {key: value for key, value in report["datasets"].items() if value["phase"] == phase}
        fig, axes = plots.plt.subplots((len(domains) + 3) // 4, 4, squeeze=False,
                                      figsize=(14, 3.4 * ((len(domains) + 3) // 4)), constrained_layout=True)
        for axis, (name, domain) in zip(axes.flat, domains.items()):
            for i, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
                value = domain["full_validation"][variant]["depth"]["full"]["MAE"]
                coverage = domain["full_validation"][variant]["depth"]["full"]["coverage"]
                axis.errorbar(i, value["mean"], yerr=value["sample_std"], fmt="o", color=color, capsize=4, ms=5)
                axis.scatter(i + np.array([-.12, 0, .12]), value["values"], s=13, color=color, alpha=.5)
                rows.append({"dataset": name, "phase": phase, "variant": variant, "units": domain["units"],
                             "MAE_mean": value["mean"], "MAE_sample_std": value["sample_std"],
                             "coverage_mean": coverage["mean"], "frames": domain["frames"],
                             **{str(seed): number for seed, number in zip(value["seeds"], value["values"])}})
            axis.set(title=plots.SHORT.get(name, name), xticks=range(5),
                     ylabel="MAE (m)" if domain["units"] == "meters" else "MAE (scene unit)")
            axis.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=8)
            axis.grid(axis="y", alpha=.2)
        for axis in list(axes.flat)[len(domains):]:
            axis.axis("off")
        plots.save(fig, output, f"handoff_{phase}_full_val", "Complete VAL, fixed last step; mean ± sample SD over 3 seeds (dots), not a confidence interval. Common V5 warm start, frozen encoder; not training from scratch.")

        matrix = []
        names = list(domains)
        for variant in VARIANTS[1:]:
            line = []
            for domain in domains.values():
                own = np.asarray(domain["full_validation"][variant]["depth"]["full"]["MAE"]["values"])
                base = np.asarray(domain["full_validation"]["v5"]["depth"]["full"]["MAE"]["values"])
                line.append(np.mean(100 * (1 - own / base)))
            matrix.append(line)
        adaptive_vs_constant = []
        for domain in domains.values():
            own = np.asarray(domain["full_validation"]["parent"]["depth"]["full"]["MAE"]["values"])
            base = np.asarray(domain["full_validation"]["constant"]["depth"]["full"]["MAE"]["values"])
            adaptive_vs_constant.append(np.mean(100 * (1 - own / base)))
        matrix.append(adaptive_vs_constant)
        comparison_labels = [label + " / V5" for label in LABELS[1:]] + ["Adaptive / Constant"]
        heatmap(matrix, comparison_labels, [plots.SHORT.get(name, name) for name in names], output,
                f"handoff_{phase}_paired_reduction", "Complete VAL: paired-seed MAE reduction",
                "Each row compares the method before / against the reference after /. Mean over three paired seeds; positive = improvement. Marginal/constant/adaptive share the group objective; V5 uses the old objective.")

        fig, axes = plots.plt.subplots(1, 2, figsize=(14, 4.9), constrained_layout=True)
        common_rows = []
        for axis, key, sign, title in zip(axes, ("destroyed_rate_paired_delta", "corrected_rate_paired_delta"),
                                          (-1, 1), ("Protection on common coarse-correct pixels", "Correction on common coarse-wrong pixels")):
            for i, (variant, label, color) in enumerate(zip(VARIANTS[1:], LABELS[1:], COLORS[1:])):
                values = [domain["paired_candidate_intersections"][variant]["full"][key] for domain in domains.values()]
                means = np.array([value["mean"] for value in values]) * sign * 100
                standard = np.array([value["sample_std"] for value in values]) * 100
                axis.errorbar(np.arange(len(names)) + (i - 1.5) * .16, means, yerr=standard,
                              fmt="o", ms=4, capsize=3, color=color, label=label)
                for name, value, mean in zip(names, values, means):
                    common_rows.append({"dataset": name, "variant": variant, "event": key,
                                        "improvement_pp_mean": mean, "sample_std_pp": value["sample_std"] * 100,
                                        "seeds": str(value["seeds"]), "raw_delta_values": str(value["values"])})
            axis.axhline(0, lw=.9, color="#66727e")
            axis.set(title=title, xticks=range(len(names)), ylabel="Improvement over V5 (percentage points)")
            axis.set_xticklabels([plots.SHORT.get(name, name) for name in names], rotation=30, ha="right")
            axis.grid(axis="y", alpha=.2)
        handles, legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend, loc="outside upper center", ncol=2, frameon=False)
        plots.save(fig, output, f"handoff_{phase}_common_support", "Complete VAL, mean ± sample SD over 3 seeds. Positive is better on both panels. Each method/V5 pair uses intersected coarse-correct or coarse-wrong pixels; these conditioning sets can differ across pairs.")
        plots.csv_write(output / f"handoff_{phase}_common_support.csv", common_rows)

        fig, axes = plots.plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for axis, direction in zip(axes, ("damage", "correction")):
            matrix = []
            for variant in VARIANTS:
                line = []
                for domain in domains.values():
                    metrics = domain["mechanism_validation"][variant]["full"]
                    if direction == "damage":
                        values = metrics["coarse_good_destroyed_fraction"]["values"]
                    else:
                        good = np.asarray(metrics["coarse_good"]["values"])
                        repaired = np.asarray(metrics["coarse_bad_fine_good"]["values"])
                        values = np.divide(repaired, 1 - good, out=np.full_like(good, np.nan), where=good < 1)
                    line.append(100 * np.mean(values))
                matrix.append(line)
            matrix = np.asarray(matrix)
            image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0)
            axis.set(xticks=range(len(names)), yticks=range(5), yticklabels=LABELS if direction == "damage" else [],
                     title="Correct coarse → wrong fine (↓)" if direction == "damage" else "Wrong coarse → correct fine (↑)")
            axis.set_xticklabels([plots.SHORT.get(name, name) for name in names], rotation=35, ha="right")
            for (i, j), value in np.ndenumerate(matrix):
                axis.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                          color="white" if value > .62 * np.nanmax(matrix) else "#263442")
            fig.colorbar(image, ax=axis, shrink=.75, label="Conditional rate (%)")
        plots.save(fig, output, f"handoff_{phase}_mechanism", "Fixed 16-frame-per-domain mechanism VAL subset, not full VAL. Correct = relative depth error ≤5%. Conditional denominators differ; both retention and correction must be reported.")
    plots.csv_write(output / "handoff_full_val.csv", rows)
    lines = ["% Requires booktabs. No statistical significance claims.",
             "\\begin{table*}[t]", "\\centering",
             "\\caption{Complete validation MAE after matched adaptation. Mean $\\pm$ sample standard deviation over three paired training seeds; units are shown per dataset. Fixed last-step checkpoints, not TEST-based selection.}",
             "\\label{tab:handoff-full-val}", "\\begin{tabular}{llrrrrr}", "\\toprule",
             "Dataset & Unit & V5 & Marginal & Constant gate & Adaptive gate & Adaptive + old loss \\\\", "\\midrule"]
    for name, domain in report["datasets"].items():
        cells = []
        for variant in VARIANTS:
            value = domain["full_validation"][variant]["depth"]["full"]["MAE"]
            cells.append(f"${value['mean']:.4f} \\pm {value['sample_std']:.4f}$")
        unit = "m" if domain["units"] == "meters" else "scene unit"
        lines.append(" & ".join([plots.SHORT.get(name, name), unit, *cells]) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (output / "handoff_full_val.tex").write_text("\n".join(lines) + "\n")


def geometry(report, output):
    if not report.get("complete_training") or not report.get("complete_full_val") or report.get("test_accessed"):
        raise ValueError("标尺绘图只接受全部训练与完整VAL真实完成的结果")
    variants = ("no_source_geometry", "no_scale_routing", "no_jacobian_width")
    labels = ("Source geometry embedding off", "Scale routing off", "Jacobian width off")
    selected = [row for row in report["rows"] if row["scope"] == "full_val"]
    for phase in ("public", "carla"):
        rows = [row for row in selected if row["phase"] == phase]
        names = list(dict.fromkeys(row["dataset"] for row in rows))
        lookup = {(row["dataset"], row["variant"], row["region"]): row for row in rows}
        for region in ("full", "near", "mid", "far"):
            if region == "full":
                fig, axes = plots.plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
                for axis, metric in zip(axes, ("MAE", "RMSE", "AbsRel")):
                    matrix = np.array([[-lookup[name, variant, region][metric + "_change_vs_control_pct"]
                                        for name in names] for variant in variants])
                    bound = max(1., float(np.nanmax(np.abs(matrix))))
                    picture = axis.imshow(matrix, cmap=plots.CMAP, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
                    axis.set(title=metric, xticks=range(len(names)), yticks=range(3),
                             yticklabels=labels if metric == "MAE" else [])
                    axis.set_xticklabels([plots.SHORT.get(name, name) for name in names], rotation=35, ha="right")
                    for (i, j), value in np.ndenumerate(matrix):
                        axis.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8,
                                  color="white" if abs(value) > .63 * bound else "#263442")
                    fig.colorbar(picture, ax=axis, shrink=.8, label="Error reduction (%)")
                plots.save(fig, output, f"geometry_{phase}_full", "Complete VAL, one seed, equal adaptation budget. Positive = removing the pathway improves that metric; negative = the trained control is better. Lower MAE does not imply lower RMSE or better far-range accuracy.")
                continue
            matrix = []
            for variant in variants:
                values = [lookup[name, variant, region]["MAE_change_vs_control_pct"] for name in names]
                matrix.append([np.nan if value is None else -value for value in values])
            heatmap(matrix, labels, [plots.SHORT.get(name, name) for name in names], output,
                    f"geometry_{phase}_{region}", f"Complete VAL / {region}: removing a geometry pathway",
                    "Single seed, equal adaptation budget, exact projection retained. Positive means the ablated model is better than the trained control; negative supports retaining that pathway. No seed uncertainty estimate.")
        mechanisms = [row for row in report["rows"] if row["phase"] == phase and row["scope"] == "final_val" and row["region"] == "full"]
        fields = ("sampling_candidate_oracle_MAE", "continuous_candidate_oracle_MAE", "group_mean_oracle_MAE", "fine_stride8_MAE")
        if mechanisms and all(all(row.get(key) is not None for key in fields) for row in mechanisms):
            by_key = {(row["dataset"], row["variant"]): row for row in mechanisms}
            fig, axes = plots.plt.subplots((len(names) + 3) // 4, 4, squeeze=False,
                                          figsize=(14, 3.4 * ((len(names) + 3) // 4)), constrained_layout=True)
            for axis, name in zip(axes.flat, names):
                for variant, label, color in zip(("control", *variants), ("Control", *labels), COLORS):
                    row = by_key[name, variant]
                    axis.plot(range(4), [row[key] for key in fields], "o-", color=color, ms=4, label=label)
                axis.set(title=plots.SHORT.get(name, name), xticks=range(4),
                         ylabel="MAE (m)" if row["units"] == "meters" else "MAE (scene unit)")
                axis.set_xticklabels(["Sample oracle", "Point oracle", "Group-mean oracle", "Actual fine"], rotation=35, ha="right", fontsize=8)
                axis.grid(axis="y", alpha=.2)
            for axis in list(axes.flat)[len(names):]:
                axis.axis("off")
            handles, legend = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, legend, loc="outside upper center", ncol=2, frameon=False)
            plots.save(fig, output, f"geometry_{phase}_oracle_gap", "Fixed mechanism VAL subset, same stride-8 lattice, single seed. Oracles use GT only offline. These are different hypothesis sets, not necessarily nested; oracle improvement alone is not reconstruction improvement.")
    plots.csv_write(output / "geometry_full_val.csv", selected)
    lines = ["% Requires booktabs. Single seed; no uncertainty or significance stars.",
             "\\begin{table*}[t]", "\\centering",
             f"\\caption{{Complete validation MAE for geometry-pathway ablations after matched adaptation, single seed {report['seed']}. Exact projection is retained in every variant. Units are shown per dataset; errors are not pooled across domains.}}",
             "\\label{tab:geometry-full-val}", "\\begin{tabular}{llrrrr}", "\\toprule",
             "Dataset & Unit & Control & Source embedding off & Scale routing off & Jacobian width off \\\\", "\\midrule"]
    full = [row for row in selected if row["region"] == "full"]
    for name in dict.fromkeys(row["dataset"] for row in full):
        by_variant = {row["variant"]: row for row in full if row["dataset"] == name}
        unit = "m" if by_variant["control"]["units"] == "meters" else "scene unit"
        cells = [f"{by_variant[variant]['MAE']:.4f}" for variant in ("control", *variants)]
        lines.append(" & ".join([plots.SHORT.get(name, name), unit, *cells]) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (output / "geometry_full_val.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/active/rulermvs_v6_20261101/full_training")
    args = parser.parse_args()
    root = plots.path(args.root)
    output = root / "paper_analysis/retraining"
    output.mkdir(parents=True, exist_ok=True)
    sources = {"handoff": root / "paper_handoff_ablation/summary.json",
               "geometry": root / "paper_geometry_ablation/analysis/summary.json"}
    evidence = {"completed": [], "missing": [], "source_sha256": {str(Path(__file__)): plots.sha(__file__)}}
    for name, source in sources.items():
        if not source.exists():
            evidence["missing"].append(name)
            continue
        {"handoff": handoff, "geometry": geometry}[name](plots.read(source), output)
        evidence["completed"].append(name)
        evidence["source_sha256"][str(source)] = plots.sha(source)
    plots.write_json(output / "audit.json", evidence)
    print(evidence)


if __name__ == "__main__":
    main()
