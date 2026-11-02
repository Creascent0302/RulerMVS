"""由实际JSON生成中文结果摘要；不补造尚未完成的baseline结果。"""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/active/v5_comparison_20261101"
NAMES = {"v5_best": "V5（VAL-best）", "original_v2": "V2（20261029）",
         "mvsanywhere_refined": "MVSAnywhere 一次细化",
         "patchmatchnet_dtu": "PatchmatchNet 官方DTU", "fast_mvsnet_dtu": "Fast-MVSNet 官方DTU"}


def read(path):
    return json.loads(Path(path).read_text())


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    original = ROOT / "runs/active/rulermvs_v5_20261031/evaluation"
    summary = read(original / "summary.json")
    audit = read(OUT / "v5_audit.json")
    native_speed = read(OUT / "four_reference_native.json")["methods"]
    speed = read(OUT / "four_reference_equal_pixels.json")["methods"]
    common = read(OUT / "carla_complete_equal_pixels/full_test/summary.json")["datasets"]["CARLA_new_upper"]
    text = ["# V5正式测试与独立baseline训练验收（2026-11-01）", "",
            "生成时间：" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S 北京时间"), "",
            "V5训练及原先排队的六组完整TEST已结束，退出码0；本次核验134个源码/权重/清单文件，并重新补测真实四参考速度、174帧完整四目共同子集和PatchmatchNet官方权重迁移。测试未用于选权重。", "",
            "V5相对V2改善了CARLA上相机，但尚未全面超过MVSAnywhere。它对近距和平滑区域的改善更明显，中距与遮挡边界仍是短板；下相机在共同像素模式下还不及V2。公共数据不能跨单位计算一个总米制MAE。", "",
            "## 共同四目CARLA测试：174帧，256×448共同RGB信息", "",
            "只输出上相机深度，范围2–80米；所有方法GT掩码与帧完全相同。V2/V5匹配分支仍运行512×896张量，但这里只能从共同256×448图像上采样，不能读取额外高分辨率信息。速度单列计时来自固定完整VAL首/中/末三帧；不使用精度评测中多GPU并行的耗时。", ""]
    rows = []
    for method, result in common["methods"].items():
        full = result["depth"]["full"]
        latency = speed.get(method + "_single")
        rows.append([NAMES[method], f"{full['MAE']:.5f}", f"{full['RMSE']:.5f}", f"{full['AbsRel']:.5f}",
                     f"{100 * full['coverage']:.5f}%", f"{result['surface']['Fscore_0.1']:.5f}",
                     f"{latency['median_ms']:.2f} ms" if latency else "未独占补测"])
    text += [table(["方法", "MAE/m ↓", "RMSE/m ↓", "AbsRel ↓", "有效预测覆盖率", "F-score@0.1m ↑", "单参考耗时"], rows), "",
             "PatchmatchNet存在少量无效预测，表中MAE/RMSE是有效预测条件误差，不能写成100%GT全覆盖误差；无效部分计入覆盖率和阈值准确率失败。DTU发布权重没有本轮CARLA适配，不能把此表解释为各架构在同样充分训练后的上限。MVSAnywhere完整任务预训练与V5训练经历也不同。", "",
             "## 完整181帧CARLA上相机，与历史结果严格对应", ""]
    for mode in ("native", "equal_pixels"):
        methods = summary["groups"]["carla_" + mode]["datasets"]["CARLA_new_upper"]["methods"]
        rows = []
        for name in ("original_v2", "v5_best", "mvsanywhere_refined"):
            r = methods[name]
            rows.append([NAMES[name], *[f"{r['depth'][region]['MAE']:.5f}" for region in ("full", "near", "mid", "far")],
                         f"{r['depth']['full']['RMSE']:.5f}", f"{r['surface']['Fscore_0.1']:.5f}"])
        text += [f"**{mode}**：" + ("自研匹配RGB512×896，baseline RGB256×448；不属于相同信息量。" if mode == "native" else "相同RGB信息，保留已训练网络的张量尺寸。"), "",
                 table(["方法", "全范围MAE/m", "2–10m MAE", "10–40m MAE", "40–80m MAE", "全范围RMSE/m", "F-score@0.1m"], rows), ""]
    text += ["## 初始候选与后续解码", "",
             "下面全部在共同stride8格点计算，不能与全分辨率最终MAE混用；coarse是粗分支输出，不是GT oracle。", ""]
    rows = []
    for name in ("original_v2", "v5_best", "mvsanywhere_refined"):
        r = summary["groups"]["carla_native"]["datasets"]["CARLA_new_upper"]["methods"][name]
        rows.append([NAMES[name], *[f"{r['stages_stride8'][stage]['MAE']:.5f}" if stage in r["stages_stride8"] else "不定义为同一中间阶段"
                                  for stage in ("coarse", "fine", "final")]])
    text += [table(["方法", "coarse MAE/m", "fine MAE/m", "final stride8 MAE/m"], rows), "",
             "V5粗层2.97574→2.33562米、最终同网格2.71908→2.15798米，新增连接的改善确实传到了输出。但V5细层2.42055米仍差于粗层2.33562米，说明后续候选判断还有损失；这不等于细层在每个像素都无益，也不是删除细层的因果证据。V5近距改善而中距略退，不能概括为各距离全部提升。", "",
             "## 七个公共TEST", "",
             "采用VAL选出的public/best.pt；七域总11970组，VKITTI2和DynamicReplica为真实双目，其余为静态多位置四视图，并非全部同步四目。scene_unit没有认证为米。表内为MAE / RMSE，各域不相互平均。", ""]
    for mode in ("native", "equal_pixels"):
        rows = []
        for domain, result in summary["groups"]["public_" + mode]["datasets"].items():
            methods = result["methods"]
            cells = [domain, result["units"], next(iter(methods.values()))["frames"]]
            for name in ("original_v2", "v5_best", "mvsanywhere_refined"):
                full = methods[name]["depth"]["full"]
                cells.append(f"{full['MAE']:.4f} / {full['RMSE']:.4f}")
            rows.append(cells)
        text += [f"**{mode}**", "", table(["数据集", "单位", "组数", "V2", "V5", "MVSAnywhere细化"], rows), ""]
    text += ["## 新CARLA其他参考相机", "", "以下为各参考完整有效TEST，各行帧数不同，不能把单参考时间乘四当作实测四参考时间。", ""]
    rows = []
    for mode in ("native", "equal_pixels"):
        for domain, result in summary["groups"]["carla_other_refs_" + mode]["datasets"].items():
            cells = [mode, domain, next(iter(result["methods"].values()))["frames"]]
            cells += [f"{result['methods'][name]['depth']['full']['MAE']:.5f}" for name in ("original_v2", "v5_best", "mvsanywhere_refined")]
            rows.append(cells)
    text += [table(["模式", "参考相机", "帧数", "V2 MAE/m", "V5 MAE/m", "MVS细化 MAE/m"], rows), "",
             "## 实际速度与适用范围", "",
             "5090单卡、FP32、TF32关闭，同样的位置编码缓存，三帧各10次预热和40次随机顺序交替计时。输入在GPU上；计入适配与全部网络调用，不含IO、CPU图像预处理/H2D和三维融合，因此不是车端完整系统速度。四参考共享入口与逐参考入口的depth/valid已逐位验证一致。", ""]
    rows = []
    for mode, timings in (("native", native_speed), ("equal_pixels", speed)):
        for name, title in (("original_v2", "V2"), ("v5_best", "V5"), ("mvsanywhere_refined", "MVSAnywhere一次细化"), ("mvsanywhere_first", "MVSAnywhere首遍")):
            single = timings[name + "_single"]
            serial = timings[name + "_serial_four"]
            best = timings.get(name + "_shared_four", serial)
            rows.append([mode, title, f"{single['median_ms']:.2f}", f"{serial['median_ms']:.2f}",
                         f"{best['median_ms']:.2f}", f"{best['hz']:.2f}"])
    text += [table(["模式", "方法", "单参考/ms", "顺序四参考/ms", "可用四参考入口/ms", "四参考Hz"], rows), "",
             "MVSAnywhere首遍只用于速度边界参考，本报告精度主表对应一次细化，不能拿首遍耗时配细化精度。V5比一次细化快，但仍比MVS首遍慢，四参考未达到10Hz。这里是当前实现的速度比较，baseline没有本轮新加跨参考缓存；不宣称其理论最快实现。", "",
             "## 权重选择与训练经历", "",
             f"复核已完成101608次原始预测、6组TEST，{audit['files_hashed']}个文件哈希一致。公共best与final参数不同；CARLA best与final参数逐张量完全相同，文件哈希不同来自checkpoint容器，不能据此当两套模型。", "",
             "公共best在第一阶段第1轮（1670次更新）选中，后续2+6轮没有刷新VAL最佳；完整公共训练总15030次更新，但报告的public-best不是末轮模型。CARLA best在第二阶段第18轮、总384次更新选中。V5继承独立V2，不能把额外训练带来的全部变化归因于137856参数的CNN→ViT连接；本轮没有全预算V2继续训练对照。旧V2正式权重及默认选择保留。", "",
             "## 新baseline独立训练与检查", "",
             "PatchmatchNet和DiffMVS已分别在5090双卡NCCL、真实四目256×448 CARLA上通过2次优化更新、有限梯度及checkpoint保存。CPU本轮23项既有测试、4项新增验证/恢复测试和DiffMVS 6项原生图测试通过。核心网络和损失未替换。新增验证逻辑推理时只传inputs，保存并恢复训练随机状态/BN模式；best/final/last分开保存。", "",
             "本轮先导训练只使用新CARLA完整403/82/174划分、上参考与其余三源。每方法384次更新，四卡×每卡2×累积4名义全局batch32，轮末短批次及DDP补齐的实际曝光写日志。PatchmatchNet官方DTU初始化、Adam 1e-4迁移，20/28轮衰减；DiffMVS随机初始化、原生AdamW/OneCycle及梯度裁剪2。两者不是与V5等预训练预算实验，也不能据此确认方法性能上限。PatchmatchNet保留原生SmoothL1，但DTU毫米与CARLA米的物理拐点不同，已登记为训练适配差异。", "",
             "完整VAL逐像素MAE用于选优，无效预测按预定78米惩罚而非删除GT；此选择分数单独记录，不当作实际MAE。测试集不选择超参数或权重。测试会在训练完成后自动运行，同一174帧分别加载best与final。", ""]
    for method in ("patchmatchnet", "diffmvs"):
        folder = OUT / f"{method}_adapt"
        state = "已完成训练和固定TEST" if (folder / "completed.json").exists() else ("已启动，尚未完成" if (folder / "audit.json").exists() else "尚未启动")
        log = folder / "training/train.jsonl"
        detail = ""
        if log.exists() and log.read_text().strip():
            last = json.loads(log.read_text().splitlines()[-1])
            detail = f"；快照step={last['step']}，曝光={last['sample_exposures']}，rank0训练loss={last['loss']:.5f}（非精度指标）"
        text += [f"- {method}：{state}{detail}。目录`runs/active/v5_comparison_20261101/{method}_adapt`。"]
    text += ["", "独立启动（先确认所选GPU空闲，避免与当前tmux重复运行）：", "", "```bash",
             "CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash official_methods/methods/patchmatchnet/train.sh \\",
             "  --config official_methods/methods/patchmatchnet/configs/train_carla_20261101.json \\",
             "  --output runs/active/你的新实验目录", "",
             "CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash official_methods/methods/diffmvs/train.sh \\",
             "  --config official_methods/methods/diffmvs/configs/train_carla_20261101.json \\",
             "  --output runs/active/另一个新实验目录", "```", "",
             "续训追加`--checkpoint 原目录/last.pt`，保持配置和卡数；推理/评测用`bash official_methods/scripts/run.sh 方法名 infer|evaluate --config 配置 --checkpoint 权重 --output 目录`。不带`--synthetic`不会回退合成数据。", "",
             "## 证据与未完成事项", "",
             "全部大文件位于`runs/active`到数据盘的既有软链接。新训练保存前检查数据盘至少200GiB，不删除旧权重。", "",
             "- V5完整指标/分区/表面/原速度：`runs/active/rulermvs_v5_20261031/evaluation/{summary.json,depth_metrics.csv,stage_metrics.csv,surface_metrics.csv,accuracy_speed.csv}`。",
             "- 本次补测/核验：`runs/active/v5_comparison_20261101/{v5_audit.json,four_reference_native.json,four_reference_equal_pixels.json,carla_complete_equal_pixels/full_test/summary.json}`。",
             "- 新原生训练代码：各方法目录`train.sh`；公共训练控制在`official_methods/cli.py`和`training_validation.py`，网络仍在各自官方目录。",
             "- 其他基线没有在此轮重新训练；MVSFormer++缺官方任务权重、SimpleRecon固定视图MLP需要独立训练。占据模型不同任务不混入本深度排名。",
             "- 没有车端硬件测试、端到端传感/融合测速、重复多种训练种子或等总预训练数据预算试验；无统计显著性或论文原指标复现声明。", ""]
    patch_file = OUT / "patchmatchnet_public/full_test/summary.json"
    if patch_file.exists():
        patch = read(patch_file)
        rows = []
        for name, result in patch["datasets"].items():
            item = result["methods"]["patchmatchnet_dtu"]
            # 与已完成V5全量结果核对GT分母；不同输入预训练经历仍需披露。
            reference = summary["groups"]["public_equal_pixels"]["datasets"][name]["methods"]["v5_best"]
            assert item["frames"] == reference["frames"]
            assert item["depth"]["full"]["gt_pixels"] == reference["depth"]["full"]["gt_pixels"]
            full = item["depth"]["full"]
            rows.append([name, result["units"], item["frames"], f"{full['MAE']:.5f}", f"{full['RMSE']:.5f}", f"{full['coverage']:.6f}"])
        text += ["## PatchmatchNet官方权重完整公共TEST补测", "", "仅发布DTU权重迁移，不是已经在七域充分训练的baseline。", "",
                 table(["数据集", "单位", "组数", "MAE", "RMSE", "覆盖率"], rows), ""]
    else:
        text += ["PatchmatchNet官方权重七域TEST正在执行，本快照不报告未完成的公共指标。", ""]
    (ROOT / "docs/v5_evaluation_20261101.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
