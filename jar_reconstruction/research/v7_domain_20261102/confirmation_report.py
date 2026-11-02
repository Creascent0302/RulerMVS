"""生成固定候选的VAL/TEST对照与图；不选择新权重、不启动训练。"""

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json

RUN = ROOT / 'runs/active/v7_domain_20261102'
OUTPUT = RUN / 'confirmation/analysis'
METHOD = 'synthetic_joint_keep'
METRICS = ('MAE', 'RMSE', 'AbsRel')


def read(path):
    return json.loads(Path(path).read_text())


def change(current, previous):
    return 100 * (current / previous - 1) if previous else None


def main():
    reports = {phase: read(RUN / f'confirmation/{phase}/results.json') for phase in ('eth', 'carla')}
    assert all(report['completed'] for report in reports.values())
    validations = {phase: read(RUN / f'full_validation_round2/{phase}.json') for phase in ('public', 'carla')}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    previous_root = ROOT / 'runs/active/rulermvs_v6_20261101/full_training'
    previous = {
        'eth': read(RUN / 'diagnosis/results.json'),
        'carla': read(previous_root / 'evaluation_recovery/carla_new_main/full_test/summary.json'),
    }
    for phase, report in reports.items():
        for name, dataset in report['datasets'].items():
            for model, record in dataset['models'].items():
                for stage, regions in record['metrics'].items():
                    for region, values in regions.items():
                        base = dataset['models']['frozen_v6']['metrics'][stage][region]
                        rows.append({'dataset': name, 'model': model, 'stage': stage, 'region': region,
                                     'frames': dataset['frames'], **values,
                                     **{key + '_vs_v6_pct': change(values[key], base[key])
                                        for key in METRICS}})
            # 已有正式分母和固定权重必须复现；nocc在本轮与all独立保留。
            if phase == 'eth' and name.endswith('_nocc'):
                continue
            for model, old_name in [('frozen_v6', 'v6_best'), ('mvsanywhere_refined', 'mvsanywhere_refined')]:
                actual = dataset['models'][model]['metrics']['depth']['full']
                old = (previous['eth']['datasets'][name]['metrics'][old_name]['full'] if phase == 'eth'
                       else previous['carla']['datasets'][name]['methods'][old_name]['depth']['full'])
                assert actual['gt_pixels'] == old['gt_pixels']
                assert actual['predicted_pixels'] == old['predicted_pixels']
                for key in METRICS:
                    difference = abs(actual[key] - old[key])
                    assert difference < max(1e-7, abs(old[key]) * 1e-9)
                    checks.append({'dataset': name, 'method': model, 'metric': key, 'difference': difference})
    with (OUTPUT / 'test_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    preservation = read(RUN / 'v6_preserved/preservation.json')
    changed = [name for name, value in preservation['sources'].items() if digest(ROOT / name) != value]
    changed += [name for name, value in preservation['weights'].items() if digest(name) != value]
    assert not changed, changed
    write_json(OUTPUT / 'audit.json', {'completed': True, 'old_metric_checks': checks,
        'preserved_source_count': len(preservation['sources']), 'v6_changed': changed,
        'source_sha256': {str(RUN / f'confirmation/{phase}/results.json'):
                         digest(RUN / f'confirmation/{phase}/results.json') for phase in reports},
        'scope': '固定一个短训末步，ETH和CARLA使用各自public/carla权重；已被多次查看的TEST，非新的盲测。'})
    lines = ['# ETH3D / CARLA 专项：本轮结果与取舍', '',
             '本轮没有得到满足“两个目标域提高、其他指标保持”的可替换版本。V6源码、配置与两套正式权重均按本轮开始时的SHA核对保持；所有V7试验独立存放。本报告中的候选是固定的synthetic_joint_keep短训末步，不是在TEST上挑选的最好checkpoint。', '',
             '## 1. 先回答植被是否是主因', '',
             '不能统一归因于植被。冻结模型的273次分层诊断发现：ETH森林场景V6/基线细化MAE约0.313/0.332米，V6略好；delivery_area、facade、terrace解释89.48%的净差，约99%的净差位于0.5–8像素投影敏感度区。该敏感度在平行双目时等于fB/Z，不是纹理或匹配正确性的证明。没有语义标签，场景名称不等于植被像素掩码。', '',
             'CARLA上参考的边界遮挡、左右参考没有其他相机覆盖的区域更突出；有源GT一致区域V6反而较好。现模型也没有将全部场景强制拟合成平面。因此本轮没有增加笼统的“非平面/植被分支”。', '',
             '诊断证据：[完整分层报告](../runs/active/v7_domain_20261102/diagnosis/诊断结论.md)。', '',
             '## 2. 实际实现和测试了什么', '',
             '- 投影坐标内的连续族中心：精确重投影、零初始化129参数残差；保持候选预算。小收益不足抵消约1.6–1.8毫秒开销，未采用为默认。',
             '- 缺证据时补充粗候选图像先验：复用四层128维特征，512→64点卷积、两种感受野深度卷积，再输出候选修正。固定96槽原型40416参数；连续log深度编码版本37760参数。不是复制MVSAnywhere完整通路。',
             '- 连续版用图像32维系数与候选深度共享编码点积。门读取真实执行的CandidateViewAttention输出Σw|g|²与null质量；只描述证据条件，不认证可见性，不读取标签。',
             '- TRAIN短基线再生成：用真实训练深度反投影、改变相机位置、双线性投放和z-buffer筛选，未知孔洞明确屏蔽。最初最近像素投放会产生半像素量化，已淘汰为正式训练方式。合成不增加推理计算，也不能生成隐藏表面或新道路。',
             '- 训练比较包括同预算旧头续训、冻结旧头只训新增模块、联合训练质量保持、灰度/真实源子集增强、合成与无合成。完整日志、权重和样本曝光核对在数据盘。', '',
             '方法依据与实际改动边界见[文献调研](../research/v7_domain_20261102/literature.md)、[数据再生成说明](../research/v7_domain_20261102/SYNTHETIC_PAIR_BILINEAR.md)。先验、宽搜索与定位分工、几何重投影增强都有研究基础，本轮不宣称首次提出。', '',
             '## 3. 完整VAL与固定TEST必须分开', '',
             '两轮完整VAL各7957个公共参考、411个CARLA参考，均七模型，共117152次模型预测；相同冻结编码复用经1078次完整forward逐位核对，旧V6指标与旧报告一致。这是离线评测加速，不是部署速度。', '',
             '以下统一256×448、真实相机数、完整输出深度、相同有效GT分母；ETH两目，CARLA四目，每次输出一个参考。MAE/RMSE单位为米；AbsRel无量纲。', '',
             '|固定TEST|方法|MAE↓|RMSE↓|AbsRel↓|',
             '|---|---|---:|---:|---:|']
    labels = {'frozen_v6': 'V6', METHOD: '本轮固定候选', 'mvsanywhere_refined': 'MVSAnywhere细化'}
    for report in reports.values():
        for name, dataset in report['datasets'].items():
            for model, record in dataset['models'].items():
                values = record['metrics']['depth']['full']
                lines.append('|' + name + '|' + labels[model] + '|' +
                             '|'.join(f'{values[key]:.6f}' for key in METRICS) + '|')
    lines += ['', 'CARLA验证集和测试集的相对变化：', '',
              '|参考|VAL MAE变化|TEST MAE变化|TEST近距MAE变化|TEST粗层MAE变化|',
              '|---|---:|---:|---:|---:|']
    for name, dataset in reports['carla']['datasets'].items():
        val = validations['carla']['models']
        v0 = val['frozen_v6']['datasets'][name]['metrics']['depth']['full']['MAE']
        v1 = val[METHOD]['datasets'][name]['metrics']['depth']['full']['MAE']
        current = dataset['models'][METHOD]['metrics']
        base = dataset['models']['frozen_v6']['metrics']
        changes = [change(v1, v0)] + [change(current[s][r]['MAE'], base[s][r]['MAE'])
                                       for s, r in [('depth', 'full'), ('depth', 'near'), ('coarse_stride8', 'full')]]
        lines.append('|' + name + '|' + '|'.join(f'{value:+.3f}%' for value in changes) + '|')
    eth = reports['eth']['datasets']['ETH3D_two_view_all']['models']
    lines += ['', 'ETH相同stride8格点的阶段变化：', '', '|阶段|V6 MAE|候选 MAE|变化|', '|---|---:|---:|---:|']
    for stage in ('coarse_stride8', 'fine_stride8', 'final_stride8'):
        a = eth['frozen_v6']['metrics'][stage]['full']['MAE']
        b = eth[METHOD]['metrics'][stage]['full']['MAE']
        lines.append(f'|{stage}|{a:.6f}|{b:.6f}|{change(b, a):+.3f}%|')
    lines += ['', 'ETH的粗层略有改善，后续没有保住；不能因末端失败否定全部前端信号。CARLA TEST的粗层已经退化，不能只归罪后端。二者必须分别立项。', '',
              '公共完整VAL中，合成联合方案改善Hypersim、Matrix、Synth、VK的全图MAE，但Blended近距MAE退9.04%、全图AbsRel退9.67%，Dynamic全图MAE退1.62%、RMSE退1.98%。CARLA VAL虽然全图三指标四参考均好，上/左/右远距RMSE仍略退。不能用平均排名掩盖用户要求保留的指标。', '',
              '[完整VAL表](../runs/active/v7_domain_20261102/full_validation_round2/analysis/完整VAL对比.md)包含所有阶段、距离和区域；[固定TEST CSV](../runs/active/v7_domain_20261102/confirmation/analysis/test_metrics.csv)保留全部结果。', '',
              '## 4. 为什么当前训练约束还不够', '',
              '144次V6 TRAIN前向和288次学生解码核查发现：粗分类网格标签与实际连续最优标签虽约29–40%不一致，但全部只差一格，CARLA相应误差代价仅占当前选择误差约3%。它不是当前主要缺陷，直接换全部标签还会破坏偏移头的区间分工。', '',
              'Dynamic已曝光TRAIN总体误差下降约2.3%，VAL却退化；软保持项本身也未逐像素满足。它比较的是增强输入上的学生与旧模型，不等于保持未扰动输入。含离散选峰的实际读出也不能依靠最终深度惩罚直接优化所有选峰变化。', '',
              '合成只改变已有场景的观察方式，没有增加新的道路或物体；CARLA训练和VAL来自不同的少量轨迹。灰度、源子集、合成分别抽签会叠加，25%合成不代表75%样本完全未变。不能通过盲目增加增强强度或单一保持权重保证泛化。', '',
              '详细依据：[TRAIN目标审计](../research/v7_domain_20261102/target_audit_findings.md)、[下一轮训练设计](../research/v7_domain_20261102/TRAINING_STRATEGY_REVIEW.md)。后者的成对清洁保持、分阶段充分训练仍是未运行方案，不是本轮已验证收益。', '',
              '## 5. 速度、保留与停止决定', '',
              '坐标编码复用只在粗候选空间广播时启用，把重复的深度MLP计算提到像素网格之外。真实N4三帧、每方法90次交替：V6/原连续先验/复用为27.894/28.228/28.145毫秒；N2为23.454/23.788/23.702毫秒。新增约0.25毫秒，不是严格零开销。该计时使用初值与两步smoke，属于研究增量测速；非最终候选整机独占正式速度，不能直接拼接为新的论文主表。', '',
              '保持V6默认与全部正式权重；V7仅作为独立研究原型。本轮未证明满足升级条件，未启动新八卡全量训练，也未把VAL最好结果当作TEST成果。独立全训接口已准备并通过CPU接口检查，但尚未实际验证该新入口DDP；文件存在不代表八卡训练已成功。', '',
              '下一轮优先验证不增加推理结构的成对清洁保持，再研究弱视差候选供给与细层交接；只有改变关键条件并有对照证据后才放大全量训练。ETH既有27对已被诊断和反复查看，不能把后续结果包装成新的盲测。', '',
              '[V6保留核验](../runs/active/v7_domain_20261102/confirmation/analysis/audit.json)；[研发入口](../research/v7_domain_20261102/README.md)。所有大产物经runs/active软链接保存在数据盘，原数据与baseline权重均未删除。']
    lines += control_note()
    lines += ['', '[VAL与TEST变化图（PDF）](../runs/active/v7_domain_20261102/confirmation/analysis/validation_test_gap.pdf)。']
    (ROOT / 'docs/v7_eth3d_carla_20261102.md').write_text('\n'.join(lines) + '\n')
    plot(reports, validations)
    print(json.dumps({'completed': True, 'report': 'docs/v7_eth3d_carla_20261102.md'}))


def control_note():
    folder = RUN / 'full_validation_control'
    if not (folder / 'completed.json').exists():
        return []
    assert read(folder / 'completed.json')['completed']
    control = read(folder / 'carla.json')['models']['control_synthetic']['datasets']
    other = read(RUN / 'full_validation_round2/carla.json')['models']
    lines = ['', '## 6. 完整减法对照：主要VAL收益不需要新增先验', '',
             '独立用原V6结构＋相同合成数据/增强/保持损失，公共896、CARLA256更新。逐条样本曝光、位姿、孔洞覆盖、抽签和初始输出均与含先验组一致。只删除新增先验，旧网络学习率与训练预算不变。', '',
             '|CARLA完整VAL|原V6|合成＋连续先验|合成＋原V6结构|', '|---|---:|---:|---:|']
    for name, record in control.items():
        values = [other[key]['datasets'][name]['metrics']['depth']['full']['MAE']
                  for key in ('frozen_v6', METHOD)]
        values.append(record['metrics']['depth']['full']['MAE'])
        lines.append('|' + name + '|' + '|'.join(f'{v:.6f}' for v in values) + '|')
    lines += ['', '单位米。删除先验后，上参考相差约0.00011米，其余三方向MAE更好，四方向RMSE均更好。先验对部分近距与AbsRel仍有小收益，不能称完全无用；当前证据不足以支持增加到正式模型。原V6结构没有新增推理操作；新权重正式速度未另计时，不虚构毫秒数。', '',
              '此对照新增7957＋411完整VAL与88次完整forward核对。三组完整VAL合计125520次预测、1166次逐位检查。26组正式短训合计14976更新、29952曝光（不含两步冒烟）。统一CPU测试实际51 passed，3.25秒。', '',
              '更简单的训练版本仍非可替换升级：Blended近距、Dynamic中距退化仍在，副作用不能只归罪先验。减法版本没有测试ETH/CARLA TEST，不能沿用含先验版本的TEST数字。完整VAL已不满足保持条件，本轮不继续拿TEST挑选它，也未启动新八卡全量训练。', '',
              '[完整减法报告](../runs/active/v7_domain_20261102/full_validation_control/analysis/完整减法对照.md)。下一轮优先改变训练约束的输入条件与数据覆盖，ETH细层交接独立处理。充分训练能否改善仍未验证，本轮失败不等于理论上不可能。']
    return lines


def plot(reports, validations):
    os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    names = list(reports['carla']['datasets'])
    changes = {'Full VAL': [], 'Fixed TEST': []}
    for name in names:
        val = validations['carla']['models']
        changes['Full VAL'].append(change(val[METHOD]['datasets'][name]['metrics']['depth']['full']['MAE'],
                                         val['frozen_v6']['datasets'][name]['metrics']['depth']['full']['MAE']))
        models = reports['carla']['datasets'][name]['models']
        changes['Fixed TEST'].append(change(models[METHOD]['metrics']['depth']['full']['MAE'],
                                           models['frozen_v6']['metrics']['depth']['full']['MAE']))
    x = np.arange(len(names))
    for offset, (label, values) in zip((-.19, .19), changes.items()):
        bars = axes[0].bar(x + offset, values, .36, label=label)
        axes[0].bar_label(bars, fmt='%+.2f', fontsize=9, padding=3)
    axes[0].set_xticks(x, ['Upper', 'Left', 'Right', 'Down'])
    axes[0].legend(loc='lower right')
    axes[0].set_title('CARLA: validation gain did not generalize')
    models = reports['eth']['datasets']['ETH3D_two_view_all']['models']
    stages = ('coarse_stride8', 'fine_stride8', 'final_stride8')
    vals = [change(models[METHOD]['metrics'][s]['full']['MAE'],
                   models['frozen_v6']['metrics'][s]['full']['MAE']) for s in stages]
    bars = axes[1].bar(np.arange(3), vals, color=['#267a98', '#c06451', '#c06451'])
    axes[1].bar_label(bars, fmt='%+.2f', padding=3)
    axes[1].set_xticks(np.arange(3), ['Coarse', 'Fine', 'Final'])
    axes[1].set_title('ETH3D: common stride-8 evaluation')
    for axis in axes:
        axis.axhline(0, color='black', linewidth=.8)
        axis.set_ylabel('MAE change vs V6 (%) | lower is better')
        axis.margins(y=.2)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(OUTPUT / ('validation_test_gap.' + extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
