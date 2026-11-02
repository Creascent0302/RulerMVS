"""仅用实际完整测试与独占计时结果生成中文报告及CSV。"""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jar.pretraining.comparison import add_statistics, write_json
from research.near_mid_evaluation_20261028.evaluate import finish
from research.near_mid_evaluation_20261028.metrics import depth_metrics
from research.near_mid_evaluation_20261028.report import table, number

OUT = ROOT / 'runs/active/v2_evaluation_20261029'
NAMES = {'v2_public': 'V2公共导出＝原推荐', 'v2_public_equal': 'V2公共·同像素信息',
         'v2_carla_new': 'V2新CARLA微调', 'v2_carla_new_equal': 'V2新CARLA·同像素信息',
         'mvsanywhere_refined': 'MVSAnywhere官方细化1次'}


def read(path):
    return json.loads(path.read_text())


def save_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def grouped_frames(folder, domain):
    groups = {}
    for part in sorted((folder/'full_test'/domain).glob('part_*.json')):
        for frame in read(part)['frames']:
            for name, record in frame['methods'].items():
                for group in ('all', str(frame['valid_views'])+'views',
                              'complete' if frame['valid_views']==4 else 'partial'):
                    dest = groups.setdefault(group, {}).setdefault(name, {'frames':0, 'sums':{}})
                    dest['frames'] += 1
                    add_statistics(dest['sums'], record['depth']['full'])
    return {g:{n:dict(frames=v['frames'], **depth_metrics(v['sums'])) for n,v in rows.items()}
            for g,rows in groups.items()}


def merge_references(configs):
    totals, frame_ids = {}, set()
    for group in ('carla','carla_other_refs'):
        config = configs[group]
        for domain in config['datasets']:
            rows = [json.loads(l) for l in Path(domain['manifests']['test']).read_text().splitlines()]
            frame_ids.update((r['sequence'], r['frame_id']) for r in rows)
            for part in sorted((OUT/group/'full_test'/domain['name']).glob('part_*.json')):
                for name, values in read(part)['totals'].items():
                    if name.endswith('_equal'):
                        continue
                    dest = totals.setdefault(name, {'indices':[], 'depth':{}, 'surface':{}, 'stages':{},
                                                   'thresholds':values['thresholds']})
                    dest['indices'] += [(domain['reference_index'],i) for i in values['indices']]
                    for kind in ('depth','stages'):
                        for region, sums in values[kind].items():
                            add_statistics(dest[kind].setdefault(region,{}), sums)
                    add_statistics(dest['surface'], values['surface'])
    return {'unique_frames':len(frame_ids), 'methods':{n:finish(v) for n,v in totals.items()}}


def main():
    groups = ('public','carla','carla_other_refs')
    configs = {g:read(Path(__file__).parent/(g+'.json')) for g in groups}
    tested = {g:read(OUT/g/'full_test/summary.json') for g in groups}
    speeds = {g:read(OUT/g/'latency/latency.json') for g in ('public','carla')}
    extra, audit = read(OUT/'extra_checks.json'), read(OUT/'training_audit.json')
    assert all(r['completed'] and r['split']=='test' for r in tested.values())
    assert all(r['completed'] for r in speeds.values()) and extra['completed']
    views = grouped_frames(OUT/'carla', 'CARLA_new_upper')
    multi = merge_references(configs)
    write_json(OUT/'carla_by_views.json',views)
    write_json(OUT/'carla_all_references.json',multi)
    depth_rows, stage_rows, surface_rows, joint_rows = [], [], [], []
    lines = ['# V2新训练后的完整评测与协议检查（2026-10-29）','',
        '所有时间按北京时间理解。本轮只测已固定的导出权重，没有根据TEST选检查点、调整范围或继续训练。大结果均在数据盘软链接 `runs/active/v2_evaluation_20261029/`。','',
        '## 训练结果先核验，再测模型','',
        '公共训练完成4轮、6532次更新、209024组曝光，2026-10-28 23:02结束；两个阶段均没有通过完整VAL准入，第三阶段按计划跳过。导出的所有state_dict张量与原推荐逐位一致，文件SHA不同来自配置/元数据。故本报告的“公共导出”也是原推荐的复测，不能写成新训练获得提升。公共last.pt在阶段结束也已回退，未保留失败阶段末参数，无法事后报告那些参数的TEST；只能如实引用其VAL日志。','',
        '新CARLA适配完成12轮、192次更新、6144组曝光，2026-10-28 23:07结束；选择第二阶段第1轮（累计第128次更新），不是最后一轮。完整111VAL的域内三项相对风险约改善46.06%，这是VAL综合变化，不是TEST MAE下降46.06%。新权重141个state_dict张量与公共起点不同，导出与incumbent逐位一致。','',
        '## 现有测试流程的检查与修正','',
        '- 保留已验证的深度公式和单参考可见表面公式；RMSE先对所有有效像素平方误差求平均再开方，不能平均每帧RMSE。CPU共8项指标/接口/缓存参数构造测试通过。',
        '- 原逐阶段评测直接调用自研模型，没有处理equal_pixels配置。本轮修复后，同像素分支确实只用256×448主图上采样得到匹配图。历史默认native结果不受这项分支修复影响。',
        '- 原评测清单指向旧CARLA，已在本轮全部替换。只用captured_data3作为新CARLA TEST，训练为0+2、验证为1；按轨迹隔离，缺少世界位姿元数据，尚不能证明地图空间完全不重合。',
        '- 七公共域保持原固定TEST，共11970组；新CARLA固定上参考181帧（四目完整174、缺相机7）。左右下另按真实参考RGB与同相机GT可用性建立清单，不复制缺失图像、不使用tele或历史帧。',
        '- 主图256×448，自研原生匹配512×896。主表是原生系统比较；同像素信息对照不是相同计算量，也不是同像素重新训练。',
        '- MVSAnywhere使用官方完整Hero权重、细化1次、向量化官方代价体。两方使用共同TRAIN固定搜索范围；不从TEST标签确定范围，不把GT或监督标尺点放入推理输入。',
        '- 只在GT原有有效区评价；公共域不按候选范围删除GT。CARLA保持训练前指定2–80m全图，无额外rig ROI。预测超范围仍计误差；缺预测报告coverage，δ分母为全部有效GT。',
        '- 新CARLA微调V2与未做新CARLA微调的官方baseline训练经历不同。公共V2迁移对照单列；本轮没有新增baseline微调，不能声称同训练预算的架构胜利。公共baseline的原预训练也可能包含本项目测试路线，历史TEST已用于诊断，均不是严格零样本盲测。',
        '- 全部精度进程退出后独占GPU0交替测速，FP32/TF32off、位置缓存一致，10次预热、3个固定无标签VAL样本各40次，另逐方法独占测显存。时延分布是固定样本重复调用，不是全TEST道路延迟分布。','',
        '计时另修复了一个具体问题：在inference_mode内首次创建模型，会让位置参数没有版本计数，缓存安全逻辑因此每次回退重算。修正后提前按实际有效视图数在普通张量上下文加载参数，再执行推理；真实完整/缺相机VAL样本均确认缓存已填充，开关缓存预测逐位一致，无标签加载的输入和输出也逐位一致。首次补充四参考V2约134.95ms、baseline176.25ms属于缓存失效状态，只保留在latency_initial/extra_checks_initial等审计文件，不用于下方最终比较。正式主测速和独占显存均已重测。','',
        '## 原生系统：精度与速度同表','',
        'MAE/RMSE单位由每个数据集注明；AbsRel无量纲，计算为平均 |预测−真值|/真值。AbsRel=0.1表示平均相对误差10%。RMSE对少量大误差更敏感，因此MAE下降而RMSE上升完全可能。米与scene_unit不能合并成一个平均米制MAE。','']
    carla_result = tested['carla']['datasets']['CARLA_new_upper']['methods']
    fresh = carla_result['v2_carla_new']['depth']
    start = carla_result['v2_public']['depth']['full']
    baseline = carla_result['mvsanywhere_refined']['depth']
    near_gap = (fresh['near']['MAE'] - baseline['near']['MAE']) * fresh['near']['gt_pixels'] / fresh['full']['gt_pixels']
    total_gap = fresh['full']['MAE'] - baseline['full']['MAE']
    findings = ['## 结果解读与下一轮评测建议','',
        f"新CARLA上参考MAE {start['MAE']:.4f}→{fresh['full']['MAE']:.4f}m，降低{100*(1-fresh['full']['MAE']/start['MAE']):.2f}%；RMSE降低{100*(1-fresh['full']['RMSE']/start['RMSE']):.2f}%，AbsRel降低{100*(1-fresh['full']['AbsRel']/start['AbsRel']):.2f}%。但baseline MAE为{baseline['full']['MAE']:.4f}m，新模型仍未追平。",'',
        f"相对baseline的总MAE差为{total_gap:.4f}m，其中近距贡献{near_gap:.4f}m（{100*near_gap/total_gap:.2f}%）。这是按共同GT像素数分解出的事实，不是证明某个模块单独导致了误差。新模型共同stride8的粗/细/最终MAE为3.5899/3.5006/3.3407m，后续读取仍有改善；不能把本次全部差距归给最终上采样。",'',
        '新CARLA左右参考也有改善，下参考MAE只小幅改善且RMSE反而升高（12.5595→13.5630m）。轮换参考训练不自动保证所有相机都受益。当前以固定上相机VAL选权重符合单上输出协议；若最终目标是四参考输出，下一轮应提前将四参考VAL纳入选择和守门。本轮保持已经选定的权重，不依据这些TEST差异重选。','',
        '公共原生结果与20261028同协议的原推荐/官方基线MAE、RMSE、AbsRel逐项完全相同。复测支持当前指标和输入口径稳定，不能由此说新公共训练成功。仍为4/7域MAE低于baseline，各指标不一致；公共与CARLA泛化用途继续分开。','',
        '同像素信息对照中MatrixCity的自研MAE与RMSE反而降低，表明更高匹配分辨率并非在每个域都稳定获益。这个结果只说明当前权重对输入分辨率/采样存在依赖，尚未隔离混叠、特征分布和候选路由的因果影响；不根据TEST修改默认输入或重新选权重。','']
    position = lines.index('## 原生系统：精度与速度同表')
    lines[position:position] = findings
    for group in groups:
        for domain, result in tested[group]['datasets'].items():
            for name, item in result['methods'].items():
                common={'group':group,'dataset':domain,'method':name,'label':NAMES[name], 'units':result['units']}
                for region, v in item['depth'].items():depth_rows.append(dict(common,region=region,frames=item['frames'],**v))
                for stage,v in item['stages_stride8'].items():stage_rows.append(dict(common,stage=stage,**v))
                surface_rows.append(dict(common,**item['surface']))
                if group in speeds:
                    speed=speeds[group]['datasets'][domain]
                    latency=speed['summary'][name]['wall_network_ms']
                    resources=speed['isolated_resources'][name]
                    joint_rows.append(dict(common,frames=item['frames'],**item['depth']['full'],
                        network_ms=latency['median_ms'],network_p90_ms=latency['p90_ms'],
                        single_reference_hz=latency['single_reference_hz'],**resources))
            if group not in speeds:continue
            display=[]
            for name,item in result['methods'].items():
                if name.endswith('_equal'):continue
                m=item['depth']['full']; s=speeds[group]['datasets'][domain]
                t=s['summary'][name]['wall_network_ms']; r=s['isolated_resources'][name]
                display.append([NAMES[name],item['frames'],number(m['MAE']),number(m['RMSE']),number(m['AbsRel']),
                                number(t['median_ms'],2),number(t['single_reference_hz'],1),
                                number(r['peak_allocated_gib'],3),number(m['coverage']*100,2)+'%'])
            lines += [f"### {domain}（{result['units']}）",'',table(
                ['方法','TEST参考组数','MAE↓','RMSE↓','AbsRel↓','网络ms↓','单参考Hz↑','峰值GiB↓','覆盖率'],display),'']
    lines += ['## 相同图像信息的补充对照','',
              '只改变自研输入信息，不改权重：匹配图由共同主图上采样，去掉原生高分辨率的额外像素信息。该退化输入实验能衡量依赖性，不能替代双方同预算训练。','']
    rows=[]
    for group in ('public','carla'):
        for domain,result in tested[group]['datasets'].items():
            for name,item in result['methods'].items():
                if name.endswith('_equal') or name=='mvsanywhere_refined':
                    m=item['depth']['full']; t=speeds[group]['datasets'][domain]['summary'][name]['wall_network_ms']
                    rows.append([domain,NAMES[name],number(m['MAE']),number(m['RMSE']),number(m['AbsRel']),number(t['median_ms'],2)])
    lines += [table(['数据集','方法','MAE↓','RMSE↓','AbsRel↓','网络ms↓'],rows),'',
              '## 新CARLA完整与缺相机分开','',
              '固定上参考：2目仅1帧、3目6帧、4目174帧。缺相机组太小，不能把组间差异解释成可靠的缺相机因果效应；场景难度也不同。','']
    rows=[]
    for group in ('complete','partial','2views','3views'):
        for name,m in views[group].items():
            if not name.endswith('_equal'):
                rows.append([group,NAMES[name],m['frames'],number(m['MAE']),number(m['RMSE']),number(m['AbsRel']),number(m['coverage'])])
    lines += [table(['输入组','方法','帧数','MAE(m)↓','RMSE(m)↓','AbsRel↓','覆盖率'],rows),'',
              '## 新CARLA多参考检查','',
              f"上/左/右/下参考分别181/184/183/183组，合计731个相机参考，来自{multi['unique_frames']}个独立帧ID。以下按各参考所有有效GT像素汇总；同一帧多参考彼此相关，不能当作731个独立场景。它是四个可见深度结果的评价，没有融合成全场景。",'']
    rows=[]
    for n,m in multi['methods'].items():
        v=m['depth']['full'];rows.append([NAMES[n],m['frames'],number(v['MAE']),number(v['RMSE']),number(v['AbsRel']),number(v['coverage'])])
    lines += [table(['方法','参考组数','MAE(m)↓','RMSE(m)↓','AbsRel↓','覆盖率'],rows),'']
    for domain,result in tested['carla_other_refs']['datasets'].items():
        rows=[[NAMES[n],v['frames'],number(v['depth']['full']['MAE']),number(v['depth']['full']['RMSE']),number(v['depth']['full']['AbsRel'])] for n,v in result['methods'].items()]
        lines += [f'### {domain}','',table(['方法','参考组数','MAE(m)↓','RMSE(m)↓','AbsRel↓'],rows),'']
    lines += ['## 近中远、边界与中间阶段','',
        '米制近/中/远为<10m、[10,40)m、≥40m；scene_unit用TRAIN中位深度的0.5/2倍划分。深度突变带为有效相邻GT的|ΔlogZ|>0.1后扩2像素，是边界代理，不是小障碍物实例识别率。','']
    for group in ('public','carla'):
        for domain,result in tested[group]['datasets'].items():
            rows=[]
            for n,v in result['methods'].items():
                if n.endswith('_equal'):continue
                rows.append([NAMES[n]]+[number(v['depth'][r]['MAE']) for r in ('near','mid','far','gradient_band','smooth')])
            lines += [f'### {domain}','',table(['方法','近MAE','中MAE','远MAE','边界MAE','平滑MAE'],rows),'']
    lines += ['逐阶段完整结果见 `stage_metrics.csv`：粗层原格点、细层每隔2点、最终每隔8点，对应共同原图(8x,8y)；不与全图MAE混用。MVSAnywhere仅列最终值，不把其不可比中间量称为自研粗候选。','',
              '## 单参考可见表面','',
              '由同一K将深度反投影。每帧每侧最多1024个均匀查询点，对另一侧完整点集求最近邻，保留完整参考点集以避免双侧抽稀偏差。Accuracy=预测→GT、Completeness=GT→预测平均欧氏距离，Chamfer_L1是二者平均；非平方距离。米制F-score阈值5/10/20/50cm，scene_unit阈值为TRAIN中位的0.5/1/2/5%，不称米。','']
    for group in ('public','carla'):
        for domain,result in tested[group]['datasets'].items():
            rows=[]
            example=next(iter(result['methods'].values()))['surface']
            fs=[k for k in example if k.startswith('Fscore_')]
            for n,v in result['methods'].items():
                if n.endswith('_equal'):continue
                rows.append([NAMES[n]]+[number(v['surface'][k]) for k in ('Accuracy','Completeness','Chamfer_L1',*fs)])
            lines += [f'### {domain}','',table(['方法','Accuracy↓','Completeness↓','Chamfer↓',*fs],rows),'']
    lines += ['## 单参考与四参考耗时的区别','',extra['scope'],'',extra['protocol'],'']
    rows=[]
    for n,v in extra['latency'].items():
        four=v['four_reference_network_ms'];cpu=v['cpu_ready_to_depth_cpu_ms']
        rows.append([NAMES[n],number(cpu['median_ms'],2),number(four['median_ms'],2),number(four['four_depth_update_hz'],2)])
    lines += [table(['方法','单参考CPU输入→GPU推理→深度回CPU(ms)','四参考网络(ms)','四深度更新Hz'],rows),'',
              '单参考Hz不能冒充整车四参考场景更新率；这里也未包含原始图像解码、全部后处理和融合，不代表车端10Hz已验证。独占显存包含输入、模型和激活，不包含CUDA驱动上下文；参数量与P90见accuracy_speed.csv。','',
              '## 复现与限制','',
              '- `training_audit.json`：完成状态、逐张量回退核查、选模阶段与权重SHA。',
              '- `public/carla/carla_other_refs/full_test/{summary.json,audit.json}`：完整结果与源/权重/清单hash；part文件保存逐帧统计和真实有效相机数。',
              '- `public/carla/latency/`：单卡交替原始计时与独占资源；`extra_checks.json`：无标签加载一致性、真实输出一致性与四参考计时。',
              '- `accuracy_speed.csv`、`depth_metrics.csv`、`stage_metrics.csv`、`surface_metrics.csv`包含全部δ、log、SqRel、bias和coverage；本报告主表按用户习惯先省略δ。',
              '- 没有可比的全场景占据/语义输出与统一标签，未报告IoU/mIoU；没有测车端硬件、动态物体实例或时序稳定性。',
              '- 未给baseline做新CARLA微调；本次是现有固定模型的实际能力比较，不能称同数据同训练预算复现。未保存的不合格公共候选权重无法恢复，不能编造其TEST。',
              '', '```bash',
              'official_methods/.venv/bin/python -m unittest research.near_mid_evaluation_20261028.test_metrics research.v2_evaluation_20261029.test_protocol -v',
              'bash research/v2_evaluation_20261029/run.sh',
              'official_methods/.venv/bin/python research/v2_evaluation_20261029/extra_checks.py',
              'official_methods/.venv/bin/python research/v2_evaluation_20261029/report.py',
              '```','', '重新测速须在配置中更换output目录，入口拒绝覆盖既有计时。完整精度分片只有协议hash一致才复用。']
    for name,rows in [('depth_metrics.csv',depth_rows),('stage_metrics.csv',stage_rows),('surface_metrics.csv',surface_rows),('accuracy_speed.csv',joint_rows)]:
        save_csv(name,rows)
    (ROOT/'docs/v2_evaluation_20261029.md').write_text('\n'.join(lines)+'\n')
    print('Report and CSV written:',OUT)


if __name__ == '__main__':
    main()
