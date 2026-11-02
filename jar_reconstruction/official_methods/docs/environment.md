# 独立环境、源码固定与 RTX 5090 构建说明

本项目使用两套隔离环境，不修改系统驱动或已有 Conda 环境。三种深度模型使用 `official_methods/.venv`；GaussianFormer-2 和 OctreeOcc 的已发布算子使用 `.venv_occ`。OctreeOcc 整网仍因上游代码不完整及四目协议冲突而暂停，安装算子不代表整网实现完成。

## 实际环境

服务器既有环境记录为：8 张 RTX 5090，每卡 32607 MiB，计算能力 12.0，驱动 570.133.07。两个环境均为 Python 3.10.21、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128。深度环境保留 NumPy 2.2.6、timm 0.9.12；占据环境使用 NumPy 1.26.4、timm 1.0.30，以适配 MMLab 和旧 nuScenes 依赖，不能随意合并两者。

系统默认 `nvcc` 实际为 CUDA 10.1，不能编译 5090 算子。所有扩展使用：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
```

MMCV 2.0.1 构建必须使用 `-std=c++17` 和 `-gencode=arch=compute_120,code=sm_120`。兼容修改完整保存在 [mmcv_201_torch28.patch](../patches/mmcv_201_torch28.patch)，只升级语言标准，不修改算子数学计算。MMCV 源码包 SHA-256 为 `2b9b2c23795b3f43cf2ebcaca46552d9b4ef7e903eacdcf7c08ee15195044071`。

MMLab 版本是 mmengine 0.10.7 / mmdet 3.0.0 / mmsegmentation 1.0.0 / mmdet3d 1.1.1。稀疏卷积使用原生 `spconv-cu126==2.3.8`、`cumm-cu126==0.7.11`，已在本机 torch cu128 下完成前后向，不能只根据包名推断支持情况。最远点采样来自固定 Pointcept 的 `libs/pointops`，原 GF2 安装文档没有列出这个必要依赖。

## 获取固定版本源码

在 `jar_reconstruction` 目录执行：

```bash
# 只读验证，不联网、不下载、不重置 checkout
python official_methods/scripts/fetch_upstream.py --audit \
  --report official_methods/artifacts/source_audit.json

# 在代码缺失的机器补齐源码，然后应用已记录的兼容补丁
python official_methods/scripts/fetch_upstream.py --apply-patches
```

脚本固定以下提交，不跟随移动的 main/master 分支：

| 模型或依赖 | 官方仓库 | 固定 commit |
|---|---|---|
| Fast-MVSNet | svip-lab/FastMVSNet | `ccb686dda2717613c67d8a289dfe7b2aeb60e2fd` |
| SimpleRecon | nianticlabs/simplerecon | `113b825084cc62166c6f5584d526650928127317` |
| MVSAnywhere | nianticlabs/mvsanywhere | `5bd49bba0992aa56060e1aa3bb40b2150b04726f` |
| OctreeOcc | 4DVLab/OctreeOcc | `54afdd64b210fa45a0b1e1f255e4f88739b4e5a8` |
| GaussianFormer-2 | huang-yh/GaussianFormer | `b7e22bfc04cd6360cdee74be5af7fdace102f0a3` |
| DINOv2 | facebookresearch/dinov2 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| Pointcept pointops | Pointcept/Pointcept | `1342eda30e96cbb5fadf5374ff8eb18f6de15c71` |

Raw 下载保存 `UPSTREAM_TREE.json`，每个文件核对 Git blob SHA-1；已有 Git checkout 从同一个固定 commit 的 tree 得到原文件校验值。下载只创建缺失文件；已有文件即使与上游不同也原样保留。兼容补丁及各修改文件 SHA-256 在 [patches/provenance.json](../patches/provenance.json)，未知修改会报告 `modified_preserved` 并以非零状态退出，绝不执行 reset/checkout 覆盖修改。显式 `--apply-patches` 先检查补丁完整适用性，已应用时跳过。

只选择网络、配置、训练代码及许可证，不下载展示素材、数据、作者数据路径软链接、缓存 `.pyc` 或大权重。源码获取成功不代表模型权重下载完成。格式整理另记在 [formatting.json](../patches/formatting.json)，记录排版前后 SHA-256 和原始 Git blob；排版通过 Black 的语法树等价检查。审计区分原始文件、算法补丁和已记录的格式变化，未记录修改仍会报错。

## 构建与验证命令

占据环境的构建脚本默认只显示计划：

```bash
bash official_methods/scripts/install_occupancy_env.sh --dry-run
bash official_methods/scripts/install_occupancy_env.sh --check

# 新环境：必须指定尚不存在的目录，避免覆盖已验证的 .venv_occ
METHOD_ENV="$PWD/official_methods/.venv_occ_new" \
  bash official_methods/scripts/install_occupancy_env.sh --install

# 只有确实需要重编译时运行；使用现有依赖，不重装 Python 包
bash official_methods/scripts/install_occupancy_env.sh --rebuild-extensions
```

`--install` 会创建 venv、下载 torch cu128、核对 MMCV 源码包、应用 C++17 补丁、编译完整 MMCV，再安装 [requirements-occupancy.txt](../requirements-occupancy.txt)。随后原位编译 GF2 的变形聚合、第二代 `localagg_prob`、OctreeOcc 布尔树掩码算子，并安装 Pointcept pointops。默认 Prob-256 不调用第一代 `localagg` 或备用 `localagg_prob_fast`，因此脚本不构建这两个未使用的变体，不宣称它们已验证。

源码路径下的 MMCV 与 GF2 `.so` 是适配器实际加载的扩展；当前 MMCV 使用本地显式导入路径，不依赖在 site-packages 中伪装一个 mmcv-lite。新安装脚本也为 MMCV 安装真实包元数据，便于依赖管理。深度环境安装脚本为 `scripts/install_depth_env.sh`，它复制基础环境到独立目录后安装深度依赖，不写回基础环境；已有可运行环境无需重跑。

构建后必须做计算测试，而不只是 import：

```bash
CUDA_VISIBLE_DEVICES=0 official_methods/.venv_occ/bin/python \
  official_methods/tools/probe_occupancy_cuda.py \
  --output official_methods/artifacts/occupancy_cuda.json
```

## 环境维护边界

依赖冻结文件保留为 [深度环境](../environment/depth_environment.freeze.txt) 和 [占据环境](../environment/occupancy_environment.freeze.txt)。已有必要 `.so` 和其源码保留，编译临时目录及历史大体积测试产物已清理。当前整理不重复下载大权重、不重建系统环境；全新空目录的端到端安装仍待单独验证。

工具的 import 成功不表示 GPU 计算通过。请分别运行算子、整网、checkpoint、DDP 检查；本轮实际执行范围见主项目 README。

上游许可证原样保留。Niantic 代码含非商业限制；GaussianFormer 顶层许可证为空，部分聚合代码另带研究/非商业声明并引用缺失许可证。下载和构建脚本不改变这些授权边界，也不能把这些代码标记为项目自有 MIT 代码。
