#!/bin/bash
#SBATCH --job-name=mv-water
#SBATCH --partition=debug_shared
#SBATCH --gres=mps:25
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/checkpoints/gyy/mamba/slurm_logs/single-%j.out   # SLURM 终端输出日志
#SBATCH --error=/data/checkpoints/gyy/mamba/slurm_logs/single-%j.err    # SLURM 报错日志

# MambaVision-UPerNet (mmseg) 水草 4 类对照实验训练
# 用法:
#   sbatch slurm_mamba.sh                      # 全新训练 (自动递增 mv_T_run_N)
#   改 RESUME="latest" 后再 sbatch                    # 续训最新的 run (24h 被杀后接着训)
# 换变体: 改下方 VARIANT 并同步 IMAGENET_CKPT (T/S/B 各一份)

# ---------------------------------------------------------
# 1. 路径配置 (只改这里) ★★★
# ---------------------------------------------------------
VARIANT="T"                                # 'T' | 'S' | 'B' (MambaVision 变体)
RESUME=""                                  # "" = 全新训练; "latest" = 续训最新 run

PROJECT_DIR="/data/users/gyy/project/semantic_segmentation"   # 本目录在集群上的位置
DATA_ROOT="/data/users/gyy/dataset_8band"                     # 数据集根 (其下 train/val/test × image/label)
STATS_FILE="/data/users/gyy/project/semantic_segmentation/global_stats_enc1_rgbnirsar.npz"  # 从 base1 拷贝的同一份 npz, 禁止重算!
IMAGENET_CKPT="ckpts/mambavision_tiny_1k.pth.tar"             # 相对 PROJECT_DIR; T=tiny S=small B=base
ENV_DIR="/data/users/gyy/project/env_cuda129"                 # conda 环境前缀 (自定义位置), 直接用其中 python, 无需 activate

# ---------------------------------------------------------
# 2. 基础目录 (权重与 SLURM 日志统一放一起: run_* 目录与 single-*.out/err 同级)
# ---------------------------------------------------------
USER_NAME="gyy"
BASE_DIR="/data/checkpoints/${USER_NAME}/mamba/slurm_logs"
mkdir -p ${BASE_DIR}

# ---------------------------------------------------------
# 3. 计算本次 run 目录:
#    - 新训练: 自动递增 mv_${VARIANT}_run_N
#    - 续训:   复用最新的 run 目录 (train_water.py --resume 自动找其中最新 checkpoint)
# ---------------------------------------------------------
RUN_PREFIX="mv_${VARIANT}_run"

if [ "${RESUME}" = "latest" ]; then
    CURRENT_SAVE_DIR=$(ls -1d ${BASE_DIR}/${RUN_PREFIX}_* 2>/dev/null \
        | awk -F'_' '{print $NF}' | sort -n | tail -1)
    if [ -z "${CURRENT_SAVE_DIR}" ]; then
        echo "[错误] RESUME=latest 但 ${BASE_DIR} 下没有 ${RUN_PREFIX}_* 目录, 请先全新训练"
        exit 1
    fi
    CURRENT_SAVE_DIR="${BASE_DIR}/${RUN_PREFIX}_${CURRENT_SAVE_DIR}"
    RESUME_ARGS="--resume"
else
    MAX_RUN=$(ls -1d ${BASE_DIR}/${RUN_PREFIX}_* 2>/dev/null \
        | awk -F'_' '{print $NF}' | sort -n | tail -1)
    if [ -z "${MAX_RUN}" ]; then
        NEXT_RUN=1
    else
        NEXT_RUN=$((MAX_RUN + 1))
    fi
    CURRENT_SAVE_DIR="${BASE_DIR}/${RUN_PREFIX}_${NEXT_RUN}"
    RESUME_ARGS=""
fi
mkdir -p ${CURRENT_SAVE_DIR}

echo "========================================================="
echo "任务启动时间: $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "分配的节点: ${SLURM_NODELIST}"
echo "变体: MambaVision-${VARIANT} + UPerNet (mmseg)"
echo "本次实验的专属输出目录: ${CURRENT_SAVE_DIR}"
echo "续训模式: ${RESUME:-否(全新训练)}"
echo "========================================================="

# ---------------------------------------------------------
# 4. 运行环境 (conda 环境在自定义前缀, 直接用其 python, 不依赖 conda activate)
# ---------------------------------------------------------
PYTHON_BIN="${ENV_DIR}/bin/python"
export PATH="${ENV_DIR}/bin:${PATH}"

# 系统 libstdc++ 过旧 (缺 CXXABI_1.3.15), 优先加载环境自带的新版 (mamba CUDA 扩展需要)
if [ -f "${ENV_DIR}/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="${ENV_DIR}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

cd ${PROJECT_DIR}

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[错误] 未找到 ${PYTHON_BIN}, 请检查顶部 ENV_DIR 是否正确"
    exit 1
fi
echo "使用 python: ${PYTHON_BIN}"

# 环境自检: 导入 water_plugins = 真实训练链路的最小冒烟测试 (mmseg 注册 + timm/transformers
# 兼容 shim + mamba_ssm CUDA 扩展 + rasterio), 缺任何一环立即退出, 不浪费排队时间
if ! ${PYTHON_BIN} -c "import water_plugins" 2>/dev/null; then
    echo "[错误] 环境 ${ENV_DIR} 缺依赖或 mamba 扩展加载失败 (下面 pip 均指 ${PYTHON_BIN} -m pip):"
    echo "   (1) mm 系全家 --no-deps, 绝不触碰 torch/numpy/opencv/timm 现有版本:"
    echo "       pip install --no-deps mmengine==0.10.1 mmcv-lite==2.1.0 mmsegmentation==1.2.2 mmdet==3.3.0"
    echo "   (2) 补依赖 (已有的自动跳过, numpy 钉死防升级):"
    echo "       pip install addict yapf rich termcolor regex matplotlib prettytable terminaltables"
    echo "       pycocotools shapely six scipy einops pyyaml pillow requests numpy==1.26.4 rasterio"
    echo "   (3) mamba CUDA 算子 (需 nvcc 12.9; torch 夜版必须 --no-build-isolation):"
    echo "       pip install --no-build-isolation causal-conv1d mamba-ssm"
    echo "   (4) 若报 libstdc++ CXXABI_1.3.15 缺失: 本脚本已自动 LD_PRELOAD 环境自带新版"
    exit 1
fi

# 训练入口 train_water.py: 路径全部由命令行传入, 不需要改 configs/
${PYTHON_BIN} train_water.py \
    --variant ${VARIANT} \
    --work-dir ${CURRENT_SAVE_DIR} \
    --data-root ${DATA_ROOT} \
    --stats-file ${STATS_FILE} \
    --imagenet-ckpt ${IMAGENET_CKPT} \
    ${RESUME_ARGS}

echo "任务运行结束时间: $(date)"
