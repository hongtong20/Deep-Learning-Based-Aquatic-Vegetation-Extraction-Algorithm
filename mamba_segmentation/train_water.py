"""
水草 4 类 · MambaVision-UPerNet (mmseg) 训练入口

目的:
  把所有需要改的路径集中到本文件顶部, 训练时只改这里;
  configs/water/ 只保留模型结构/调度/数据管线 (不含机器相关路径)。

用法 (装有 mmseg/mamba_ssm 的服务器, 在 semantic_segmentation/ 目录下):
    python train_water.py                          # 用下方默认路径
    python train_water.py --variant B --work-dir work_dirs/mv_B_run1
    python train_water.py --resume work_dirs/mv_T_run1/iter_8940.pth
    python train_water.py --resume                 # 恢复 work_dir 中最新 checkpoint

对齐点与差异声明: 见仓库根 WATER_ADAPTATION.md
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)          # 使 water_plugins 可被 custom_imports 导入

# =============================================================================
# ★★★ 路径配置 (只改这里, 不要改 configs/) ★★★
# =============================================================================
VARIANT = 'T'                                       # 'T' | 'S' | 'B' (MambaVision 变体)
WORK_DIR = 'work_dirs/mv_T_run1'                    # 输出目录 (权重/日志)

# 数据集根目录 (其下为 train/val/test × image/label)
DATA_ROOT = 'E:/水草样本512修改二次/dataset_8band'

# 6 波段全局统计量: 与 ours / SMP 对照共用同一份 npz, 禁止重算!
# (从 base1 的 checkpoint/global_stats_enc1_rgbnirsar.npz 拷贝到服务器后改此路径)
STATS_FILE = r'C:\Users\hongtong\PyCharmMiscProject\smp-trans-net sub base 1\checkpoint\global_stats_enc1_rgbnirsar.npz'

# ImageNet-1K 预训练权重 (下载地址见 WATER_ADAPTATION.md, T/S/B 各一份)
IMAGENET_CKPT = 'ckpts/mambavision_tiny_1k.pth.tar'  # 换变体时记得同步改; None=随机初始化
# =============================================================================

VARIANT_CONFIGS = {
    'T': 'configs/water/upernet_mamba_vision-T_water_6ch_512.py',
    'S': 'configs/water/upernet_mamba_vision-S_water_6ch_512.py',
    'B': 'configs/water/upernet_mamba_vision-B_water_6ch_512.py',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='水草对比实验训练入口: MambaVision-UPerNet (mmseg), '
                    '路径默认取本文件顶部常量, 可用命令行临时覆盖')
    parser.add_argument('--variant', choices=sorted(VARIANT_CONFIGS.keys()),
                        default=VARIANT, help='MambaVision 变体 (默认取顶部常量)')
    parser.add_argument('--config', default=None,
                        help='直接指定 config 路径 (覆盖 --variant)')
    parser.add_argument('--work-dir', default=WORK_DIR)
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--stats-file', default=STATS_FILE)
    parser.add_argument('--imagenet-ckpt', default=IMAGENET_CKPT,
                        help='ImageNet-1K 权重路径; 传 none 表示随机初始化')
    parser.add_argument('--resume', nargs='?', const='latest', default=None,
                        help='恢复训练: 传 checkpoint 路径, 或不带参数 (=最新)')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_path = args.config or VARIANT_CONFIGS[args.variant]
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(REPO_ROOT, cfg_path)

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(cfg_path)

    # ---- 路径覆盖: 数据集根目录 + 管线内统计量 (train/val/test 全部一致) ----
    for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        dataset_cfg = cfg[key]['dataset']
        if 'data_root' in dataset_cfg:
            dataset_cfg['data_root'] = args.data_root
        for t in dataset_cfg.get('pipeline', []):
            if t['type'] == 'NormalizeWaterZScore':
                t['stats_file'] = args.stats_file

    # ---- ImageNet 预训练权重 ----
    cfg.model.backbone.pretrained = (
        None if (args.imagenet_ckpt is None or args.imagenet_ckpt == 'none')
        else args.imagenet_ckpt)

    work_dir = args.work_dir
    if not os.path.isabs(work_dir):
        work_dir = os.path.join(REPO_ROOT, work_dir)
    cfg.work_dir = work_dir

    if args.resume is not None:
        cfg.resume = True
        if args.resume != 'latest':
            cfg.load_from = args.resume

    print('=' * 70)
    print(f'水草对比实验 · MambaVision-{args.variant} + UPerNet (mmseg)')
    print(f'  config:     {cfg_path}')
    print(f'  work_dir:   {work_dir}')
    print(f'  data_root:  {args.data_root}')
    print(f'  stats_file: {args.stats_file}')
    print(f'  imagenet:   {cfg.model.backbone.pretrained}')
    print(f'  iters:      {cfg.train_cfg["max_iters"]} '
          f'(val every {cfg.train_cfg["val_interval"]}, bs={cfg.train_dataloader["batch_size"]})')
    print(f'  bands:      {cfg.train_dataloader.dataset.pipeline[0]["band_indices"]}')
    print('=' * 70)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
