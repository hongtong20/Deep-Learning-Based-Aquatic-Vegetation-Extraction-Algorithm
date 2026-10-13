# =============================================================================
# 水草 4 类 · MambaVision-UPerNet 对比实验 · 公共基础配置
#
# 对齐目标: ours 的 train_fusion.CONFIG 与 SMP 对照 train_baseline.py
# (近似对齐 + 差异声明, 逐项声明见仓库根 WATER_ADAPTATION.md)
#
# 迭代数换算 (与 ours 等效优化步数):
#   训练样本 715 × ours 5 模式增强 = 3575 样本/等效epoch
#   bs=4 → 894 iter/等效epoch; 100 epochs → 89400 iters
#   warmup 15 epochs → 13410 iters; 验证/早停单位 = 1 等效epoch (894 iters)
#
# 该文件为公共基座, 骨干具体参数 (dim/depths/...) 由 T/S/B 子配置覆盖,
# 不要直接运行本文件。
# =============================================================================
_base_ = ['../_base_/default_runtime.py']

# ---------- 路径 (默认值; 会被训练入口 train_water.py 顶部常量覆盖) ----------
data_root = '/data/users/gyy/project/dataset_8band'
# 训练集 6 波段全局统计量: 与 ours / SMP 对照共用同一份 npz (键: means/stds), 禁止重算
stats_file = r'/data/users/gyy/project/semantic_segmentation/global_stats_enc1_rgbnirsar.npz'

dataset_type = 'WaterGeoTIFFDataset'
img_band_indices = [3, 2, 1, 4, 8, 7]   # R,G,B,NIR,VV,VH (rasterio 1-based) — 同 ours

num_classes = 4
ignore_index = 255
norm_cfg = dict(type='BN', requires_grad=True)   # 单卡训练用 BN (官方 SyncBN 需多卡)

# ---------- 数据管线 (数值与 FusionDataset(mode='val') 等价, 见 water_plugins) ----------
train_pipeline = [
    dict(type='LoadWaterMultiBand', band_indices=img_band_indices),
    dict(type='LoadWaterAnnotations'),
    # 近似 ours 的 5 模式增强 (旋转/缩放): 用水平/垂直翻转替代, 差异见声明文档
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='NormalizeWaterZScore', stats_file=stats_file),
    dict(type='PackSegInputs'),
]
val_pipeline = [
    dict(type='LoadWaterMultiBand', band_indices=img_band_indices),
    dict(type='LoadWaterAnnotations'),
    dict(type='NormalizeWaterZScore', stats_file=stats_file),
    dict(type='PackSegInputs'),
]
test_pipeline = val_pipeline

# 数据已在管线内 z-score 归一化 → mean=None 跳过二次归一化; 512 可被 32 整除不 padding
# 注: mmseg 1.2.2 SegDataPreProcessor 的参数名是 size_divisor
#     (pad_size_divisor 是 mmdet DetDataPreprocessor 的名字, 勿混用)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    size_divisor=32,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,          # 由 T/S/B 子配置按变体完整覆盖
        type='MM_mamba_vision_6ch',
        out_indices=(0, 1, 2, 3),
        in_chans=6,
        pretrained=None),       # ← 子配置指向 ImageNet-1K 权重路径
    decode_head=dict(
        type='UPerHead',
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,   # head 级参数: 会传给每个 loss 的 forward
        # 近似 ours 的 CombinedLoss(CE+Dice): mmseg 无 DWA/Focal/Boundary 现成实现,
        # 用 CE + Dice (权重同 DWA 初始值 1.0), 差异见声明文档。mmseg 1.2.2 注意:
        #   - CrossEntropyLoss 的 ignore_index 是 forward 参数 (head 自动传入),
        #     写进构造参数会 TypeError
        #   - SMP 的 smooth=1.0 对应 naive_dice=True (一阶幂) + eps=1.0 (平滑项 1)
        #   - 多类 Dice: use_sigmoid=False + activate=True → 内部 softmax
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                 avg_non_ignore=True),
            dict(type='DiceLoss', use_sigmoid=False, activate=True,
                 naive_dice=True, eps=1.0, loss_weight=1.0),
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=ignore_index,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=0.4, avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ---------- 优化器: AdamW(lr=5e-5, wd=1e-4) + AMP + 梯度裁剪 1.0 (同 ours) ----------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, error_if_nonfinite=False))

# ---------- 调度: warmup(15 等效epoch) + cosine → 5e-6 (同 ours) ----------
max_iters = 89400       # 100 等效epoch
warmup_iters = 13410    # 15 等效epoch
val_interval = 894      # 每 1 等效epoch 验证一次

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, end_factor=1.0,
         begin=0, end=warmup_iters, by_epoch=False),
    # 注: mmengine 的 CosineAnnealingLR 参数名是 eta_min (绝对最小 lr,
    # min_lr 会 TypeError); eta_min_ratio 才是相对 base_lr 的比例
    dict(type='CosineAnnealingLR', eta_min=5e-6, T_max=max_iters - warmup_iters,
         begin=warmup_iters, end=max_iters, by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)

# 训练监控: ours 风格进度条 (stderr, 每等效 epoch 一条; 无 tqdm 自动降级)
custom_hooks = [dict(type='TqdmEpochHook', val_interval=val_interval)]
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 每 10 等效epoch 存一次 periodic; 同时按 val mIoU 保存最优 (同 ours 的 best 策略)
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8940,
                    save_best='mIoU', rule='greater', max_keep_ckpts=3),
    # 同 ours: 早停基于 val mIoU, patience=20 (单位=等效epoch); rule 显式声明 greater
    early_stop=dict(type='EarlyStoppingHook', monitor='mIoU',
                    patience=20, min_delta=0.0, rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='train/image', seg_map='train/label'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='val/image', seg_map='val/label'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

# 验证/测试: ours 同一公式的指标表 (IoU/Prec/Recall/F1 + mIoU/mPA/OA/Kappa),
# 返回值保留 'mIoU' 键 → save_best / 早停不受影响
val_evaluator = dict(type='WaterOursMetric')
test_evaluator = val_evaluator

# 训练前注册水草插件 (数据集 / 变换 / 6ch 骨干)
custom_imports = dict(imports=['water_plugins'], allow_failed_imports=False)
