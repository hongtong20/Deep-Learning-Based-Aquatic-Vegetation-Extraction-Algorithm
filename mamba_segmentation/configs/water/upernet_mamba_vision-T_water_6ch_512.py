# MambaVision-T + UPerNet · 水草 4 类 · 6 波段 512×512 (对比实验)
# 骨干参数与官方 configs/mamba_vision/mamba_vision_160k_ade20k-512x512_tiny.py 逐项一致
_base_ = ['./_base_water.py']

# ImageNet-1K 预训练权重 (与 ours/SMP 对照的 ImageNet 预训练变量对齐)。
# 下载地址与放置方式见仓库根 WATER_ADAPTATION.md; 离线训练设为 None
imagenet_ckpt = 'ckpts/mambavision_tiny_1k.pth.tar'

model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_mamba_vision_6ch',
        out_indices=(0, 1, 2, 3),
        in_chans=6,
        pretrained=imagenet_ckpt,
        depths=(1, 3, 8, 4),
        num_heads=(2, 4, 8, 16),
        window_size=(8, 8, 64, 32),
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.3,
        norm_layer='ln2d',
        layer_scale=None),
    decode_head=dict(in_channels=[80, 160, 320, 640], num_classes=4),
    auxiliary_head=dict(in_channels=320, num_classes=4))
