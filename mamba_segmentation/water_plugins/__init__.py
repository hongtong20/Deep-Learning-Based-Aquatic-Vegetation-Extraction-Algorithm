"""
水草 4 类语义分割 · mmseg 对比实验插件

导入即注册 (供 config 的 custom_imports 使用):
  数据集:   WaterGeoTIFFDataset
  变换:     LoadWaterMultiBand / LoadWaterAnnotations / NormalizeWaterZScore
  骨干:     MM_mamba_vision_6ch (6 通道输入版 MambaVision)
  监控:     WaterOursMetric (ours 同一公式验证表) / TqdmEpochHook (进度条)

数值等价性以 ours 的 dataset_enhanced.FusionDataset(mode='val') 为基准,
逐条对应关系见各文件 docstring 与 WATER_ADAPTATION.md。
"""

from . import backbone_6ch, val_report, water_dataset  # noqa: F401

__all__ = ['backbone_6ch', 'val_report', 'water_dataset']
