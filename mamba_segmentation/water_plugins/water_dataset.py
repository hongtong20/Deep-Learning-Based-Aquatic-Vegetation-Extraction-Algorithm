"""
水草 GeoTIFF 数据集 + 与 ours 数值等价的预处理变换 (mmseg 插件)

数值等价基准: ours 的 dataset_enhanced.FusionDataset(mode='val', augment=False)
单编码器 6 通道路径, 逐条对应:

  [FusionDataset.__getitem__]                     [本文件]
  ---------------------------------------------------------------
  raw = src.read([3,2,1,4,8,7])                   LoadWaterMultiBand
  raw = process_outliers(raw)          NaN/Inf→0   LoadWaterMultiBand
  all_zero = np.all(raw==0, axis=ch)   全零无效区   LoadWaterMultiBand (zero_mask)
  label==4 → 255; 越界值 → 255                      LoadWaterAnnotations
  label[all_zero] = 255                             LoadWaterAnnotations
  normalize_image: (x-mean)/std (std<=1e-8 仅减均值) NormalizeWaterZScore
  nan_to_num + clip(-5, 5)                          NormalizeWaterZScore

统计量文件与 ours/SMP 对照共用同一份 npz
(global_stats_enc1_rgbnirsar.npz, 键名: means / stds)。
"""

import os
import os.path as osp

import numpy as np
import rasterio
from mmcv.transforms import BaseTransform
from mmengine.dataset import BaseDataset

from mmseg.registry import DATASETS, TRANSFORMS


# =============================================================================
# 数据集
# =============================================================================

@DATASETS.register_module()
class WaterGeoTIFFDataset(BaseDataset):
    """水草 4 类 GeoTIFF 数据集 (8 波段影像 + 单波段标签, 文件名一一对应)

    Args:
        data_root:    数据集根目录 (相对路径基于此拼接; 传绝对路径时可留空)
        data_prefix:  dict(img=影像目录, seg_map=标签目录)
        pipeline:     变换列表 (dict 或已构建的变换对象)
        test_mode:    是否为测试模式

    METAINFO 的 4 类与 ours 的 CLASS_NAMES 一致:
        0=Water, 1=Submerged, 2=Floating, 3=Emergent (255=ignore)
    """

    METAINFO = dict(
        classes=('Water', 'Submerged', 'Floating', 'Emergent'),
        palette=[[0, 0, 139], [0, 200, 0], [255, 255, 0], [255, 0, 0]],
    )

    def __init__(self,
                 data_root='',
                 data_prefix=dict(img=None, seg_map=None),
                 pipeline=None,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            ann_file='',
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            lazy_init=False,
            **kwargs)
        self.pipeline = []
        for t in (pipeline or []):
            # 允许传入 dict (由 mmseg TRANSFORMS 构建) 或已构建的变换对象
            self.pipeline.append(t if callable(t) else TRANSFORMS.build(dict(t)))

    def load_data_list(self):
        img_dir = self.data_prefix.get('img')
        seg_dir = self.data_prefix.get('seg_map')
        if not img_dir or not seg_dir:
            raise RuntimeError(
                'WaterGeoTIFFDataset 需要 data_prefix=dict(img=..., seg_map=...) '
                f'当前: {self.data_prefix}')

        names = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.tif', '.tiff')))
        data_list = []
        skipped = 0
        for name in names:
            seg_path = osp.join(seg_dir, name)
            if not osp.isfile(seg_path):
                skipped += 1
                continue
            data_list.append(dict(
                img_path=osp.join(img_dir, name), seg_map_path=seg_path))

        if not data_list:
            raise RuntimeError(f'未在 {img_dir} 下找到成对的 .tif 影像/标签')
        if skipped:
            print(f'[WaterGeoTIFFDataset] 警告: {skipped} 个影像缺少同名标签, 已跳过')

        # 与 infer_test.py / infer_baseline.py 的 dataset.image_names 用法对齐
        self.image_names = [osp.basename(d['img_path']) for d in data_list]
        print(f'WaterGeoTIFFDataset: {len(data_list)} 对样本 ({img_dir})')
        return data_list

    def __getitem__(self, idx):
        # get_data_info: 兼容 mmengine serialize_data (data_list 可能被序列化)
        results = self.get_data_info(idx)
        results['sample_idx'] = idx
        for t in self.pipeline:
            results = t(results)
        return results


# =============================================================================
# 变换 (与 FusionDataset 数值等价)
# =============================================================================

@TRANSFORMS.register_module()
class LoadWaterMultiBand(BaseTransform):
    """读取 GeoTIFF 指定波段 → (H, W, C) float32, 并计算全零无效区掩膜

    默认 band_indices=(3, 2, 1, 4, 8, 7) 即 R,G,B,NIR,VV,VH
    (rasterio 1-based 索引, 与 ours 的 ENC1_BAND_PRESETS[6] 完全一致)。

    数值对应 FusionDataset:
      raw = nan_to_num(read(bands))          # process_outliers: NaN/Inf → 0
      zero_mask = np.all(raw == 0, axis=ch)  # 全零区 → 标签将被置 255
    """

    def __init__(self, band_indices=(3, 2, 1, 4, 8, 7)):
        self.band_indices = list(band_indices)

    def transform(self, results):
        with rasterio.open(results['img_path']) as src:
            arr = src.read(self.band_indices)
        img = np.nan_to_num(arr.astype(np.float32, copy=True),
                            nan=0.0, posinf=0.0, neginf=0.0)   # process_outliers
        img = np.ascontiguousarray(img.transpose(1, 2, 0))     # CHW → HWC
        results['img'] = img
        results['ori_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape
        results['zero_mask'] = np.all(img == 0, axis=2)        # 全零无效区
        results['img_fields'] = ['img']
        return results


@TRANSFORMS.register_module()
class LoadWaterAnnotations(BaseTransform):
    """读取标签 GeoTIFF 并做与 FusionDataset 完全相同的预处理:
        1. 标签 4 → 255 (ignore)
        2. 超出 {0..3, 255} 的异常值 → 255
        3. 影像全零区 (zero_mask) → 255
    """

    def transform(self, results):
        with rasterio.open(results['seg_map_path']) as src:
            gt = src.read(1)
        gt = gt.astype(np.int64)
        gt[gt == 4] = 255
        valid = ((gt >= 0) & (gt <= 3)) | (gt == 255)
        gt[~valid] = 255
        zero_mask = results.get('zero_mask', None)
        if zero_mask is not None:
            gt[zero_mask] = 255
        results['gt_seg_map'] = gt
        results['seg_fields'] = ['gt_seg_map']
        return results


@TRANSFORMS.register_module()
class NormalizeWaterZScore(BaseTransform):
    """与 FusionDataset 完全同一数值的全局 z-score 归一化 (逐条等价):

        1. NaN/Inf → 0                                  (normalize_image 第 1 步)
        2. std 有限且 > 1e-8: (x-mean)/std; 否则仅减均值  (normalize_image 第 2 步)
        3. 再次 NaN/Inf → 0 + 裁剪到 [-5, 5]             (__getitem__ 末尾防护)

    Args:
        stats_file: 训练集统计量 npz (与 ours 共用同一份
                    global_stats_enc1_rgbnirsar.npz, 键名 means/stds,
                    顺序对应 band_indices)
    """

    def __init__(self, stats_file):
        stats = np.load(stats_file)
        self.stats_file = stats_file
        self.mean = np.asarray(stats['means'], dtype=np.float32).reshape(1, 1, -1)
        self.std = np.asarray(stats['stds'], dtype=np.float32).reshape(1, 1, -1)

    def transform(self, results):
        img = np.nan_to_num(results['img'].astype(np.float32, copy=True),
                            nan=0.0, posinf=0.0, neginf=0.0)
        if img.shape[2] != self.mean.shape[2]:
            raise ValueError(
                f'波段数不匹配: 影像 {img.shape[2]}ch vs 统计量 {self.mean.shape[2]}ch '
                f'({self.stats_file}); 请检查 LoadWaterMultiBand.band_indices')
        std_valid = np.isfinite(self.std) & (self.std > 1e-8)
        safe_std = np.where(std_valid, self.std, np.float32(1.0))
        out = (img - self.mean) / safe_std
        out = np.where(std_valid, out, img - self.mean)     # 无效 std: 仅减均值
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -5.0, 5.0)                       # 同 ours 的 ±5 裁剪
        results['img'] = np.ascontiguousarray(out.astype(np.float32, copy=False))
        return results
