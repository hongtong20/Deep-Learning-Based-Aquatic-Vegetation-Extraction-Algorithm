"""
MambaVision-UPerNet (mmseg) 切片推理 + 拼接还原整幅 + 矢量化 shp + 可选精度评估

位置说明:
  本脚本自包含于对比实验工作区 (semantic_segmentation/), 只依赖本目录的
  water_plugins, 不引用 base1 (smp-trans-net sub base 1) 目录的任何文件。
  流程结构参照 ours 的 infer_test.py (切片分组 → 拼接还原 → 掩膜/矢量输出)。

功能:
  1. 按 --config 重建 MambaVision-UPerNet 并加载 --checkpoint 训练权重
  2. 对切片 (512×512 等) 批量推理, 按文件名把切片分组匹配回各自的原整幅影像
     (如 th_20240825_175.tif / th_20240825_177.tif → 原影像 th_20240825_image.tif),
     按裁剪网格规则 (或切片自带地理偏移) 拼接还原为整幅 → 全零无效区置 255 → 输出:
       <name>_mask.tif           拼接还原后的整幅 uint8 标签掩膜 (0-3, 255=nodata)
       <name>.shp/.shx/.dbf/.prj 矢量面文件, 字段: class_id, class_name
  3. 提供 --test_lbl_dir 时, 额外计算与训练日志完全一致格式的指标并写 metrics.txt:
     每类 IoU / Prec / Recall / F1 + mIoU / mPA / OA / Kappa + 混淆矩阵
  4. 逐切片预测 GeoTIFF 默认不导出, 需要 --save_tile_tif 开启

  [数据]   复用 config 中的 test 管线 (LoadWaterMultiBand / NormalizeWaterZScore)
           —— 与 FusionDataset(mode='val') 数值逐条等价, 波段组合与统计量 npz
           与训练时完全一致; 全零无效区判定同 LoadWaterMultiBand.zero_mask
  [指标]   同一混淆矩阵累积 (np.bincount, 忽略 255) → 同一公式
           compute_all_metrics_from_cm → 同一表格+混淆矩阵文本格式
           (实现取自 water_plugins/val_report.py, 已与 ours train_fusion.py /
           infer_test.py 逐行对照验证: 公式、表格、混淆矩阵块均一致)
  [导出]   同一 GeoTIFF 导出 (uint8, LZW, nodata=255, 保留地理参考) + metrics.txt
           (论文表格数字请使用本脚本输出, 不要用 mmseg IoUMetric 的打印)

用法 (在 semantic_segmentation/ 目录下):
    # 切片推理 → 拼接还原整幅 → 矢量化 shp (核心用法, 不需要标签)
    python infer_mmseg.py \
        --config configs/water/upernet_mamba_vision-T_water_6ch_512.py \
        --checkpoint work_dirs/mv_T_run1/best_mIoU_iter_xxxx.pth \
        --test_img_dir E:\...\dataset_8band\test\image \
        --src_img_dir  E:\...\整幅影像目录   # 含 th_20240825_image.tif 等 \
        --out_dir     E:\...\结果目录        # *_mask.tif + *.shp + metrics.txt

    # 同时评估精度 (切片标签目录, 可选)
    python infer_mmseg.py ... --test_lbl_dir E:\...\dataset_8band\test\label

注意:
  - --src_img_dir 必须提供: 拼接需要原影像的尺寸 (推算切片网格位置)、地理参考
    (shp 落位) 与全零无效区判断; 按文件名 <name>_<序号>.tif 与 <name>_image.tif
    自动配对 (忽略大小写, 递归子目录)
  - --overlap 必须与裁剪切片时使用的重叠率一致 (裁剪脚本默认 0)
  - 切片定位自动双模式: 若切片 tif 自带正确地理偏移 → 按坐标写入;
    若所有切片地理参考都等于原影像左上角 (当前裁剪脚本的情况) →
    按文件名序号 + 裁剪网格规则重算位置
  - 切片预处理兼容非 32 倍数切片的边缘 padding (edge 模式, 推理后裁回原尺寸)
  - --stats_file 可临时覆盖 config 中的统计量路径 (默认与训练同一份);
    config 是 mmseg 的"实验配方"文件: 推理时按它重建网络再载入 --checkpoint 权重
  - shp 输出需要额外安装: pip install geopandas  (未安装时只输出掩膜 tif 并提示)
"""

import os
import re
import sys

# =============================================================================
# 路径准备: 本目录 (water_plugins 在此)
# =============================================================================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:                 # 集群未装 tqdm 时降级为直通迭代 (不影响结果)
    def tqdm(x, **kwargs):
        return x
import rasterio
from rasterio import features, windows as rio_windows

try:
    import geopandas as gpd
    from shapely.geometry import shape
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ★ 指标实现来源: 本目录 water_plugins (与 ours 逐行验证一致), 无需引用 base1
from water_plugins.val_report import (
    _compute_all_metrics_from_cm as compute_all_metrics_from_cm,
    _format_metrics_table,
)

# 与 ours infer_test.py 完全同一组常量 (= WaterGeoTIFFDataset.METAINFO)
NUM_CLASSES = 4
IGNORE_INDEX = 255
CLASS_NAMES = ('Water', 'Submerged', 'Floating', 'Emergent')
PAD_MULTIPLE = 32  # MambaVision-UPerNet 总下采样倍数


def metrics_report(metrics, cm, title='测试集评估结果'):
    """与 ours infer_test.py metrics_report 逐行一致 (表格 + 混淆矩阵块)"""
    table = _format_metrics_table(metrics, CLASS_NAMES, title=title)
    lines = table.rstrip('\n').split('\n')
    lines.append(f"  {'-' * 48}")
    lines.append('  混淆矩阵 (行=真实, 列=预测):')
    header = f"  {'':<16s}" + ''.join(f'{n:>12s}' for n in CLASS_NAMES)
    lines.append(header)
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f'  {name:<16s}'
                     + ''.join(f'{int(cm[i, j]):>12d}' for j in range(NUM_CLASSES)))
    return '\n'.join(lines)


import water_plugins  # noqa: F401  # 注册水草数据集/变换/6ch 骨干
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS, TRANSFORMS


def parse_args():
    parser = argparse.ArgumentParser(
        description='MambaVision-UPerNet (mmseg) 切片推理 + 拼接还原整幅 + '
                    '矢量化 shp + 可选精度评估 (流程与 infer_test.py 一致)')
    parser.add_argument('--config', type=str,
                        default=r'C:\Users\hongtong\PyCharmMiscProject\semantic_segmentation'
                                r'\configs\water\upernet_mamba_vision-T_water_6ch_512.py',
                        help='训练用的水草 config (提供模型结构与 test 数据管线)')
    parser.add_argument('--checkpoint', type=str,
                        default=r"D:\小论文\训练权重文件\mamba vision\mv_T_run_6\best_mIoU_iter_23244.pth",
                        help='mmseg 训练输出权重 (best_mIoU_iter_*.pth)')
    parser.add_argument('--test_img_dir', type=str,
                        default=r'E:\水草样本512修改二次\dataset_8band\test\image',
                        help='切片影像目录 (与 infer_test.py 默认一致)')
    parser.add_argument('--test_lbl_dir', type=str,
                        default=r'E:\水草样本512修改二次\dataset_8band\test\label',
                        help='切片标签目录 (可选; 提供时计算精度指标并写 metrics.txt)')
    parser.add_argument('--src_img_dir', type=str,
                        default=r'E:\水草样本512修改二次\dataset\test\image',
                        help='[拼接必需] 原整幅影像目录 (含 <name>_image.tif, 如 '
                             'th_20240825_image.tif); 按文件名把切片 <name>_<序号>.tif '
                             '拼回整幅, 输出 <name>_mask.tif + <name>.shp')
    parser.add_argument('--stats_file', type=str,
                        default=r'C:\Users\hongtong\PyCharmMiscProject\semantic_segmentation\global_stats_enc1_rgbnirsar.npz',
                        help='统计量 npz (默认: config 中的路径; 该路径本机不存在时'
                             '自动回退到本机常见位置, 一般无需传)')
    parser.add_argument('--out_dir', type=str,
                        default=r'D:\小论文\训练权重文件\mamba vision\test6',
                        help='预测输出目录 (默认: <test>/predict_<config名>)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='裁剪切片时使用的重叠率 repetition_rate, 须与裁剪脚本一致 (默认 0)')
    parser.add_argument('--fp16', action='store_true',
                        help='CUDA autocast 半精度推理')
    parser.add_argument('--min_area', type=int, default=0,
                        help='shp 矢量化的最小图斑面积过滤, 单位=像素 (0=不过滤)')
    parser.add_argument('--no_shp', action='store_true',
                        help='只输出掩膜 GeoTIFF, 不做矢量化')
    parser.add_argument('--save_tile_tif', action='store_true',
                        help='额外导出逐切片预测 GeoTIFF (默认不导出, 只输出整幅拼接结果)')
    return parser.parse_args()


def _resolve_stats_file(stats_file):
    """config 里的统计量路径是集群绝对路径; 本机不存在时按常见位置回退查找"""
    if stats_file is None or os.path.isfile(stats_file):
        return stats_file
    cands = [
        os.path.join(REPO_ROOT, 'global_stats_enc1_rgbnirsar.npz'),
        os.path.join(r'C:\Users\hongtong\PyCharmMiscProject\smp-trans-net sub base 1',
                     'checkpoint', 'global_stats_enc1_rgbnirsar.npz'),
    ]
    for c in cands:
        if os.path.isfile(c):
            print(f'[stats] config 中的 {stats_file} 本机不存在, 回退使用 {c}')
            return c
    return stats_file      # 找不到则保持原路径, 由 np.load 报出明确错误


def build_pipeline(cfg, stats_file_override):
    """从 config 的 test 管线构建影像变换

    - 去掉 PackSegInputs (直接取 numpy 结果)
    - 去掉 LoadWaterAnnotations (标签仅在 --test_lbl_dir 提供时单独读取评估,
      数值等价逻辑见 tile_confusion; 无标签时也能推理拼接)
    - NormalizeWaterZScore 的 stats_file 按 --stats_file / 回退规则解析
    """
    transforms = []
    for t in cfg.test_dataloader.dataset.pipeline:
        t = dict(t)
        if t['type'] in ('PackSegInputs', 'LoadWaterAnnotations'):
            continue
        if t['type'] == 'NormalizeWaterZScore':
            stats = _resolve_stats_file(stats_file_override or t.get('stats_file'))
            if stats:
                t['stats_file'] = stats
        transforms.append(TRANSFORMS.build(t))
    return transforms


# =============================================================================
# 切片文件名解析与分组 (命名规则与裁剪脚本一致: <原影像名>_<序号>.tif)
# =============================================================================

def parse_tile_name(fname):
    """'XXX_123.tif' → ('XXX', 123)。序号与裁剪脚本中的 count 一致 (从 1 开始)"""
    m = re.match(r'^(.*)_(\d+)$', os.path.splitext(fname)[0])
    if m is None:
        return None, None
    return m.group(1), int(m.group(2))


def group_tile_paths(tile_paths):
    """把切片按原影像名分组。返回 (groups: {base: {idx: path}}, skipped)"""
    groups, skipped = {}, []
    for p in tile_paths:
        base, idx = parse_tile_name(os.path.basename(p))
        if base is None or idx <= 0:
            skipped.append(p)
            continue
        groups.setdefault(base, {})[idx] = p
    return groups, skipped


def collect_source_images(src_img_dir):
    """递归扫描原整幅影像目录, 建立 {小写影像名: 路径} 映射 (匹配 *_image.tif)"""
    src_map = {}
    for root, _, files in os.walk(src_img_dir):
        for f in files:
            low = f.lower()
            if low.endswith('_image.tif'):
                src_map[low[:-len('_image.tif')]] = os.path.join(root, f)
    return src_map


# =============================================================================
# 裁剪网格位置还原 (与裁剪脚本的切片生成顺序严格一致, 同 ours infer_test.py)
# =============================================================================

def grid_count(full_size, crop, rep):
    """裁剪脚本主网格数: int((size - crop*rep) / (crop*(1-rep)))"""
    return int((full_size - crop * rep) / (crop * (1 - rep)))


def tile_origin(idx, num_i, num_j, height, width, crop, rep):
    """
    切片序号 → 原影像内左上角像素坐标 (row, col)。
    顺序与裁剪脚本一致: 主网格(行优先) → 最后一列 → 最后一行 → 右下角。
    """
    n_main = num_i * num_j
    if idx <= n_main:                                    # 阶段1: 主网格
        i, j = (idx - 1) // num_j, (idx - 1) % num_j
        return int(i * crop * (1 - rep)), int(j * crop * (1 - rep))
    k = idx - n_main
    if k <= num_i:                                       # 阶段2: 最后一列
        return int((k - 1) * crop * (1 - rep)), width - crop
    if k <= num_i + num_j:                               # 阶段3: 最后一行
        return height - crop, int((k - num_i - 1) * crop * (1 - rep))
    return height - crop, width - crop                   # 阶段4: 右下角


# =============================================================================
# 切片推理 (预处理 = config test 管线, 与 FusionDataset 数值等价)
# =============================================================================

def predict_tiles(model, tile_paths, transforms, device, fp16=False, batch_size=4):
    """
    逐批推理切片 (兼容非 32 倍数切片的边缘 padding)。

    预处理直接复用 config 的 LoadWaterMultiBand + NormalizeWaterZScore
    (与 ours infer_test.py 的 rasterio 读取 + normalize_bands 数值等价);
    全零无效区取 LoadWaterMultiBand 计算的 zero_mask。

    Returns:
        preds: {path: (h, w) uint8 标签}
        metas: {path: (h, w, transform, invalid)}
    """
    preds, metas = {}, {}
    for b in tqdm(range(0, len(tile_paths), batch_size), desc='切片推理'):
        chunk = tile_paths[b:b + batch_size]
        arrs = []
        for p in chunk:
            results = {'img_path': p}
            for t in transforms:
                results = t(results)
            with rasterio.open(p) as src:
                transform = src.transform
            invalid = results['zero_mask']                       # 全零无效区
            h, w = results['img'].shape[:2]
            metas[p] = (h, w, transform, invalid)
            arrs.append(results['img'])                          # (H, W, C) 已归一化

        x = np.stack(arrs).transpose(0, 3, 1, 2)                 # → (N, C, H, W)
        h, w = x.shape[2:]
        pad_h = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
        pad_w = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
        if pad_h or pad_w:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='edge')
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=fp16):
            feats = model.backbone(x)
            logits = model.decode_head(feats)
            logits = F.interpolate(logits, size=x.shape[2:],
                                   mode='bilinear', align_corners=False)
        batch_pred = torch.argmax(logits.float(), dim=1).cpu().numpy().astype(np.uint8)

        for p, pred in zip(chunk, batch_pred):
            preds[p] = pred[:h, :w]
    return preds, metas


# =============================================================================
# 标签评估 (与 ours infer_test.py / LoadWaterAnnotations 数值等价)
# =============================================================================

def sanitize_label(label, invalid):
    """标签切片预处理 (与 FusionDataset 完全一致):
    label==4→255, 越界值→255, 影像全零区→255。返回 int64 标签。"""
    label = label.astype(np.int64)
    label[label == 4] = IGNORE_INDEX
    valid = ((label >= 0) & (label <= 3)) | (label == IGNORE_INDEX)
    label[~valid] = IGNORE_INDEX
    label[invalid] = IGNORE_INDEX
    return label


def tile_confusion(pred, label_raw, invalid):
    """单切片混淆矩阵累积 (跳过 ignore_index)。返回 (4×4 int64 累积阵, 有效像素数)"""
    label = sanitize_label(label_raw, invalid)
    valid = label != IGNORE_INDEX
    gt = label[valid]
    pr = pred[valid].astype(np.int64)
    cm = np.bincount(gt * NUM_CLASSES + pr,
                     minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cm, int(valid.sum())


# =============================================================================
# 导出 (与 ours infer_test.py 同一 GeoTIFF / shp 规格)
# =============================================================================

def save_mask_tif(mask, profile, dst_path):
    """uint8 标签 GeoTIFF, 保留原地理参考"""
    profile = profile.copy()
    profile.update(count=1, dtype='uint8', compress='lzw', nodata=IGNORE_INDEX)
    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(mask, 1)


def mask_to_shp(mask, transform, crs, shp_path, min_area_px=0):
    """标签掩膜 → ESRI Shapefile 面要素 (过滤 255 nodata, 可选最小面积过滤)"""
    shapes_gen = features.shapes(
        mask, mask=(mask != IGNORE_INDEX), transform=transform)

    # 单像素面积 (CRS 单位²), 用于把几何面积换算回像素数
    pix_area = abs(transform.a * transform.e - transform.b * transform.d) or 1.0

    records = []
    for geom, value in shapes_gen:
        poly = shape(geom)
        if min_area_px > 0 and (poly.area / pix_area) < min_area_px:
            continue
        v = int(value)
        if 0 <= v < NUM_CLASSES:
            records.append({'class_id': v,
                            'class_name': CLASS_NAMES[v],
                            'geometry': poly})

    if not records:
        print('[警告] 无可矢量化的有效区域 (全部为 255 无效区), 跳过 shp 输出')
        return 0

    gdf = gpd.GeoDataFrame(records, columns=['class_id', 'class_name', 'geometry'],
                           crs=crs)
    gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
    return len(gdf)


def export_tile_pred(tile_path, pred, dst_path):
    """导出单切片预测 GeoTIFF (uint8, LZW, nodata=255, 保留切片地理参考)"""
    with rasterio.open(tile_path) as src:
        profile = src.profile
    profile.update(count=1, dtype='uint8', compress='lzw', nodata=IGNORE_INDEX)
    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(pred, 1)


def compute_invalid_mask(src, band_groups, height, width, block=2048):
    """分块读取原影像, 判断全零无效区 (所有用到波段的像素值全为 0 → 无效)

    与 LoadWaterMultiBand 的切片级判定保持一致: NaN/Inf 先置 0 后再判全零。"""
    all_idx = sorted({i for grp in band_groups for i in grp})
    pos = {v: k for k, v in enumerate(all_idx)}
    invalid = np.ones((height, width), dtype=bool)
    for r0 in range(0, height, block):
        for c0 in range(0, width, block):
            h = min(block, height - r0)
            w = min(block, width - c0)
            win = rio_windows.Window(c0, r0, w, h)
            data = np.nan_to_num(src.read(all_idx, window=win).astype(np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)
            inv = np.ones((h, w), dtype=bool)
            for grp in band_groups:
                sel = data[[pos[i] for i in grp]]
                inv &= np.all(sel == 0, axis=0)
            invalid[r0:r0 + h, c0:c0 + w] = inv
    return invalid


# =============================================================================
# 拼接还原: 切片预测 → 整幅掩膜 + 矢量 shp
# =============================================================================

def process_group(base, tiles, src_path, preds, metas, band_groups, args):
    """
    单幅原影像的处理: 定位切片 → 拼接 → 掩膜 GeoTIFF + 矢量 shp 输出。
    (推理已完成, preds/metas 为 predict_tiles 的返回值)

    Args:
        base (str): 原影像名 (切片文件名前缀)
        tiles (dict): {切片序号: 切片路径}
        src_path (str): 原整幅影像路径
        preds / metas: predict_tiles 返回的预测结果与切片元信息
        band_groups (list): 参与无效区判断的波段索引组
        args: 命令行参数 (使用 overlap/min_area/no_shp/out_dir)
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*60}\n处理: {base} | 切片 {len(tiles)} 个 | 原影像: {src_path}")

    # 原影像信息: 尺寸/地理参考 + 全零无效区 (分块读, 防止大图 OOM)
    with rasterio.open(src_path) as src:
        height, width = src.height, src.width
        profile, transform, crs = src.profile, src.transform, src.crs
        invalid = compute_invalid_mask(src, band_groups, height, width)
    print(f"原影像尺寸: {height}×{width}, 无效区占 {invalid.mean()*100:.1f}%")

    # 切片尺寸一致性检查
    sizes = {(metas[p][0], metas[p][1]) for p in tiles.values()}
    if len(sizes) != 1:
        print(f"[错误] 组内切片尺寸不一致 {sizes}, 跳过该组")
        return False
    crop_h, crop_w = sizes.pop()
    if crop_h != crop_w:
        print(f"[错误] 切片非正方形 ({crop_h}×{crop_w}), 网格还原要求正方形, 跳过该组")
        return False
    if crop_h > height or crop_w > width:
        print(f"[错误] 切片尺寸 {crop_h} 大于原影像 {height}×{width}, 跳过该组")
        return False

    # 定位模式: 切片 transform 互不相同 → 自带地理偏移, 直接按坐标写入;
    #           全部相同 (即都等于原影像左上角, 当前裁剪脚本的情况) → 按序号+网格还原
    uniq_transforms = {tuple(list(metas[p][2])[:6]) for p in tiles.values()}
    use_geo = len(uniq_transforms) >= 2

    origins = {}
    if use_geo:
        if transform.b != 0 or transform.d != 0:
            print("[警告] 原影像含旋转项, 按坐标定位可能不准, 已改用网格还原")
            use_geo = False
        else:
            for idx, p in tiles.items():
                t = metas[p][2]
                c0 = int(round((t.c - transform.c) / transform.a))
                r0 = int(round((t.f - transform.f) / transform.e))
                origins[idx] = (r0, c0)
            print("切片定位: 按切片自带地理坐标偏移")

    if not use_geo:
        num_i = grid_count(height, crop_h, args.overlap)
        num_j = grid_count(width, crop_w, args.overlap)
        total = (num_i + 1) * (num_j + 1)
        max_idx = max(tiles)
        if max_idx > total:
            print(f"[错误] 最大切片序号 {max_idx} 超过网格总数 {total} "
                  f"(num_i={num_i}, num_j={num_j}); 请检查 --overlap 是否与裁剪时一致, 跳过该组")
            return False
        origins = {idx: tile_origin(idx, num_i, num_j, height, width,
                                    crop_h, args.overlap) for idx in tiles}
        print(f"切片定位: 按文件名序号 + 网格规则 (主网格 {num_i}×{num_j}, "
              f"总切片数上限 {total})")

    # 拼接 (重叠区后写覆盖)
    mosaic = np.full((height, width), IGNORE_INDEX, dtype=np.uint8)
    for idx in sorted(origins):
        r0, c0 = origins[idx]
        if not (0 <= r0 <= height - crop_h and 0 <= c0 <= width - crop_w):
            print(f"[错误] 切片 #{idx} 计算位置 ({r0},{c0}) 越界, 跳过该组")
            return False
        mosaic[r0:r0 + crop_h, c0:c0 + crop_w] = preds[tiles[idx]]

    mosaic[invalid] = IGNORE_INDEX

    # 每类像素统计
    counts = np.bincount(mosaic.ravel(), minlength=256)
    valid_total = int(height * width - counts[IGNORE_INDEX])
    stat = ", ".join(f"{CLASS_NAMES[c]}={counts[c]/max(valid_total, 1)*100:.1f}%"
                     for c in range(NUM_CLASSES))
    print(f"预测构成 (有效区 {valid_total/(height*width)*100:.1f}%): {stat}")

    # 掩膜 GeoTIFF + 矢量化 shp
    tif_path = os.path.join(args.out_dir, f"{base}_mask.tif")
    save_mask_tif(mosaic, profile, tif_path)
    print(f"拼接掩膜已保存: {tif_path}")

    if not args.no_shp and HAS_GEOPANDAS:
        shp_path = os.path.join(args.out_dir, f"{base}.shp")
        try:
            n_feat = mask_to_shp(mosaic, transform, crs, shp_path,
                                 min_area_px=args.min_area)
            print(f"矢量已保存: {shp_path} ({n_feat} 个面要素)")
        except Exception as e:
            print(f"[错误] {base} 矢量化失败: {e} (掩膜 tif 已保存, 不影响后续影像)")
    elif not args.no_shp:
        print("[提示] 未安装 geopandas, 跳过矢量化 (安装后重跑本脚本即可: pip install geopandas)")

    return True


# =============================================================================
# 推理主流程
# =============================================================================

def main():
    args = parse_args()
    tag = os.path.splitext(os.path.basename(args.config))[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fp16 = args.fp16 and device.type == 'cuda'
    print(f'使用设备: {device}' + (' (fp16)' if fp16 else ''))

    if args.no_shp is False and not HAS_GEOPANDAS:
        print('[警告] 未安装 geopandas, 无法输出 shp (本次只输出掩膜 GeoTIFF)\n'
              '       安装: pip install geopandas')

    if not os.path.isdir(args.test_img_dir):
        raise SystemExit(f'[错误] 切片目录不存在: {args.test_img_dir}')
    if not os.path.isdir(args.src_img_dir):
        raise SystemExit(f'[错误] 原影像目录不存在: {args.src_img_dir}')
    if args.test_lbl_dir and not os.path.isdir(args.test_lbl_dir):
        print(f"[提示] 标签目录不存在: {args.test_lbl_dir}, 跳过精度指标计算")
        args.test_lbl_dir = None

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.test_img_dir)), f'predict_{tag}')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'预测结果输出目录: {args.out_dir}')

    cfg = Config.fromfile(args.config)

    # ========== 1. 模型: 按 config 重建, 禁用 ImageNet 加载, 载入训练权重 ==========
    print(f'\n加载权重: {args.checkpoint}')
    cfg.model.backbone.pretrained = None
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    model = MODELS.build(cfg.model)
    # torch>=2.6 torch.load 默认 weights_only=True: 该 ckpt 的 meta/message_hub
    # 含集群 numpy1.x 路径的 global (numpy.core.multiarray), 本机 numpy>=2 下
    # add_safe_globals 按 __module__ 字符串匹配会失配。ckpt 为本仓库训练产出
    # (可信), 直接 weights_only=False 载入 state_dict
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
    if missing or unexpected:
        print(f'[warn] state_dict missing={len(missing)} '
              f'unexpected={len(unexpected)}')
    model = model.to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'模型参数量: {params_m:.2f}M')

    # ========== 2. 数据: 复用 config 的 test 管线 (与训练同一波段/统计量) ==========
    init_default_scope('mmseg')
    transforms = build_pipeline(cfg, args.stats_file)
    band_indices = next((t.band_indices for t in transforms
                         if hasattr(t, 'band_indices')), None)
    print(f'波段组合: rasterio 索引 {band_indices} (来自 config, 与训练一致)')
    if band_indices is None:
        raise ValueError('未在 config test 管线中找到 LoadWaterMultiBand (band_indices)')

    band_groups = [list(band_indices)]   # 参与整幅无效区判断的波段组 (单编码器)

    # ========== 3. 切片批量推理 ==========
    tile_paths = sorted(
        os.path.join(args.test_img_dir, f)
        for f in os.listdir(args.test_img_dir)
        if f.lower().endswith(('.tif', '.tiff')))
    if not tile_paths:
        raise SystemExit(f'[错误] 切片目录下没有 tif 文件: {args.test_img_dir}')
    print(f'\n切片样本数: {len(tile_paths)}')

    preds, metas = predict_tiles(model, tile_paths, transforms,
                                 device, fp16, args.batch_size)

    # ========== 4. 精度评估 (有标签时) + 可选的逐切片导出 ==========
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    n_labeled = 0
    for p in tile_paths:
        fname = os.path.basename(p)
        if args.save_tile_tif:
            export_tile_pred(p, preds[p], os.path.join(args.out_dir, fname))
        if args.test_lbl_dir:
            lbl_path = os.path.join(args.test_lbl_dir, fname)
            if os.path.exists(lbl_path):
                with rasterio.open(lbl_path) as lsrc:
                    label = lsrc.read(1)
                cm_add, _ = tile_confusion(preds[p], label, metas[p][3])
                cm += cm_add
                n_labeled += 1

    if n_labeled > 0:
        metrics = compute_all_metrics_from_cm(cm, NUM_CLASSES)
        report = metrics_report(metrics, cm, title=f'测试集评估结果 [{tag}]')
        print(report)
        report_path = os.path.join(args.out_dir, 'metrics.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f'model: {tag} (MambaVision-UPerNet, mmseg)\n')
            f.write(f'params: {params_m:.3f}M\n')
            f.write(f'config: {args.config}\n')
            f.write(f'checkpoint: {args.checkpoint}\n')
            f.write(f'test_img_dir: {args.test_img_dir}\n')
            f.write(f'input bands: {len(band_indices)}ch, rasterio {band_indices}\n')
            f.write(f'样本数: {len(tile_paths)} (含标签 {n_labeled} 个)\n\n')
            f.write(report + '\n')
        print(f'\n指标已保存: {report_path}')
    elif args.test_lbl_dir:
        print(f'\n[提示] 标签目录 {args.test_lbl_dir} 中未找到与切片同名的标签, 跳过精度指标计算')

    if args.save_tile_tif:
        print(f'逐切片预测 ({len(tile_paths)} 个 GeoTIFF) 已保存: {args.out_dir}')

    # ========== 5. 拼接还原整幅 + 矢量化 shp ==========
    groups, skipped = group_tile_paths(tile_paths)
    src_map = collect_source_images(args.src_img_dir)
    unmatched = [b for b in groups if b.lower() not in src_map]
    print(f'\n切片分组: {len(groups)} 幅影像, 共 {len(tile_paths)} 个切片')
    if skipped:
        print(f'[警告] {len(skipped)} 个切片无法解析文件名序号, 无法参与拼接')
    if unmatched:
        print(f'[警告] {len(unmatched)} 个分组在原影像目录找不到对应的 <name>_image.tif, 跳过: '
              + ', '.join(unmatched[:5]) + (' ...' if len(unmatched) > 5 else ''))

    for base in sorted(groups):
        if base.lower() in src_map:
            process_group(base, groups[base], src_map[base.lower()],
                          preds, metas, band_groups, args)

    print(f'\n全部完成, 结果目录: {args.out_dir}')


if __name__ == '__main__':
    main()
