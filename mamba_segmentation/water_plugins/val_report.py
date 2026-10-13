"""训练监控插件 (mmseg): ours 风格进度条 + ours 同一公式的逐 epoch 验证表

1. WaterOursMetric:
   混淆矩阵与全部指标公式与 ours train_fusion.compute_all_metrics_from_cm
   逐行同一实现 (mIoU 只对 union>0 的类平均; mPA = mean Recall; OA/Kappa
   标准式), 表格格式与 train_fusion.print_metrics_table 一致。
   返回 dict 保留 'mIoU' 键 (×100), 供 CheckpointHook(save_best) 与
   EarlyStoppingHook(monitor) 继续工作。

2. TqdmEpochHook:
   每个等效 epoch (894 iters) 一条 tqdm 进度条, 形如
     E38/100: 36%|███▌ | 323/894 [03:18<04:54, 1.94it/s, loss=..., acc=...]
   输出到 stderr (SLURM 的 single-*.err); 集群未装 tqdm 时自动禁用
   (不影响训练, mmengine 默认的逐 50 iter 日志仍然存在)。
"""

import sys

import numpy as np

from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmseg.registry import HOOKS, METRICS
from mmseg.evaluation.metrics import IoUMetric

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# =============================================================================
# 验证指标: 与 ours compute_all_metrics_from_cm 逐行同一公式
# =============================================================================

def _compute_all_metrics_from_cm(cm, num_classes):
    """与 ours/train_fusion.py compute_all_metrics_from_cm 逐行一致 (勿改动)"""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    union = tp + fp + fn
    iou = np.where(union > 0, tp / union, 0.0)

    pred_total = tp + fp
    precision = np.where(pred_total > 0, tp / pred_total, 0.0)

    gt_total = tp + fn
    recall = np.where(gt_total > 0, tp / gt_total, 0.0)

    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / (precision + recall), 0.0)

    valid_iou = iou[union > 0]
    miou = valid_iou.mean() if len(valid_iou) > 0 else 0.0

    valid_recall = recall[gt_total > 0]
    mpa = valid_recall.mean() if len(valid_recall) > 0 else 0.0

    total_pixels = cm.sum()
    oa = tp.sum() / total_pixels if total_pixels > 0 else 0.0

    if total_pixels > 0:
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        pe = np.sum(row_sums * col_sums) / (total_pixels ** 2)
        kappa = (oa - pe) / (1.0 - pe) if pe < 1.0 else 0.0
    else:
        kappa = 0.0

    return {
        'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1,
        'miou': miou, 'mpa': mpa, 'oa': oa, 'kappa': kappa,
    }


def _format_metrics_table(metrics, class_names, title=''):
    """与 ours/train_fusion.py print_metrics_table 同一格式"""
    lines = []
    if title:
        lines.append('')
        lines.append(f"  {'=' * 70}")
        lines.append(f'  {title}')
        lines.append(f"  {'=' * 70}")

    lines.append(f"  {'Class':<16s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
    lines.append(f"  {'-' * 48}")
    for i in range(len(class_names)):
        lines.append(
            f"  {class_names[i]:<16s} {metrics['iou'][i]:>8.4f} "
            f"{metrics['precision'][i]:>8.4f} {metrics['recall'][i]:>8.4f} "
            f"{metrics['f1'][i]:>8.4f}")

    lines.append(f"  {'-' * 48}")
    lines.append(f"  {'mIoU':<16s} {metrics['miou']:>8.4f}")
    lines.append(f"  {'mPA':<16s} {metrics['mpa']:>8.4f}")
    lines.append(f"  {'OA':<16s} {metrics['oa']:>8.4f}")
    lines.append(f"  {'Kappa':<16s} {metrics['kappa']:>8.4f}")
    lines.append(f"  {'=' * 70}")
    return '\n'.join(lines)


@METRICS.register_module()
class WaterOursMetric(IoUMetric):
    """每次验证输出 ours 同一公式/格式的指标表 (替换 IoUMetric)"""

    def process(self, data_batch, data_samples):
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            # Evaluator 传入的已是 dict (BaseDataElement.to_dict()),
            # 与 mmseg IoUMetric 相同的字典式访问
            pred = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
            mask = label != self.ignore_index          # 行=GT, 列=预测
            cm = np.bincount(
                label[mask].astype(np.int64) * num_classes
                + pred[mask].astype(np.int64),
                minlength=num_classes ** 2).reshape(num_classes, num_classes)
            self.results.append(cm)

    def compute_metrics(self, results):
        class_names = list(self.dataset_meta['classes'])
        cm = np.sum(np.stack(results, axis=0), axis=0)
        metrics = _compute_all_metrics_from_cm(cm, len(class_names))
        table = _format_metrics_table(
            metrics, class_names, title='水草 4 类 · 验证 (与 ours 同一公式)')
        MMLogger.get_current_instance().info(table)
        return {
            'mIoU': float(metrics['miou']) * 100.0,
            'mPA': float(metrics['mpa']) * 100.0,
            'OA': float(metrics['oa']) * 100.0,
            'Kappa': float(metrics['kappa']) * 100.0,
        }


# =============================================================================
# 进度条: ours 风格 E{n}/100: xx%|...| iter/894
# =============================================================================

@HOOKS.register_module()
class TqdmEpochHook(Hook):
    """iter-based 训练按等效 epoch 切段的 tqdm 进度条 (stderr)"""

    priority = 'LOW'          # 在 LoggerHook 之后取最新 loss/acc

    def __init__(self, val_interval=894, mininterval=5.0):
        self.val_interval = val_interval
        self.mininterval = mininterval
        self._bar = None
        self._max_epoch = None

    # ---- tqdm 句柄 ----
    def _new_bar(self, runner):
        epoch = runner.iter // self.val_interval + 1
        if self._max_epoch is None:
            max_iters = runner.train_loop.max_iters
            self._max_epoch = (max_iters + self.val_interval - 1) // self.val_interval
        self._bar = tqdm(
            total=self.val_interval, dynamic_ncols=True,
            mininterval=self.mininterval, file=sys.stderr,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                       '[{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        self._bar.set_description(f'E{epoch}/{self._max_epoch}')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self._bar is None:
            if _HAS_TQDM:
                self._new_bar(runner)
            else:                      # 降级: 无 tqdm 时每等效 epoch 一行摘要
                return
        try:
            loss = runner.message_hub.get_scalar('train/loss').current()
            acc = runner.message_hub.get_scalar('train/decode.acc_seg').current()
            self._bar.set_postfix(
                loss=f'{loss:.4f}', acc=f'{acc:.4f}', refresh=False)
        except Exception:
            pass
        self._bar.update(1)
        if (runner.iter + 1) % self.val_interval == 0:
            self._bar.close()
            self._bar = None           # 下一 iter 自动开下个 epoch 的条

    def after_train(self, runner):
        if self._bar is not None:
            self._bar.close()
            self._bar = None
