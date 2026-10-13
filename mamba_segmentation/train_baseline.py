"""
SMP 对照模型统一训练脚本 (论文对比实验用)

位置说明:
  本脚本位于对比实验工作区 (semantic_segmentation/), 通过 sys.path 只读引用
  ours 模型目录 (默认为本机 base1 路径, 服务器上用 --ours_dir 指定);
  ours 目录 (smp-trans-net sub base 1) 保持零改动。

目的:
  以与 ours (train_fusion.py + SelectiveFusionModel) 完全相同的实验条件训练
  SMP 标准分割架构 (UNet / UNet++ / DeepLabV3+ / MANet / PSPNet / FPN / Linknet / PAN,
  backbone 可换 ResNet34/50/101 等), 保证对比实验的唯一变量是"模型结构"。

与 train_fusion.py 的严格对齐 (单一事实来源, 本脚本不复制任何超参数值,
全部在运行时直接读取 train_fusion.CONFIG / 复用其函数):

  [数据]   完全复用 dataset_enhanced.FusionDataset 与 CONFIG 中的路径:
           同一 train/val 划分, 同一 6 波段输入 (R,G,B,NIR,VV,VH, rasterio [3,2,1,4,8,7]),
           同一训练集全局 z-score 统计量 (global_stats_enc1_rgbnirsar.npz, 推理时禁止重算),
           同一 5 模式数据增强 (set_epoch 逐轮重排, DataLoader shuffle=False 与 ours 一致),
           同一标签预处理 (label==4→255, 全零区置 255)
  [优化]   AdamW(lr=5e-5, weight_decay=1e-4) + LinearLR Warmup(15) + CosineAnnealing(100, 5e-6),
           batch_size=4, AMP 开启且损失计算在 autocast 外 fp32 进行,
           梯度 NaN 检测 + clip_grad_norm(1.0) + NaN batch 跳过/回滚 —— 数值全部来自 CONFIG
  [损失]   主损失 CombinedLoss (CE(label_smoothing=0.05) + DWA 加权 Dice/Focal)
           + boundary_loss_weight × BoundaryLoss, 与 ours 完全同一实现 (直接 import);
           ours 独有的辅助监督 (aux 0.1 / s1_aux 0.1) 依附于其模型结构的多头输出,
           对照模型无对应输出头, 天然不参与 —— 结构差异正是对比对象
  [评估]   逐 epoch 混淆矩阵 → compute_all_metrics_from_cm (与 ours 同一函数),
           早停 patience 同 CONFIG (基于 val mIoU), 保存 best/last 裸 state_dict (同格式)

用法 (服务器):
    python train_baseline.py --model unetpp_r50 \
        --ours_dir /data/users/gyy/project/smp-trans-net \
        --save_dir /data/checkpoints/gyy/baseline/unetpp_r50_run1

    # 自定义架构/骨干 (任意 SMP 支持的组合):
    python train_baseline.py --arch deeplabv3plus --encoder resnet101 --save_dir ...

    # 无外网环境 (首次运行 SMP 需联网下载 ImageNet 编码器权重, 离线时关闭):
    python train_baseline.py --model unet_r50 --encoder_weights none --save_dir ...

注意:
  - 对比实验建议每个模型多跑几次 (不同 --save_dir), 与 ours 的 single_500~509
    多次运行保持一致, 论文中报 mean±std
  - 编码器默认加载 ImageNet 预训练 (与 ours 使用 ConvNeXtV2-Base ImageNet 预训练对齐,
    控制预训练这一变量)
"""

import os
import sys

# =============================================================================
# 只读引用 ours 模型目录 (base1): 须在 import train_fusion 等模块之前加入 sys.path
# 服务器上路径不同时用 --ours_dir 指定 (支持 "--ours_dir X" 和 "--ours_dir=X")
# =============================================================================
def _ours_dir_from_argv(default):
    for i, a in enumerate(sys.argv):
        if a == '--ours_dir' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith('--ours_dir='):
            return a.split('=', 1)[1]
    return default

OURS_MODEL_DIR = _ours_dir_from_argv(
    r'C:\Users\hongtong\PyCharmMiscProject\smp-trans-net sub base 1')
if OURS_MODEL_DIR not in sys.path:
    sys.path.insert(0, OURS_MODEL_DIR)

import json
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 与 train_fusion.py 一致: 强制离线, 避免构建模型时联网
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from datetime import datetime

import segmentation_models_pytorch as smp

# ★ 复用 ours 的组件 (保证单一事实来源, 不复制粘贴任何实现)
from train_fusion import (CONFIG, CombinedLoss, calculate_accuracy,
                          compute_all_metrics_from_cm, print_metrics_table,
                          plot_cm_figure, save_ckpt)
from triple_model import BoundaryLoss
from dataset_enhanced import FusionDataset

NUM_CLASSES = 4
CLASS_NAMES_4 = ['Water', 'Submerged', 'Floating', 'Emergent']
IN_CHANNELS = 6  # R,G,B,NIR,VV,VH —— 与 ours 编码器1输入完全一致

# =============================================================================
# 对照模型注册表: 预设名 → (SMP 架构, 编码器骨干)
# =============================================================================
SMP_ARCHS = {
    'unet': smp.Unet,
    'unetplusplus': smp.UnetPlusPlus,
    'deeplabv3plus': smp.DeepLabV3Plus,
    'manet': smp.MAnet,
    'pspnet': smp.PSPNet,
    'fpn': smp.FPN,
    'linknet': smp.Linknet,
    'pan': smp.PAN,
}

BASELINE_MODELS = {
    'unet_r34':        ('unet', 'resnet34'),
    'unet_r50':        ('unet', 'resnet50'),
    'unet_r101':       ('unet', 'resnet101'),
    'unetpp_r50':      ('unetplusplus', 'resnet50'),
    'unetpp_r101':     ('unetplusplus', 'resnet101'),
    'deeplabv3p_r50':  ('deeplabv3plus', 'resnet50'),
    'deeplabv3p_r101': ('deeplabv3plus', 'resnet101'),
    'manet_r50':       ('manet', 'resnet50'),
    'pspnet_r50':      ('pspnet', 'resnet50'),
    'fpn_r50':         ('fpn', 'resnet50'),
    'linknet_r50':     ('linknet', 'resnet50'),
    'pan_r50':         ('pan', 'resnet50'),
}


def parse_args():
    parser = argparse.ArgumentParser(description="SMP 对照模型统一训练 (对比实验, 条件与 train_fusion.py 完全一致)")
    parser.add_argument('--save_dir', type=str, required=True,
                        help='输出目录 (权重/日志)')
    parser.add_argument('--ours_dir', type=str, default=OURS_MODEL_DIR,
                        help='ours 模型目录 (只读引用, 含 train_fusion.py/dataset_enhanced.py 等)')
    parser.add_argument('--model', type=str, default=None, choices=sorted(BASELINE_MODELS.keys()),
                        help='预设对照模型名 (见 BASELINE_MODELS)')
    parser.add_argument('--arch', type=str, default=None, choices=sorted(SMP_ARCHS.keys()),
                        help='自定义 SMP 架构 (与 --model 二选一, 与 --encoder 连用)')
    parser.add_argument('--encoder', type=str, default='resnet50',
                        help='自定义编码器骨干 (SMP/timm 支持的名称, 默认 resnet50)')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', choices=['imagenet', 'none'],
                        help='编码器预训练权重: imagenet=ImageNet (默认, 与 ours 对齐); none=随机初始化 (离线环境用)')
    # 数据路径覆盖 (默认取 train_fusion.CONFIG, 保证与 ours 同一数据)
    parser.add_argument('--train_img_dir', type=str, default=CONFIG['train_img_dir'])
    parser.add_argument('--train_lbl_dir', type=str, default=CONFIG['train_lbl_dir'])
    parser.add_argument('--val_img_dir', type=str, default=CONFIG['val_img_dir'])
    parser.add_argument('--val_lbl_dir', type=str, default=CONFIG['val_lbl_dir'])
    parser.add_argument('--stats_dir', type=str, default=CONFIG['stats_dir'])
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers (train_fusion.py 固定 8, 此处仅作本地适配, 不影响训练语义)')
    args = parser.parse_args()

    if args.model is None and args.arch is None:
        parser.error("必须指定 --model (预设) 或 --arch (自定义架构)")
    if args.model is not None and args.arch is not None:
        parser.error("--model 与 --arch 不可同时使用")
    return args


def build_baseline_model(arch, encoder, encoder_weights):
    """构建 SMP 对照模型: 输入 6ch (与 ours 编码器1一致), 输出 4 类"""
    weights = None if encoder_weights == 'none' else encoder_weights
    model = SMP_ARCHS[arch](
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )
    return model


def train_baseline(args):
    cfg = CONFIG  # ★ 全部超参直接读 ours 的 CONFIG, 保证零漂移
    arch, encoder = (BASELINE_MODELS[args.model] if args.model
                     else (args.arch, args.encoder))

    print("=" * 70)
    print(f"对照模型训练: {arch} + {encoder} (对比实验, 条件与 ours 完全一致)")
    print(f"  ours 目录 (只读): {args.ours_dir}")
    print("=" * 70)
    print(f"  epochs={cfg['epochs']}, lr={cfg['lr']}, batch_size={cfg['batch_size']}")
    print(f"  boundary_loss_weight={cfg['boundary_loss_weight']}")
    print(f"  early_stop_patience={cfg['early_stop_patience']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    checkpoint_dir = args.save_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(checkpoint_dir, "tb_logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"  输出目录: {checkpoint_dir}")

    # ======== 数据集: 与 ours 完全同一 FusionDataset / 路径 / 增强管线 ========
    print("\n加载数据集...")
    train_dataset = FusionDataset(
        args.train_img_dir, args.train_lbl_dir,
        stats_dir=args.stats_dir, mode='train', augment=True,
        need_s1=False,  # 对照模型与 ours 当前 CONFIG 一致: 单编码器 6ch 输入
    )
    val_dataset = FusionDataset(
        args.val_img_dir, args.val_lbl_dir,
        stats_dir=args.stats_dir, mode='val', augment=False,
        need_s1=False,
    )
    print(f"  训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    batch_size = cfg['batch_size']
    # shuffle=False 与 train_fusion.py 一致: 增强数据集内部用 set_epoch 逐轮重排
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ======== 模型 ========
    print(f"\n创建对照模型: {arch} + {encoder} "
          f"(in_channels={IN_CHANNELS}, classes={NUM_CLASSES}, "
          f"encoder_weights={args.encoder_weights})...")
    model = build_baseline_model(arch, encoder, args.encoder_weights).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  总参数: {total_params:.2f}M, 可训练: {trainable_params:.2f}M")

    # 记录运行配置 (论文可追溯性)
    run_config = {
        'purpose': 'baseline comparison, identical conditions to train_fusion.py',
        'arch': arch, 'encoder': encoder,
        'encoder_weights': args.encoder_weights,
        'in_channels': IN_CHANNELS, 'num_classes': NUM_CLASSES,
        'params_M': round(total_params, 3),
        'ours_dir': args.ours_dir,
        'train_img_dir': args.train_img_dir, 'val_img_dir': args.val_img_dir,
        'stats_dir': args.stats_dir,
        'hyperparams_source': 'train_fusion.CONFIG (unchanged)',
        'config': {k: (str(v) if isinstance(v, str) else v) for k, v in cfg.items()},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(checkpoint_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # ======== 优化器 / 调度器: 与 ours 完全一致 (读 CONFIG) ========
    all_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_params, lr=cfg['lr'], weight_decay=1e-4)

    from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
    epochs = cfg['epochs']
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg['warmup_epochs'])
    cosine = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg['min_lr'])
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg['warmup_epochs']])
    print(f"  学习率: Warmup({cfg['warmup_epochs']}) + CosineAnnealing({epochs})")

    # ======== 损失: 与 ours 完全同一实现 (import 复用) ========
    criterion = CombinedLoss(num_classes=NUM_CLASSES, ignore_index=255)
    boundary_loss_fn = BoundaryLoss(boundary_width=3, ignore_index=255)
    boundary_w = cfg['boundary_loss_weight']
    use_amp = cfg.get('use_amp', True)
    print(f"  损失: CombinedLoss(CE+DWA·Dice/Focal) + {boundary_w}×BoundaryLoss (同 ours)")
    print(f"  混合精度(AMP): {'开启' if use_amp else '关闭'}")

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_miou = 0.0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    patience = cfg['early_stop_patience']

    # 数值稳定性: recovery checkpoint 管理 (同 ours)
    recovery_ckpt_path = os.path.join(checkpoint_dir, "recovery_baseline.pth")
    recovery_state = None
    consecutive_nan_count = 0
    max_consecutive_nan = cfg.get('max_consecutive_nan', 20)
    checkpoint_interval = cfg.get('checkpoint_interval', 10)

    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        model.train()

        train_loss_total = 0.0
        train_loss_seg = 0.0
        train_loss_boundary = 0.0
        train_loss_ce = 0.0
        train_loss_dice = 0.0
        train_loss_focal = 0.0
        train_acc = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"E{epoch+1}/{epochs}")
        for batch in pbar:
            x_s2 = batch["image_s2"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            # GPU 端 NaN 检查 + clamp (同 ours)
            if torch.isnan(x_s2).any():
                del x_s2, label
                continue
            x_s2 = torch.clamp(x_s2, -5.0, 5.0)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x_s2)

            # ===== 损失计算在 autocast 外 (fp32), 与 ours 一致 =====
            logits_f = logits.float()
            loss_seg, dict_seg = criterion(logits_f, label)
            loss_boundary, boundary_ratio = boundary_loss_fn(logits_f, label)

            # 对照模型无辅助头 (ours 的 aux/s1_aux 依附其结构输出), 仅主损失 + 边界损失
            loss = loss_seg + boundary_w * loss_boundary

            if torch.isnan(loss) or torch.isinf(loss):
                consecutive_nan_count += 1
                print(f"  [WARN] NaN/Inf loss at step {train_batches} "
                      f"(consecutive={consecutive_nan_count}/{max_consecutive_nan})")

                if consecutive_nan_count >= max_consecutive_nan:
                    print(f"  [ERROR] 连续 {consecutive_nan_count} 次 NaN, 尝试回滚...")
                    if recovery_state is not None:
                        model.load_state_dict(recovery_state)
                        print(f"  [RECOVERY] 已回滚到上一个健康 checkpoint")
                        consecutive_nan_count = 0
                        for pg in optimizer.param_groups:
                            old_lr = pg['lr']
                            pg['lr'] = old_lr * 0.5
                            print(f"  [RECOVERY] LR {old_lr:.2e} -> {pg['lr']:.2e}")
                    else:
                        print(f"  [ERROR] 无可用 recovery checkpoint, 终止训练")
                        raise RuntimeError("训练发散: 连续 NaN 且无 recovery checkpoint 可用")

                del logits, loss, loss_seg, loss_boundary
                del x_s2, label
                optimizer.zero_grad(set_to_none=True)
                continue

            consecutive_nan_count = 0

            # ===== 反向传播 + 梯度裁剪 + 梯度 NaN 检测 (同 ours) =====
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_has_nan = False
            for p in all_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_has_nan = True
                    break

            if grad_has_nan:
                print(f"  [WARN] NaN/Inf in gradients at step {train_batches}, skipping optimizer step")
                scaler.update()
                del logits, loss, loss_seg, loss_boundary
                del x_s2, label
                continue

            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = calculate_accuracy(preds, label, ignore_index=255)
                train_loss_total += loss.item()
                train_loss_seg += loss_seg.item()
                bnd_val = loss_boundary.item() if isinstance(loss_boundary, torch.Tensor) else float(loss_boundary)
                train_loss_boundary += bnd_val
                train_loss_ce += dict_seg['ce']
                train_loss_dice += dict_seg['dice']
                train_loss_focal += dict_seg['focal']
                train_acc += acc

            train_batches += 1

            pbar.set_postfix({
                "loss": f"{train_loss_total / max(train_batches, 1):.4f}",
                "acc": f"{acc:.4f}",
            })

            del logits, loss, loss_seg, loss_boundary, preds
            del x_s2, label

            if train_batches % 100 == 0:
                torch.cuda.empty_cache()

        n = max(train_batches, 1)
        avg_train_loss = train_loss_total / n
        avg_train_seg = train_loss_seg / n
        avg_train_bnd = train_loss_boundary / n
        avg_train_ce = train_loss_ce / n
        avg_train_dice = train_loss_dice / n
        avg_train_focal = train_loss_focal / n
        avg_train_acc = train_acc / n

        criterion.dwa.update_epoch_losses(avg_train_dice, avg_train_focal, epoch=epoch)
        w_dice, w_focal = criterion.dwa.get_weights()

        torch.cuda.empty_cache()

        # ======== 验证 (与 ours 完全同一流程) ========
        model.eval()
        val_acc = 0.0
        all_preds, all_gt = [], []

        pbar_val = tqdm(val_loader, desc=f"E{epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for batch in pbar_val:
                x_s2 = batch["image_s2"].to(device, non_blocking=True)
                gt_label = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(x_s2)

                pred = torch.argmax(logits, dim=1)
                acc = calculate_accuracy(pred, gt_label, ignore_index=255)
                val_acc += acc
                pbar_val.set_postfix({"acc": f"{acc:.4f}"})

                all_preds.append(pred.cpu())
                all_gt.append(gt_label.cpu())
                del logits, pred, x_s2, gt_label

        avg_val_acc = val_acc / max(len(val_loader), 1)

        ap = torch.cat(all_preds, dim=0).numpy()
        ag = torch.cat(all_gt, dim=0).numpy()
        vm = ag != 255

        cm4 = confusion_matrix(ag[vm].flatten(), ap[vm].flatten(), labels=[0, 1, 2, 3])
        metrics = compute_all_metrics_from_cm(cm4, NUM_CLASSES)
        miou = metrics['miou']

        scheduler.step()

        # ======== 周期性 recovery checkpoint (同 ours) ========
        recovery_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        if (epoch + 1) % checkpoint_interval == 0:
            save_ckpt(model.state_dict(), "recovery_baseline.pth", checkpoint_dir)

        # ======== 打印 ========
        print(f"\nE{epoch+1}/{epochs} [{arch}+{encoder}]:")
        print(f"  Train Loss={avg_train_loss:.4f} (Seg={avg_train_seg:.4f}, Bnd={avg_train_bnd:.4f})")
        print(f"  Seg: CE={avg_train_ce:.4f}, Dice={avg_train_dice:.4f}, Focal={avg_train_focal:.4f}")
        print(f"  Val Acc={avg_val_acc:.4f}")
        print_metrics_table(metrics, CLASS_NAMES_4, title=f"E{epoch+1} 4类指标")
        print(f"  DWA: wD={w_dice:.4f}, wF={w_focal:.4f} | LR={scheduler.get_last_lr()[0]:.2e}")

        # ======== TensorBoard (与 ours 相同的 tag, 便于曲线同图对比) ========
        gs = epoch + 1
        writer.add_scalar('Loss/Train_Total', avg_train_loss, gs)
        writer.add_scalar('Loss/Train_Seg', avg_train_seg, gs)
        writer.add_scalar('Loss/Train_Boundary', avg_train_bnd, gs)
        writer.add_scalar('Loss/Train_CE', avg_train_ce, gs)
        writer.add_scalar('Loss/Train_Dice', avg_train_dice, gs)
        writer.add_scalar('Loss/Train_Focal', avg_train_focal, gs)
        writer.add_scalar('Accuracy/Val', avg_val_acc, gs)
        writer.add_scalar('Accuracy/Train', avg_train_acc, gs)

        for i, name in enumerate(CLASS_NAMES_4):
            writer.add_scalar(f'IoU/{name}', metrics['iou'][i], gs)
            writer.add_scalar(f'Precision/{name}', metrics['precision'][i], gs)
            writer.add_scalar(f'Recall/{name}', metrics['recall'][i], gs)
            writer.add_scalar(f'F1/{name}', metrics['f1'][i], gs)

        writer.add_scalar('Summary/mIoU', metrics['miou'], gs)
        writer.add_scalar('Summary/mPA', metrics['mpa'], gs)
        writer.add_scalar('Summary/OA', metrics['oa'], gs)
        writer.add_scalar('Summary/Kappa', metrics['kappa'], gs)

        writer.add_figure('CM/4class', plot_cm_figure(cm4, CLASS_NAMES_4, gs), gs, close=True)

        writer.add_scalar('Weights/Dice', w_dice, gs)
        writer.add_scalar('Weights/Focal', w_focal, gs)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], gs)

        # ======== 早停 (同 ours: 基于 val mIoU) ========
        if miou > best_miou:
            best_miou = miou
            best_metrics = metrics
            best_epoch = epoch + 1
            patience_counter = 0
            criterion.dwa.record_best()
            save_ckpt(model.state_dict(), "best_baseline.pth", checkpoint_dir)
            print(f"  ** 保存最佳模型: mIoU={miou:.4f}")
        else:
            patience_counter += 1
            print(f"  mIoU未提升 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  早停触发! 最佳mIoU: {best_miou:.4f}")
                break

    save_ckpt(model.state_dict(), "last_baseline.pth", checkpoint_dir)

    if best_metrics is not None:
        print_metrics_table(best_metrics, CLASS_NAMES_4,
                            title=f"[{arch}+{encoder}] 最优轮次 E{best_epoch} 4类指标")

    best_path = os.path.join(checkpoint_dir, "best_baseline.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"\n已加载最佳模型: {best_path}")

    writer.close()
    print("\n训练完成!")
    return best_miou


if __name__ == '__main__':
    train_baseline(parse_args())
