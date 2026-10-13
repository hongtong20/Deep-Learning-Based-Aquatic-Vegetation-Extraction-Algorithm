"""
6 通道输入版 MambaVision 骨干 (水草对比实验, mmseg 插件)

与官方 tools/mamba_vision.MM_mamba_vision 的两点差异 (其余逐行一致):

  1. stem 通道扩展 (3ch → 6ch):
     官方 MM 包装不透传 in_chans, 恒为 3 通道输入。本类在加载 ImageNet-1K
     3 通道预训练权重后, 将 stem (patch_embed.conv_down[0], Conv2d) 扩展为
     in_chans 输入:
       - 前 3 通道 (R,G,B) 直接继承预训练权重
       - 其余通道 (NIR,VV,VH) 用 RGB 权重均值初始化
     与 ours 模型 (ConvNeXtV2 6ch 编码器) 的预训练适配策略一致。

  2. drop_path_rate 生效:
     官方 MM 包装未把 drop_path_rate 转发给 MambaVision.__init__
     (被 **kwargs 吞掉, 实际恒为默认 0.2, 官方 config 中的 0.3/0.7/0.4
     被静默忽略)。本类显式透传, 使配置值生效。
"""

import importlib.util
import os
import sys

import torch          # _expand_stem 的 torch.no_grad 需要 (仅 import torch.nn 不绑定 torch 名)
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

# =============================================================================
# timm>=1.0 兼容 shim (必须在下方 _load_official() 之前执行):
# 官方 tools/mamba_vision.py 使用旧 API 路径 timm.models.registry /
# timm.models.layers (timm>=1.0 已移除)。集群环境 timm==1.0.27 与
# segmentation_models_pytorch 共用、不允许降级, 故在此注册等价别名模块;
# tools/mamba_vision.py 本身保持零改动。
# =============================================================================
import types

import timm.models as _timm_models

if not hasattr(_timm_models, 'registry'):
    _reg = types.ModuleType('timm.models.registry')
    try:
        from timm.models._registry import register_model   # timm>=1.0 (1.0.27 实测)
    except ImportError:
        try:
            from timm.registry import register_model       # 部分旧版顶层路径
        except ImportError:
            from timm.models import register_model         # 兜底: 包级导出
    _reg.register_model = register_model
    _reg.__spec__ = importlib.util.spec_from_loader('timm.models.registry', None)
    sys.modules['timm.models.registry'] = _reg
    _timm_models.registry = _reg

if not hasattr(_timm_models, 'layers'):
    _lay = types.ModuleType('timm.models.layers')
    from timm.layers import DropPath, trunc_normal_
    _lay.DropPath = DropPath
    _lay.trunc_normal_ = trunc_normal_
    _lay.__spec__ = importlib.util.spec_from_loader('timm.models.layers', None)
    sys.modules['timm.models.layers'] = _lay
    _timm_models.layers = _lay

# 新版 timm 可能把 resolve_pretrained_cfg 移到 timm.models._pretrained
import timm.models._builder as _timm_builder
if not hasattr(_timm_builder, 'resolve_pretrained_cfg'):
    from timm.models._pretrained import resolve_pretrained_cfg
    _timm_builder.resolve_pretrained_cfg = resolve_pretrained_cfg


# -----------------------------------------------------------------------------
# mamba_ssm.__init__ 会连带 import 其 LM 语言模型头 (MambaLMHeadModel →
# transformers.generation; 我们只用 selective_scan_fn, 完全用不到)。集群环境
# transformers 5.18 与共用的 huggingface-hub 1.16.4 不匹配且不许动版本。
# 真实 transformers 可导入则不干预; 否则注册最小 stub, 环境零改动。
# -----------------------------------------------------------------------------
def _stub_transformers_if_broken():
    if 'transformers' in sys.modules:
        return
    try:
        import transformers.generation  # noqa: F401
        return
    except Exception:
        pass

    class _Dummy:
        pass

    def _getattr(name):
        if name == '__version__':
            return '0.0.0-stub'
        if name == 'Adafactor':
            # mmengine.optim.builder 会 try: from transformers import Adafactor:
            # 拒绝 → ImportError → mmengine 优雅跳过; 放行 Dummy 则会注册撞名
            # (torch>=2.12 的 torch.optim 新增了 Adafactor) → KeyError
            raise AttributeError(name)
        if name.startswith('__'):
            raise AttributeError(name)
        return _Dummy

    tf = types.ModuleType('transformers')
    tf.__path__ = []                      # 使其可被当作包
    tf.__spec__ = importlib.util.spec_from_loader('transformers', None)
    tf.__dict__['__getattr__'] = _getattr
    gen = types.ModuleType('transformers.generation')
    gen.__spec__ = importlib.util.spec_from_loader('transformers.generation', None)
    gen.__dict__['__getattr__'] = _getattr
    tf.generation = gen
    sys.modules['transformers'] = tf
    sys.modules['transformers.generation'] = gen


_stub_transformers_if_broken()


# -----------------------------------------------------------------------------
# mmcv-lite 没有 CUDA 编译的 mmcv._ext, 但 mmseg.utils 会无条件 import
# mmcv.ops (如 point_sample, 仅 import 引用)。本实验不执行任何 mmcv CUDA
# 算子 (MambaVision 的 CUDA 核在 mamba-ssm/causal-conv1d 内), 注册一个
# 按需抛错的 dummy _ext: 让 mmcv.ops.* 全部可导入, 若真被误调用会在调用处
# 立即报 RuntimeError, 而不是在 import 阶段崩溃。
# -----------------------------------------------------------------------------
def _stub_mmcv_ext_if_missing():
    try:
        import mmcv._ext  # noqa: F401  (完整版 mmcv 存在则不干预)
        return
    except ImportError:
        pass
    import mmcv as _mmcv

    def _ext_getattr(name):
        def _unavailable(*args, **kwargs):
            raise RuntimeError(
                f'mmcv._ext 为 dummy stub: 算子 {name} 不可用 (mmcv-lite 无 '
                'CUDA 算子; 本实验不应调用任何 mmcv 算子)')
        if name.startswith('__'):
            raise AttributeError(name)
        return _unavailable

    _ext = types.ModuleType('mmcv._ext')
    # __spec__ 不可省: mmengine 的 mmcv_full_available() 会
    # pkgutil.find_loader('mmcv._ext') — 手工造的模块无 __spec__ 直接
    # ValueError (job 703)。给 loader=None 的 spec, find_loader 返回 None
    # → 判定 "mmcv-full 不可用" → revert_sync_batchnorm 等走纯 torch 分支
    _ext.__spec__ = importlib.util.spec_from_loader('mmcv._ext', None)
    _ext.__dict__['__getattr__'] = _ext_getattr
    sys.modules['mmcv._ext'] = _ext
    _mmcv._ext = _ext


_stub_mmcv_ext_if_missing()

# -----------------------------------------------------------------------------
# mmdet stub: tools/mamba_vision.py:37 无条件 `from mmdet.registry import MODELS`,
# 但 MODELS_MMDET 仅作装饰器 (@MODELS_MMDET.register_module()) 使用。
# 本机 (Windows) 推理未装 mmdet 时注册 no-op 替身, 免去为一条 import 装 mmdet
# 全家 (pycocotools 等); 集群/已装环境不干预。
# -----------------------------------------------------------------------------
def _stub_mmdet_if_missing():
    try:
        import mmdet.registry  # noqa: F401
        return
    except Exception:
        pass

    class _StubRegistry:
        @staticmethod
        def register_module(name=None, force=False, module=None):
            def deco(obj):
                return obj
            return deco if module is None else module

    mmdet = types.ModuleType('mmdet')
    mmdet.__spec__ = importlib.util.spec_from_loader('mmdet', None)
    mmdet.__path__ = []
    registry = types.ModuleType('mmdet.registry')
    registry.__spec__ = importlib.util.spec_from_loader('mmdet.registry', None)
    registry.MODELS = _StubRegistry
    mmdet.registry = registry
    sys.modules['mmdet'] = mmdet
    sys.modules['mmdet.registry'] = registry


_stub_mmdet_if_missing()

# -----------------------------------------------------------------------------
# mamba_ssm CPU 回退: mamba_ssm 官方仅支持 Linux+CUDA (selective_scan_fn 是编译
# 的 CUDA 核, tools/mamba_vision.py 的 forward 亦无纯 torch 分支)。Windows/CPU
# 本机推理时注册 mamba_ssm.ops.selective_scan_interface 替身:
#
#   _selective_scan_ref 与 mamba_ssm 官方 selective_scan_ref 同一数学定义
#   (S6 选择性扫描递推):
#       delta' = softplus(delta + delta_bias)
#       h_l = exp(delta'_l A) ⊙ h_{l-1} + delta'_l (B_l ⊗ u_l),  h_{-1} = 0
#       y_l = C_l · h_l + D ⊙ u_l
#   其中并行前缀扫描用 Hillis–Steele (log2 L 轮张量运算, 无逐步 Python 循环;
#   L=16384 token 时逐循环不可用)。A 为负实数、delta'>0 → 系数 exp(ΔA)∈(0,1),
#   乘积单调衰减, 无除法, 数值稳定。
#
# 仅推理用途 (torch.no_grad); 集群上真包存在 → 本 stub 完全不生效, 训练零影响。
# -----------------------------------------------------------------------------
def _prefix_scan(a, b):
    """对第 2 维 (L) 做前缀递推: h_l = a_l·h_{l-1} + b_l (h_{-1}=0)

    Hillis–Steele 包含式并行扫描: 第 k 轮后位置 l 持有窗口
    [max(0, l-2^k+1), l] 的组合结果, ceil(log2 L) 轮后即为完整前缀。
    a, b: (B, D, L, N) → 返回 h: (B, D, L, N)
    """
    L = a.shape[2]
    stride = 1
    while stride < L:
        a_new = a.clone()
        b_new = b.clone()
        a_new[:, :, stride:] = a[:, :, stride:] * a[:, :, :-stride]
        b_new[:, :, stride:] = (a[:, :, stride:] * b[:, :, :-stride]
                                + b[:, :, stride:])
        a, b = a_new, b_new
        stride *= 2
    return b


def _selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                        delta_bias=None, delta_softplus=False,
                        return_last_state=None):
    """selective_scan_fn 的纯 torch 等价实现 (fp32; 与官方 selective_scan_ref
    同一定义)。MambaVision 调用点: z=None, delta_softplus=True。

    u: (b, d, L); delta: (b, d, L); A: (d, n); B/C: (b, n, L); D: (d,);
    delta_bias: (d,)。返回 y: (b, d, L)。
    """
    if z is not None:
        raise NotImplementedError('CPU 回退不支持 z 参数 (MambaVision 未使用)')
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().reshape(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    A = A.float()
    B = B.float()
    C = C.float()

    a = torch.exp(delta.unsqueeze(-1) * A.reshape(1, -1, 1, A.shape[-1]))
    bu = (delta * u).unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1)
    h = _prefix_scan(a, bu)
    y = torch.einsum('bdln,bnl->bdl', h, C)
    if D is not None:
        y = y + D.float().reshape(1, -1, 1) * u
    return y


def _stub_mamba_ssm_if_missing():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa
        return
    except Exception:
        pass
    mamba_ssm = types.ModuleType('mamba_ssm')
    mamba_ssm.__spec__ = importlib.util.spec_from_loader('mamba_ssm', None)
    mamba_ssm.__path__ = []
    ops = types.ModuleType('mamba_ssm.ops')
    ops.__spec__ = importlib.util.spec_from_loader('mamba_ssm.ops', None)
    ops.__path__ = []
    iface = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    iface.__spec__ = importlib.util.spec_from_loader(
        'mamba_ssm.ops.selective_scan_interface', None)
    iface.selective_scan_fn = _selective_scan_ref
    mamba_ssm.ops = ops
    ops.selective_scan_interface = iface
    sys.modules['mamba_ssm'] = mamba_ssm
    sys.modules['mamba_ssm.ops'] = ops
    sys.modules['mamba_ssm.ops.selective_scan_interface'] = iface


_stub_mamba_ssm_if_missing()

# -----------------------------------------------------------------------------
# torch>=2.6 起 torch.load 默认 weights_only=True: 官方 MambaVision ImageNet
# ckpt 除权重外还存了 argparse.Namespace (训练参数 args), 不在默认安全白名单
# → mmengine load_checkpoint 拒载 (UnpicklingError)。已枚举该 ckpt pickle 的
# 全部 global: Namespace 是唯一需放行者 (其余 OrderedDict / torch.Storage /
# _rebuild_tensor_v2 均在 torch 默认白名单)。按 torch 报错信息的官方建议用
# add_safe_globals 精确放行 Namespace 一个类, 其余安全策略不变;
# T/S/B 三份 ckpt 由同一保存代码产出, 同样适用。
# -----------------------------------------------------------------------------
import argparse as _argparse

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([_argparse.Namespace])

# =============================================================================
# 只读加载官方骨干实现 (tools/mamba_vision.py, 保持零改动)
# 优先按包导入 (repo 根在 sys.path 时); 失败则按文件路径加载,
# 避免与环境中其他同名 tools 包冲突。
# =============================================================================
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_OFFICIAL_SRC = os.path.join(_REPO_ROOT, 'tools', 'mamba_vision.py')


def _load_official():
    try:
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from tools import mamba_vision  # noqa: F401
        return mamba_vision
    except Exception:
        pass
    spec = importlib.util.spec_from_file_location(
        'mamba_vision_official', _OFFICIAL_SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules['mamba_vision_official'] = module
    spec.loader.exec_module(module)
    return module


_official = _load_official()
MM_mamba_vision = _official.MM_mamba_vision
MambaVision = _official.MambaVision
LayerNorm2d = _official.LayerNorm2d


@MODELS.register_module()
class MM_mamba_vision_6ch(MM_mamba_vision):
    """MambaVision 骨干 + 6 通道 stem (见模块 docstring)"""

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 out_indices=(0, 1, 2, 3),
                 pretrained=None,
                 norm_layer='ln2d',
                 layer_scale=None,
                 drop_path_rate=0.2,
                 in_chans=6,
                 **kwargs):
        # 与官方 MM_mamba_vision 唯一不同: 显式透传 drop_path_rate
        MambaVision.__init__(
            self,
            dim=dim,
            in_dim=in_dim,
            depths=depths,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            layer_scale=layer_scale,
        )
        # ---- 以下与 MM_mamba_vision.__init__ 逐行一致 ----
        self.dims = [int(dim * 2 ** i) for i in range(0, 4)]
        self.channel_first = True
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _norm = _NORMLAYERS.get(str(norm_layer).lower(), LayerNorm2d)

        self.out_indices = out_indices
        for i in out_indices:
            self.add_module(f'outnorm{i}', _norm(self.dims[i]))

        del self.norm
        del self.head

        # 先加载 ImageNet 3ch 预训练 (strict=False, 同官方), 再做 stem 扩展
        self.init_weights(pretrained)
        self._expand_stem(in_chans)

    def _expand_stem(self, in_chans):
        """把 patch_embed.conv_down[0] 由 3ch 卷积扩展为 in_chans 卷积

        前 3 通道继承原 (预训练) 权重, 其余通道用其均值初始化。
        """
        old = self.patch_embed.conv_down[0]
        n_keep = min(3, old.in_channels, in_chans)
        if old.in_channels == in_chans:
            return
        new = nn.Conv2d(in_chans, old.out_channels, 3, 2, 1, bias=False)
        with torch.no_grad():
            new.weight[:, :n_keep] = old.weight.data[:, :n_keep]
            if in_chans > n_keep:
                extra = old.weight.data[:, :n_keep].mean(
                    dim=1, keepdim=True).repeat(1, in_chans - n_keep, 1, 1)
                new.weight[:, n_keep:] = extra
        self.patch_embed.conv_down[0] = new
