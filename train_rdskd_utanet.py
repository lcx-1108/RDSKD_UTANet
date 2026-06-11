# -*- coding: utf-8 -*-
"""
train_rrdskd_utanet.py

Dual-Stage Knowledge Distillation for UTANet (RRDSKD-UTANet).

This script does NOT change the UTANet architecture. It improves the original
UTANet two-stage training procedure:

    Stage1 teacher: origin-skip UTANet (pretrained=False), already trained.
    Stage2 student: TA-MoSC UTANet (pretrained=True), initialized from Stage1.

During Stage2, the student is trained with:
    L = L_seg + lambda_aux * L_TA-MoSC + lambda_kd * L_KD(student, teacher)

Recommended first run:
    --train_only_tamosc --teacher_ckpt <stage1/best.pth> --init_ckpt <stage1/best.pth>

Inference uses the normal UTANet student only; teacher is training-time only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utanet"))

try:
    from UTANet import UTANet  # original uploaded/project model
except Exception:
    from utanet.UTANet import UTANet


# -----------------------------------------------------------------------------
# Reproducibility / IO
# -----------------------------------------------------------------------------

def seed_everything(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    keys = list(row.keys())
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv_single(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_id(path: Path, is_mask: bool) -> str:
    stem = path.stem
    if is_mask:
        patterns = [
            r"_Segmentation$", r"_segmentation$",
            r"_mask$", r"_Mask$", r"_MASK$",
            r"_label$", r"_Label$", r"_LABEL$",
            r"_gt$", r"_GT$",
            r"_anno$", r"_annotation$", r"_Annotation$",
        ]
        for pat in patterns:
            stem = re.sub(pat, "", stem)
    return stem.replace(" ", "").lower()


def find_pairs(images_dir: str, masks_dir: str) -> List[Tuple[Path, Path, str]]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks_dir not found: {masks_dir}")

    imgs = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    msks = [p for p in masks_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

    img_map = {normalize_id(p, False): p for p in imgs}
    mask_map = {normalize_id(p, True): p for p in msks}
    ids = sorted(set(img_map) & set(mask_map))
    pairs = [(img_map[i], mask_map[i], i) for i in ids]

    if len(pairs) == 0:
        missing_masks = sorted(set(img_map) - set(mask_map))[:10]
        missing_images = sorted(set(mask_map) - set(img_map))[:10]
        raise RuntimeError(
            f"No matched pairs found: {images_dir} / {masks_dir}; "
            f"missing_masks sample={missing_masks}; missing_images sample={missing_images}"
        )
    return pairs


class BinarySegDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path, str]], img_size: int = 224, aug: bool = False, normalize: str = "imagenet"):
        self.pairs = pairs
        self.img_size = int(img_size)
        self.aug = bool(aug)
        self.normalize = normalize

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            k = random.choice([0, 1, 2, 3])
            if k:
                img = img.rotate(90 * k, resample=Image.BILINEAR)
                mask = mask.rotate(90 * k, resample=Image.NEAREST)
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path, name = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        if self.aug:
            img, mask = self._augment(img, mask)

        img_np = np.asarray(img).astype(np.float32) / 255.0
        mask_np = (np.asarray(mask).astype(np.float32) > 127).astype(np.float32)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        if self.normalize == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_t = (img_t - mean) / std
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        return img_t, mask_t, str(name)


class SynapseNPZDataset(Dataset):
    def __init__(self, npz_dir: str, img_size: int = 224, aug: bool = False, normalize: str = "none"):
        self.root = Path(npz_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"npz_dir not found: {self.root}")
        self.files = sorted(self.root.rglob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No npz files found in {self.root}")
        self.img_size = int(img_size)
        self.aug = bool(aug)
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with np.load(p, allow_pickle=True) as data:
            keys = list(data.keys())
            image = None
            label = None
            for k in ["image", "img", "data", "x"]:
                if k in data:
                    image = data[k]
                    break
            for k in ["label", "mask", "seg", "y"]:
                if k in data:
                    label = data[k]
                    break
            if image is None or label is None:
                raise KeyError(f"{p} missing image/label keys, found keys={keys}")

        image = np.asarray(image).astype(np.float32)
        label = np.asarray(label).astype(np.int64)

        if image.ndim == 2:
            lo, hi = float(image.min()), float(image.max())
            if hi > lo:
                image = (image - lo) / (hi - lo)
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] >= 3:
            image = image[..., :3]
            if image.max() > 2:
                image = image / 255.0
        else:
            raise ValueError(f"Unexpected image shape {image.shape} in {p}")

        img_t = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        lab_t = torch.from_numpy(label.astype(np.int64))
        if img_t.shape[-2:] != (self.img_size, self.img_size):
            img_t = F.interpolate(img_t.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
            lab_t = F.interpolate(lab_t.float().unsqueeze(0).unsqueeze(0), size=(self.img_size, self.img_size), mode="nearest").squeeze(0).squeeze(0).long()
        if self.aug:
            if random.random() < 0.5:
                img_t = torch.flip(img_t, dims=[2])
                lab_t = torch.flip(lab_t, dims=[1])
            if random.random() < 0.5:
                img_t = torch.flip(img_t, dims=[1])
                lab_t = torch.flip(lab_t, dims=[0])
        return img_t, lab_t.long(), p.stem


def dataset_defaults(dataset: str) -> Dict:
    dataset = dataset.lower()
    if dataset == "glas":
        return {"num_classes": 1, "topk": 3}
    if dataset == "isic16":
        return {"num_classes": 1, "topk": 3}
    if dataset == "monuseg":
        return {"num_classes": 1, "topk": 3}
    if dataset == "synapse":
        return {"num_classes": 9, "topk": 4}
    raise ValueError(dataset)


def build_loaders(args):
    dft = dataset_defaults(args.dataset)
    if args.num_classes is None:
        args.num_classes = int(dft["num_classes"])
    if args.topk is None:
        args.topk = int(dft["topk"])

    if args.dataset in ["glas", "isic16", "monuseg"]:
        train_pairs = find_pairs(args.train_images, args.train_masks)
        val_pairs = find_pairs(args.val_images, args.val_masks)
        train_set = BinarySegDataset(train_pairs, args.img_size, args.aug, args.normalize)
        val_set = BinarySegDataset(val_pairs, args.img_size, False, args.normalize)
    elif args.dataset == "synapse":
        train_set = SynapseNPZDataset(args.train_npz, args.img_size, args.aug, args.normalize)
        val_set = SynapseNPZDataset(args.val_npz, args.img_size, False, args.normalize)
    else:
        raise ValueError(args.dataset)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, len(train_set), len(val_set)


# -----------------------------------------------------------------------------
# Model / checkpoint helpers
# -----------------------------------------------------------------------------

def unwrap_output(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v) and v.ndim == 4:
                return v
    if isinstance(out, dict):
        for k in ["out", "pred", "prediction", "logits"]:
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v
        for v in out.values():
            if torch.is_tensor(v) and v.ndim == 4:
                return v
    raise ValueError(f"Cannot unwrap output from {type(out)}")


def extract_aux_loss(out, device=None):
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v) and v.ndim <= 1:
                return v.mean()
    if isinstance(out, dict):
        for k in ["aux_loss", "tamosc_aux_loss", "moe_loss"]:
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v.mean()
    if device is None:
        try:
            device = unwrap_output(out).device
        except Exception:
            device = torch.device("cpu")
    return torch.tensor(0.0, device=device)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def strip_prefix(k: str) -> str:
    for p in ["module.", "model.", "base."]:
        if k.startswith(p):
            k = k[len(p):]
    return k


def load_matching_checkpoint(model: nn.Module, path: str) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    src = extract_state_dict(ckpt)
    src = {strip_prefix(k): v for k, v in src.items()}
    dst = model.state_dict()
    matched, skipped = {}, []
    for k, v in src.items():
        if k in dst and tuple(dst[k].shape) == tuple(v.shape):
            matched[k] = v
        else:
            skipped.append(k)
    msg = model.load_state_dict(matched, strict=False)
    print(f"[Load matching] {path}")
    print(f"  loaded={len(matched)}, skipped={len(skipped)}, missing_after_load={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    if skipped:
        print("  skipped sample:", skipped[:20])
    return {"loaded": len(matched), "skipped": len(skipped), "missing": len(msg.missing_keys), "unexpected": len(msg.unexpected_keys)}


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, best_dice, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "best_dice": float(best_dice),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args) if hasattr(args, "__dict__") else {},
    }, path)


def set_train_only_tamosc(model: nn.Module) -> Dict:
    for p in model.parameters():
        p.requires_grad = False
    train_prefixes = ("fuse", "moe", "docker1", "docker2", "docker3", "docker4")
    enabled = []
    for name, p in model.named_parameters():
        if name.startswith(train_prefixes):
            p.requires_grad = True
            enabled.append(name)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[RDSKD Stage2] train only TA-MoSC modules")
    print(f"  trainable params: {trainable:,} / {total:,}")
    print(f"  enabled param sample: {enabled[:20]}")
    if trainable == 0:
        raise RuntimeError("No trainable TA-MoSC parameters found. Did you forget --pretrained?")
    return {"trainable_params_after_freeze": trainable, "total_params_after_freeze": total}


def keep_frozen_bn_eval(model: nn.Module):
    """Keep BN layers belonging to frozen modules in eval mode after model.train()."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            params = list(m.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                m.eval()


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total / 1e6


def _clear_thop_hooks(model: nn.Module):
    """
    Remove stale THOP hooks/buffers from a model.

    THOP profiles a model by registering forward hooks and temporary buffers
    such as total_ops / total_params. If profiling is accidentally run on the
    real training model and the buffers are removed while hooks remain, training
    can crash with:
        AttributeError: 'Conv2d' object has no attribute 'total_ops'
    This helper is defensive and safe to call before normal training.
    """
    for m in model.modules():
        # Remove THOP-created buffers/attributes when present.
        for attr in ["total_ops", "total_params"]:
            if hasattr(m, attr):
                try:
                    delattr(m, attr)
                except Exception:
                    pass

        # Remove THOP forward hooks. THOP hooks live in the thop package;
        # keeping other user-defined hooks intact is safer, but here the
        # training models do not rely on forward hooks.
        for hook_dict_name in ["_forward_hooks", "_forward_pre_hooks", "_backward_hooks"]:
            hook_dict = getattr(m, hook_dict_name, None)
            if hook_dict:
                for k, h in list(hook_dict.items()):
                    mod = getattr(h, "__module__", "")
                    if "thop" in str(mod).lower():
                        hook_dict.pop(k, None)


def compute_flops_optional(model: nn.Module, img_size: int, device: torch.device):
    """
    Compute FLOPs/MACs with THOP on a deepcopy of the model.

    Important:
    THOP registers forward hooks. If profiling is performed on the real
    training model, stale hooks may remain and crash training with:
        AttributeError: 'Conv2d' object has no attribute 'total_ops'.
    Therefore we profile a temporary copied model and delete it immediately.
    """
    try:
        import copy
        import gc
        from thop import profile

        was_training = model.training
        _clear_thop_hooks(model)

        flops_model = copy.deepcopy(model)
        flops_model = flops_model.to(device)
        flops_model.eval()

        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        with torch.no_grad():
            flops, params = profile(flops_model, inputs=(dummy,), verbose=False)

        del flops_model
        del dummy
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        _clear_thop_hooks(model)
        model.train(was_training)
        return float(flops), float(flops) / 1e9
    except Exception as e:
        print(f"[FLOPs warning] thop/profile failed: {e}")
        try:
            _clear_thop_hooks(model)
            model.train()
        except Exception:
            pass
        return float("nan"), float("nan")


# -----------------------------------------------------------------------------
# Losses / metrics
# -----------------------------------------------------------------------------

def dice_loss_binary(prob, target, eps=1e-6):
    if target.ndim == 3:
        target = target.unsqueeze(1)
    prob = prob.float().clamp(1e-6, 1 - 1e-6)
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (den + eps)).mean()


def dice_loss_multiclass(logits, target, num_classes: int, eps=1e-6):
    prob = torch.softmax(logits, dim=1)
    target = target.long()
    losses = []
    for c in range(1, int(num_classes)):
        pc = prob[:, c]
        tc = (target == c).float()
        inter = (pc * tc).sum(dim=(1, 2))
        den = pc.sum(dim=(1, 2)) + tc.sum(dim=(1, 2))
        losses.append(1.0 - ((2.0 * inter + eps) / (den + eps)).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def supervised_loss(pred, target, num_classes: int):
    if int(num_classes) == 1:
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()
        prob = pred.float().clamp(1e-6, 1 - 1e-6)
        # binary_cross_entropy(prob, target) is unsafe inside AMP autocast.
        # UTANet returns sigmoid probabilities for binary tasks, so keep Dice on prob
        # and compute BCE in fp32 with autocast disabled.
        with torch.cuda.amp.autocast(enabled=False):
            bce = F.binary_cross_entropy(prob.float(), target.float())
        dloss = dice_loss_binary(prob, target)
        return bce + dloss
    ce = F.cross_entropy(pred, target.long())
    dloss = dice_loss_multiclass(pred, target.long(), num_classes)
    return ce + dloss


def _logit(p, eps=1e-6):
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def _weighted_mean(loss_map: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6):
    if weight is None:
        return loss_map.mean()
    weight = weight.float().detach()
    while weight.ndim < loss_map.ndim:
        weight = weight.unsqueeze(1)
    denom = weight.sum().clamp_min(eps)
    if float(denom.detach().cpu()) <= eps:
        return loss_map.sum() * 0.0
    return (loss_map.float() * weight).sum() / denom


def _class_balance_weight(target: torch.Tensor, num_classes: int, ignore_background: bool = True, power: float = 0.5, eps: float = 1e-6):
    """Return BxHxW class weights based on target frequency in the current batch."""
    target = target.long()
    flat = target.reshape(-1)
    counts = torch.bincount(flat, minlength=int(num_classes)).float().to(target.device)
    valid = counts > 0
    if ignore_background and counts.numel() > 0:
        valid[0] = False
    cw = torch.ones_like(counts)
    if valid.any():
        cw[valid] = 1.0 / torch.pow(counts[valid].clamp_min(1.0), float(power))
        # Normalize valid class weights to mean 1, preventing uncontrolled loss scale changes.
        cw[valid] = cw[valid] / cw[valid].mean().clamp_min(eps)
    if ignore_background and counts.numel() > 0:
        cw[0] = 0.0
    return cw[target]


def kd_loss_fn(student_pred, teacher_pred, target, num_classes: int, args):
    """
    Reliability-aware KD.

    If --kd_reliable is disabled, this reduces to ordinary DSKD.
    If enabled, KD is weighted by teacher confidence and optionally by
    teacher-GT agreement. For Synapse/multiclass, optional class balancing
    emphasizes small organs and ignores background by default.
    """
    T = float(args.kd_temperature)
    reliable = bool(getattr(args, "kd_reliable", False))
    conf_power = float(getattr(args, "kd_conf_power", 1.0))
    conf_min = float(getattr(args, "kd_conf_min", 0.0))
    agree_only = bool(getattr(args, "kd_agree_only", False))

    if int(num_classes) == 1:
        # Original UTANet returns sigmoid probabilities for binary tasks.
        if target.ndim == 3:
            target_b = target.unsqueeze(1)
        else:
            target_b = target
        target_b = target_b.float()

        sp0 = student_pred.float().clamp(1e-6, 1 - 1e-6)
        tp0 = teacher_pred.detach().float().clamp(1e-6, 1 - 1e-6)

        weight = None
        if reliable:
            conf = (tp0 - 0.5).abs().mul(2.0).clamp(0.0, 1.0)
            weight = torch.pow(conf, conf_power)
            if conf_min > 0:
                weight = weight * (conf >= conf_min).float()
            if agree_only:
                agree = ((tp0 > 0.5) == (target_b > 0.5)).float()
                weight = weight * agree
            if bool(getattr(args, "kd_foreground_boost", False)):
                # Optional foreground emphasis for binary tasks; default off.
                fg_boost = float(getattr(args, "kd_foreground_boost_factor", 1.0))
                weight = weight * (1.0 + (fg_boost - 1.0) * target_b.float())

        sp = sp0
        tp = tp0
        if abs(T - 1.0) > 1e-8:
            sp = torch.sigmoid(_logit(sp0) / T)
            tp = torch.sigmoid(_logit(tp0) / T)

        # BCE on probabilities is unsafe under AMP autocast; compute it in fp32.
        with torch.cuda.amp.autocast(enabled=False):
            loss_map = F.binary_cross_entropy(sp.float(), tp.float(), reduction="none")
            kd_bce = _weighted_mean(loss_map, weight)
        return kd_bce * (T ** 2)

    # Multi-class UTANet returns logits.
    target_m = target.long()
    with torch.no_grad():
        pt0 = F.softmax(teacher_pred.detach().float(), dim=1)
        conf, teacher_cls = torch.max(pt0, dim=1)
        weight = None
        if reliable:
            weight = torch.pow(conf.clamp(0.0, 1.0), conf_power)
            if conf_min > 0:
                weight = weight * (conf >= conf_min).float()
            if agree_only:
                weight = weight * (teacher_cls == target_m).float()
            if bool(getattr(args, "kd_ignore_background", True)):
                weight = weight * (target_m != 0).float()
            if bool(getattr(args, "kd_class_balance", False)):
                cb = _class_balance_weight(
                    target_m,
                    int(num_classes),
                    ignore_background=bool(getattr(args, "kd_ignore_background", True)),
                    power=float(getattr(args, "kd_class_balance_power", 0.5)),
                )
                weight = weight * cb

    log_p_s = F.log_softmax(student_pred.float() / T, dim=1)
    p_t = F.softmax(teacher_pred.detach().float() / T, dim=1)
    # Per-pixel KL: BxHxW
    kl_map = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1)
    kd = _weighted_mean(kl_map, weight)
    return kd * (T ** 2)


@torch.no_grad()
def metrics(pred, target, num_classes: int, eps=1e-6):
    if int(num_classes) == 1:
        if target.ndim == 3:
            target = target.unsqueeze(1)
        pred_bin = (pred > 0.5).float()
        target = target.float()
        inter = (pred_bin * target).sum(dim=(1, 2, 3))
        den = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
        dice = ((2.0 * inter + eps) / (den + eps)).mean().item()
        iou = ((inter + eps) / (union + eps)).mean().item()
        return dice, iou

    pred_cls = torch.argmax(pred, dim=1)
    target = target.long()
    dices, ious = [], []
    for c in range(1, int(num_classes)):
        pc = (pred_cls == c).float()
        tc = (target == c).float()
        inter = (pc * tc).sum(dim=(1, 2))
        den = pc.sum(dim=(1, 2)) + tc.sum(dim=(1, 2))
        union = pc.sum(dim=(1, 2)) + tc.sum(dim=(1, 2)) - inter
        dices.append(((2.0 * inter + eps) / (den + eps)).mean())
        ious.append(((inter + eps) / (union + eps)).mean())
    if not dices:
        return 0.0, 0.0
    return torch.stack(dices).mean().item(), torch.stack(ious).mean().item()


def kd_weight_at_epoch(args, epoch: int) -> float:
    if args.kd_warmup_epochs <= 0:
        return float(args.kd_weight)
    return float(args.kd_weight) * min(1.0, float(epoch) / float(args.kd_warmup_epochs))


def train_one_epoch(student, teacher, loader, optimizer, device, args, scaler, epoch):
    student.train()
    if args.train_only_tamosc:
        keep_frozen_bn_eval(student)
    teacher.eval()

    sums = {"loss": 0.0, "seg_loss": 0.0, "kd_loss": 0.0, "aux_loss": 0.0, "dice": 0.0, "iou": 0.0, "n": 0}
    kd_w = kd_weight_at_epoch(args, epoch)

    pbar = tqdm(loader, desc=f"Train e{epoch}", ncols=110)
    for images, masks, _names in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_out = teacher(images)
            teacher_pred = unwrap_output(teacher_out)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            student_out = student(images)
            pred = unwrap_output(student_out)
            aux = extract_aux_loss(student_out, device=device)
            seg = supervised_loss(pred, masks, args.num_classes)
            kd = kd_loss_fn(pred, teacher_pred, masks, args.num_classes, args)
            loss = seg + kd_w * kd + float(args.tamosc_aux_weight) * aux

        scaler.scale(loss).backward()
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            d, i = metrics(pred.detach(), masks, args.num_classes)
        bs = int(images.size(0))
        sums["loss"] += float(loss.detach().item()) * bs
        sums["seg_loss"] += float(seg.detach().item()) * bs
        sums["kd_loss"] += float(kd.detach().item()) * bs
        sums["aux_loss"] += float(aux.detach().item()) * bs
        sums["dice"] += float(d) * bs
        sums["iou"] += float(i) * bs
        sums["n"] += bs
        pbar.set_postfix(loss=sums["loss"] / max(1, sums["n"]), dice=sums["dice"] / max(1, sums["n"]), kd=kd_w)

    n = max(1, sums.pop("n"))
    return {k: v / n for k, v in sums.items()} | {"kd_weight_eff": kd_w}


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    sums = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "n": 0}
    pbar = tqdm(loader, desc="Val", ncols=110)
    for images, masks, _names in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        out = model(images)
        pred = unwrap_output(out)
        loss = supervised_loss(pred, masks, args.num_classes)
        d, i = metrics(pred, masks, args.num_classes)
        bs = int(images.size(0))
        sums["loss"] += float(loss.item()) * bs
        sums["dice"] += float(d) * bs
        sums["iou"] += float(i) * bs
        sums["n"] += bs
        pbar.set_postfix(dice=sums["dice"] / max(1, sums["n"]), iou=sums["iou"] / max(1, sums["n"]))
    n = max(1, sums.pop("n"))
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def benchmark_inference(model, img_size: int, device: torch.device, warmup: int = 20, repeat: int = 100):
    model.eval()
    x = torch.randn(1, 3, int(img_size), int(img_size), device=device)
    for _ in range(max(0, int(warmup))):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max(1, int(repeat))):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms = dt / max(1, int(repeat)) * 1000.0
    fps = 1000.0 / ms if ms > 0 else 0.0
    return ms, fps


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("RRDSKD-UTANet Stage2 training")
    p.add_argument("--dataset", required=True, choices=["glas", "isic16", "monuseg", "synapse"])

    # Data paths.
    p.add_argument("--train_images", default="")
    p.add_argument("--train_masks", default="")
    p.add_argument("--val_images", default="")
    p.add_argument("--val_masks", default="")
    p.add_argument("--train_npz", default="")
    p.add_argument("--val_npz", default="")

    # Model.
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--require_tamosc", action="store_true")

    # Checkpoints.
    p.add_argument("--teacher_ckpt", required=True, help="Stage1 origin-skip teacher checkpoint.")
    p.add_argument("--init_ckpt", required=True, help="Stage1 checkpoint used to initialize Stage2 student.")

    # Training.
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aug", action="store_true")
    p.add_argument("--normalize", default="imagenet", choices=["imagenet", "none"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--early_stop_patience", type=int, default=50)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--train_only_tamosc", action="store_true")

    # RDSKD loss.
    p.add_argument("--kd_weight", type=float, default=0.05)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--kd_warmup_epochs", type=int, default=0)
    p.add_argument("--tamosc_aux_weight", type=float, default=0.001)
    p.add_argument("--kd_reliable", action="store_true",
                   help="Enable reliability-aware KD weighting. If disabled, use ordinary DSKD.")
    p.add_argument("--kd_conf_power", type=float, default=1.0,
                   help="Power applied to teacher confidence weight.")
    p.add_argument("--kd_conf_min", type=float, default=0.0,
                   help="Ignore KD pixels/voxels with teacher confidence below this threshold.")
    p.add_argument("--kd_agree_only", action="store_true",
                   help="Use KD only where teacher hard prediction agrees with GT.")
    p.add_argument("--kd_class_balance", action="store_true",
                   help="For multiclass KD, apply inverse-sqrt target class-frequency weights.")
    p.add_argument("--kd_class_balance_power", type=float, default=0.5,
                   help="Class-balance power; 0.5 means inverse sqrt frequency.")
    p.add_argument("--kd_ignore_background", action="store_true",
                   help="For multiclass KD, ignore background pixels in KD. Recommended for Synapse.")
    p.add_argument("--kd_foreground_boost", action="store_true",
                   help="For binary KD, optionally boost foreground pixels.")
    p.add_argument("--kd_foreground_boost_factor", type=float, default=1.0)

    # Output.
    p.add_argument("--exp_root", default="/root/autodl-fs/data/UTANet_01_storage/experiments_rdskd_utanet_5fold")
    p.add_argument("--exp_name", required=True)
    p.add_argument("--benchmark_warmup", type=int, default=20)
    p.add_argument("--benchmark_repeat", type=int, default=100)
    args = p.parse_args()

    dft = dataset_defaults(args.dataset)
    if args.num_classes is None:
        args.num_classes = int(dft["num_classes"])
    if args.topk is None:
        args.topk = int(dft["topk"])
    if args.require_tamosc and args.topk is None:
        raise RuntimeError("Internal error: topk not resolved.")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.exp_root) / args.dataset / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_json(exp_dir / "args.json", vars(args))

    train_loader, val_loader, n_train, n_val = build_loaders(args)

    print("=" * 90)
    print(f"dataset={args.dataset}, model=RRDSKD-UTANet")
    print(f"train={n_train}, val/test={n_val}, device={device}")
    print(f"teacher_ckpt={args.teacher_ckpt}")
    print(f"student_init_ckpt={args.init_ckpt}")
    print(f"kd_weight={args.kd_weight}, T={args.kd_temperature}, kd_warmup={args.kd_warmup_epochs}")
    print(f"reliable={args.kd_reliable}, agree_only={args.kd_agree_only}, conf_power={args.kd_conf_power}, conf_min={args.kd_conf_min}, class_balance={args.kd_class_balance}, ignore_bg={args.kd_ignore_background}")
    print(f"exp_dir={exp_dir}")
    print("=" * 90)

    # Teacher: Stage1 origin-skip UTANet.
    teacher = UTANet(pretrained=False, topk=args.topk, n_channels=3, n_classes=args.num_classes, img_size=args.img_size)
    teacher_info = load_matching_checkpoint(teacher, args.teacher_ckpt)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(device)
    teacher.eval()

    # Student: Stage2 TA-MoSC UTANet initialized from Stage1.
    student = UTANet(pretrained=True, topk=args.topk, n_channels=3, n_classes=args.num_classes, img_size=args.img_size)
    init_info = load_matching_checkpoint(student, args.init_ckpt)
    freeze_info = {}
    if args.train_only_tamosc:
        freeze_info = set_train_only_tamosc(student)
    student = student.to(device)

    total_params, trainable_params, params_m = count_params(student)
    flops_raw, flops_g = compute_flops_optional(student, args.img_size, device)

    trainable_parameters = [p for p in student.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters. Check --train_only_tamosc / model flags.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_dice = -1.0
    best_row = {}
    no_improve = 0
    best_ckpt = exp_dir / "best.pth"
    last_ckpt = exp_dir / "last.pth"
    history_path = exp_dir / "history.csv"
    t0 = time.perf_counter()

    # Defensive cleanup: make sure no THOP hooks remain on the real training models.
    _clear_thop_hooks(student)
    _clear_thop_hooks(teacher)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}] | RDSKD kd_weight_eff={kd_weight_at_epoch(args, epoch):.6f}, T={args.kd_temperature}")
        tr = train_one_epoch(student, teacher, train_loader, optimizer, device, args, scaler, epoch)
        va = validate(student, val_loader, device, args)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "dataset": args.dataset,
            "model": "rdskd_utanet",
            "seed": args.seed,
            "train_loss": tr.get("loss", -1),
            "train_seg_loss": tr.get("seg_loss", -1),
            "train_kd_loss": tr.get("kd_loss", -1),
            "train_aux_loss": tr.get("aux_loss", -1),
            "kd_weight_eff": tr.get("kd_weight_eff", args.kd_weight),
            "train_dice": tr.get("dice", -1),
            "train_iou": tr.get("iou", -1),
            "val_loss": va.get("loss", -1),
            "val_dice": va.get("dice", -1),
            "val_iou": va.get("iou", -1),
            "params_total": total_params,
            "params_trainable": trainable_params,
            "params_M": params_m,
            "flops_raw": flops_raw,
            "flops_G": flops_g,
            "kd_weight": args.kd_weight,
            "kd_temperature": args.kd_temperature,
            "kd_warmup_epochs": args.kd_warmup_epochs,
            "kd_reliable": int(args.kd_reliable),
            "kd_agree_only": int(args.kd_agree_only),
            "kd_conf_power": args.kd_conf_power,
            "kd_conf_min": args.kd_conf_min,
            "kd_class_balance": int(args.kd_class_balance),
            "kd_ignore_background": int(args.kd_ignore_background),
            "tamosc_aux_weight": args.tamosc_aux_weight,
        }
        append_csv(history_path, row)
        save_checkpoint(last_ckpt, student, optimizer, scheduler, scaler, epoch, best_dice, args)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Dice {row['train_dice']:.4f} IoU {row['train_iou']:.4f} "
            f"Loss {row['train_loss']:.4f} KD {row['train_kd_loss']:.4f} | "
            f"Val Dice {row['val_dice']:.4f} IoU {row['val_iou']:.4f}"
        )

        improved = float(row["val_dice"]) > (best_dice + float(args.early_stop_min_delta))
        if improved:
            best_dice = float(row["val_dice"])
            best_row = row
            no_improve = 0
            save_checkpoint(best_ckpt, student, optimizer, scheduler, scaler, epoch, best_dice, args)
            print("[Saved best]", best_ckpt)
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[Early stopping] no improvement for {no_improve} epochs. Best Dice={best_dice:.6f}")
            break

    train_time_sec = time.perf_counter() - t0
    (exp_dir / "time.txt").write_text(f"train_time_sec={train_time_sec}\ntrain_time_min={train_time_sec/60}\n", encoding="utf-8")

    infer_ms, fps = benchmark_inference(student, args.img_size, device, warmup=args.benchmark_warmup, repeat=args.benchmark_repeat)

    summary = {
        **best_row,
        "best_dice": best_dice,
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
        "exp_dir": str(exp_dir),
        "train_size": n_train,
        "val_size": n_val,
        "train_time_sec": train_time_sec,
        "train_time_min": train_time_sec / 60.0,
        "infer_ms_per_image": infer_ms,
        "fps": fps,
        "teacher_ckpt": args.teacher_ckpt,
        "init_ckpt": args.init_ckpt,
        "teacher_loaded_tensors": teacher_info.get("loaded", 0),
        "init_loaded_tensors": init_info.get("loaded", 0),
        "init_skipped_tensors": init_info.get("skipped", 0),
        "num_classes": args.num_classes,
        "topk": args.topk,
        "params_M": params_m,
        "flops_G": flops_g,
        "train_only_tamosc": args.train_only_tamosc,
        "trainable_params_after_freeze": freeze_info.get("trainable_params_after_freeze", ""),
        "total_params_after_freeze": freeze_info.get("total_params_after_freeze", ""),
        "kd_weight": args.kd_weight,
        "kd_temperature": args.kd_temperature,
        "kd_warmup_epochs": args.kd_warmup_epochs,
        "tamosc_aux_weight": args.tamosc_aux_weight,
    }
    write_csv_single(exp_dir / "summary.csv", summary)
    write_json(exp_dir / "summary.json", summary)

    print("=" * 90)
    print("Training finished.")
    print("Best Dice:", best_dice)
    print("Best checkpoint:", best_ckpt)
    print("Summary CSV:", exp_dir / "summary.csv")
    print(f"Inference: {infer_ms:.3f} ms/image | FPS {fps:.2f}")
    print(f"Train time: {train_time_sec/60:.2f} min")
    print("=" * 90)


if __name__ == "__main__":
    main()
