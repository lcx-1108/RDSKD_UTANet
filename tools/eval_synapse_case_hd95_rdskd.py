# -*- coding: utf-8 -*-
"""
tools/eval_synapse_case_hd95_rdskd.py

Case-level Synapse evaluation for RDSKD-UTANet checkpoints.
The RDSKD student is the original UTANet(pretrained=True), so no custom model
file is needed at inference time.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utanet"))

try:
    from UTANet import UTANet
except Exception:
    from utanet.UTANet import UTANet


class SynapseNPZDataset(Dataset):
    def __init__(self, npz_dir: str, img_size: int = 224, normalize: str = "none"):
        self.root = Path(npz_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"npz_dir not found: {self.root}")
        self.files = sorted(self.root.rglob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No npz files found in {self.root}")
        self.img_size = int(img_size)
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with np.load(p, allow_pickle=True) as data:
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
                raise KeyError(f"{p} missing image/label keys, found keys={list(data.keys())}")

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
        return img_t, lab_t.long(), p.stem


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


def load_ckpt(model, path: str):
    ckpt = torch.load(path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = {strip_prefix(k): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[Load] {path}")
    print(f"[Load] missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print("[Load] missing sample:", msg.missing_keys[:20])
    if msg.unexpected_keys:
        print("[Load] unexpected sample:", msg.unexpected_keys[:20])


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


def parse_case_id(stem: str) -> str:
    s = str(stem)
    m = re.search(r"(case\d+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"(img\d+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    s = re.sub(r"[_-]?slice[_-]?\d+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_-]?z[_-]?\d+$", "", s, flags=re.IGNORECASE)
    return s


def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter + eps) / (pred.sum() + gt.sum() + eps))


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return np.array([0.0], dtype=np.float32)
    if pred.sum() == 0 or gt.sum() == 0:
        diag = np.sqrt(np.sum(np.array(pred.shape, dtype=np.float32) ** 2))
        return np.array([diag], dtype=np.float32)
    footprint = np.ones((3,) * pred.ndim, dtype=bool)
    pred_surface = np.logical_xor(pred, binary_erosion(pred, structure=footprint, border_value=0))
    gt_surface = np.logical_xor(gt, binary_erosion(gt, structure=footprint, border_value=0))
    if pred_surface.sum() == 0:
        pred_surface = pred
    if gt_surface.sum() == 0:
        gt_surface = gt
    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)
    return np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]]).astype(np.float32)


def hd95_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.percentile(surface_distances(pred, gt), 95))


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser("RDSKD-UTANet Synapse case-level Dice/HD95 evaluation")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--normalize", default="none", choices=["none", "imagenet"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SynapseNPZDataset(args.val_npz, img_size=args.img_size, normalize=args.normalize)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = UTANet(pretrained=True, topk=args.topk, n_channels=3, n_classes=args.num_classes, img_size=args.img_size)
    load_ckpt(model, args.ckpt)
    model = model.to(device)
    model.eval()

    cases: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
    with torch.no_grad():
        for images, labels, names in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out = unwrap_output(model(images))
            if out.shape[-2:] != labels.shape[-2:]:
                out = F.interpolate(out, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.argmax(out, dim=1).detach().cpu().numpy().astype(np.int16)
            gt = labels.detach().cpu().numpy().astype(np.int16)
            for i, name in enumerate(names):
                cid = parse_case_id(str(name))
                cases.setdefault(cid, []).append((str(name), pred[i], gt[i]))

    rows = []
    for cid, items in sorted(cases.items()):
        items = sorted(items, key=lambda x: x[0])
        pred_vol = np.stack([x[1] for x in items], axis=0)
        gt_vol = np.stack([x[2] for x in items], axis=0)
        for c in range(1, int(args.num_classes)):
            pred_c = pred_vol == c
            gt_c = gt_vol == c
            rows.append({
                "case": cid,
                "class": c,
                "dice": dice_binary(pred_c, gt_c),
                "hd95": hd95_binary(pred_c, gt_c),
                "pred_voxels": int(pred_c.sum()),
                "gt_voxels": int(gt_c.sum()),
                "slices": int(pred_vol.shape[0]),
            })

    write_csv(out_dir / "synapse_case_class_dice_hd95.csv", rows)
    dice_vals = [float(r["dice"]) for r in rows]
    hd_vals = [float(r["hd95"]) for r in rows]
    summary = [{
        "dice": float(np.mean(dice_vals)) if dice_vals else 0.0,
        "hd95": float(np.mean(hd_vals)) if hd_vals else 0.0,
        "num_cases": len(cases),
        "num_case_class_rows": len(rows),
        "ckpt": str(args.ckpt),
        "val_npz": str(args.val_npz),
    }]
    write_csv(out_dir / "summary.csv", summary)
    print("[Saved]", out_dir / "summary.csv")
    print("[Saved]", out_dir / "synapse_case_class_dice_hd95.csv")
    print(summary[0])


if __name__ == "__main__":
    main()
