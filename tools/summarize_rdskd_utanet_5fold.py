# -*- coding: utf-8 -*-
"""
tools/summarize_rdskd_utanet_5fold.py

Summarize RDSKD-UTANet 5-fold results.

Expected directory:
    {exp_root}/{dataset}/{dataset}_fold{fold}_stage2_rdskd_topk{topk}_{tag}/summary.csv

For Synapse, if case-level evaluation exists, this script uses:
    .../case_level_eval/summary.csv
for Dice/HD95 in the paper table.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_csv_one(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    return rows[0]


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fnum(x, default=np.nan):
    try:
        if x == "" or x is None:
            return default
        return float(x)
    except Exception:
        return default


def mean_std(vals):
    vals = [float(v) for v in vals if not np.isnan(float(v))]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def fmt_ms(vals, scale=1.0, digits=2):
    m, s = mean_std(vals)
    if np.isnan(m):
        return ""
    return f"{m*scale:.{digits}f}±{s*scale:.{digits}f}"


def dataset_topk(dataset: str) -> int:
    return 4 if dataset.lower() == "synapse" else 3


def main():
    p = argparse.ArgumentParser("Summarize RDSKD-UTANet 5-fold results")
    p.add_argument("--datasets", default="monuseg glas isic16 synapse")
    p.add_argument("--folds", default="0 1 2 3 4")
    p.add_argument("--exp_root", default="/root/autodl-fs/data/UTANet_01_storage/experiments_rdskd_utanet_5fold")
    p.add_argument("--table_dir", default="/root/autodl-fs/data/UTANet_01_storage/paper_tables/rdskd_utanet_5fold")
    p.add_argument("--tag", default="rdskd")
    args = p.parse_args()

    datasets = [x.strip().lower() for x in args.datasets.split() if x.strip()]
    folds = [int(x) for x in args.folds.split() if x.strip()]
    exp_root = Path(args.exp_root)
    table_dir = Path(args.table_dir)

    detail_rows = []
    summary_rows = []
    paper_rows = []

    for ds in datasets:
        topk = dataset_topk(ds)
        ds_rows = []
        for fold in folds:
            exp_name = f"{ds}_fold{fold}_stage2_rdskd_topk{topk}_{args.tag}"
            exp_dir = exp_root / ds / exp_name
            summary_path = exp_dir / "summary.csv"
            if not summary_path.exists():
                print("[Missing]", summary_path)
                continue
            r = read_csv_one(summary_path)
            row = dict(r)
            row["fold"] = fold
            row["dataset"] = ds
            row["exp_dir"] = str(exp_dir)
            if ds == "synapse":
                case_sum = exp_dir / "case_level_eval" / "summary.csv"
                if case_sum.exists():
                    cr = read_csv_one(case_sum)
                    row["case_dice"] = cr.get("dice", "")
                    row["case_hd95"] = cr.get("hd95", "")
                else:
                    row["case_dice"] = ""
                    row["case_hd95"] = ""
            detail_rows.append(row)
            ds_rows.append(row)

        if not ds_rows:
            continue

        params = [fnum(r.get("params_M")) for r in ds_rows]
        flops = [fnum(r.get("flops_G")) for r in ds_rows]
        fps = [fnum(r.get("fps")) for r in ds_rows]

        if ds == "synapse":
            dice = [fnum(r.get("case_dice", r.get("best_dice"))) for r in ds_rows]
            hd95 = [fnum(r.get("case_hd95")) for r in ds_rows]
            summary_rows.append({
                "dataset": ds,
                "dice_mean": mean_std(dice)[0],
                "dice_std": mean_std(dice)[1],
                "hd95_mean": mean_std(hd95)[0],
                "hd95_std": mean_std(hd95)[1],
                "params_M": mean_std(params)[0],
                "flops_G": mean_std(flops)[0],
                "fps_mean": mean_std(fps)[0],
                "fps_std": mean_std(fps)[1],
            })
            paper_rows.append({
                "Dataset": "Synapse",
                "Metric": "Dice / HD95",
                "Dice(%)": fmt_ms(dice, scale=100.0),
                "IoU(%)": "",
                "HD95": fmt_ms(hd95, scale=1.0),
                "Params(M)": f"{mean_std(params)[0]:.2f}",
                "FLOPs(G)": f"{mean_std(flops)[0]:.2f}",
                "FPS": fmt_ms(fps, scale=1.0),
            })
        else:
            dice = [fnum(r.get("best_dice", r.get("val_dice"))) for r in ds_rows]
            iou = [fnum(r.get("val_iou")) for r in ds_rows]
            name_map = {"glas": "GlaS", "isic16": "ISIC16", "monuseg": "MoNuSeg"}
            summary_rows.append({
                "dataset": ds,
                "dice_mean": mean_std(dice)[0],
                "dice_std": mean_std(dice)[1],
                "iou_mean": mean_std(iou)[0],
                "iou_std": mean_std(iou)[1],
                "params_M": mean_std(params)[0],
                "flops_G": mean_std(flops)[0],
                "fps_mean": mean_std(fps)[0],
                "fps_std": mean_std(fps)[1],
            })
            paper_rows.append({
                "Dataset": name_map.get(ds, ds),
                "Metric": "Dice / IoU",
                "Dice(%)": fmt_ms(dice, scale=100.0),
                "IoU(%)": fmt_ms(iou, scale=100.0),
                "HD95": "",
                "Params(M)": f"{mean_std(params)[0]:.2f}",
                "FLOPs(G)": f"{mean_std(flops)[0]:.2f}",
                "FPS": fmt_ms(fps, scale=1.0),
            })

    write_csv(table_dir / "rdskd_utanet_5fold_detail.csv", detail_rows)
    write_csv(table_dir / "rdskd_utanet_5fold_summary.csv", summary_rows)
    write_csv(table_dir / "rdskd_utanet_5fold_paper_table.csv", paper_rows)
    print("[Saved]", table_dir / "rdskd_utanet_5fold_detail.csv")
    print("[Saved]", table_dir / "rdskd_utanet_5fold_summary.csv")
    print("[Saved]", table_dir / "rdskd_utanet_5fold_paper_table.csv")


if __name__ == "__main__":
    main()
