# RDSKD-UTANet Code Package

This package contains the core implementation of **Reliability-aware Dual-Stage Knowledge Distillation UTANet (RDSKD-UTANet)** and a ready-to-run 5-fold training script. The README is written so that reviewers can quickly understand the file structure, dataset layout, and exact commands used for the reported experiments.

## 1. Package contents

```text
RDSKD_UTANet/
├── README.md
├── train_rdskd_utanet.py
├── utanet/
│   ├── UTANet.py
│   └── ta_mosc.py
├── scripts/
│   └── run_rdskd_utanet_5fold.sh
└── tools/
    ├── eval_synapse_case_hd95_rdskd.py
    └── summarize_rdskd_utanet_5fold.py
```

### File descriptions

- `train_rdskd_utanet.py`  
  Main script for RDSKD-UTANet training. It implements the second-stage task-adaptive student training, reliability-aware knowledge distillation, checkpoint loading, validation, inference-speed benchmarking, and result saving.

- `utanet/UTANet.py`  
  Core UTANet model definition.

- `utanet/ta_mosc.py`  
  Task-Adaptive Mixture of Skip Connections (TA-MoSC) module.

- `scripts/run_rdskd_utanet_5fold.sh`  
  Example shell script for reproducing the 5-fold RDSKD-UTANet experiments.

- `tools/eval_synapse_case_hd95_rdskd.py`  
  Optional case-level evaluation script for Synapse HD95.

- `tools/summarize_rdskd_utanet_5fold.py`  
  Optional summary script for collecting 5-fold results.

## 2. Environment

The experiments were run in a PyTorch environment with CUDA support. A typical environment is:

```bash
python >= 3.8
torch >= 2.0
numpy
Pillow
tqdm
```

A quick syntax check can be performed with:

```bash
python -m py_compile train_rdskd_utanet.py
python -m py_compile utanet/UTANet.py
python -m py_compile utanet/ta_mosc.py
```

## 3. Expected project placement

Copy or unzip this package into the UTANet project root:

```bash
unzip RDSKD_UTANet_Code_5fold.zip
```

To overwrite the corresponding project files, use:

```bash
cp RDSKD_UTANet_Code_5fold/train_rdskd_utanet.py ./train_rdskd_utanet.py
cp RDSKD_UTANet_Code_5fold/utanet/UTANet.py ./utanet/UTANet.py
cp RDSKD_UTANet_Code_5fold/utanet/ta_mosc.py ./utanet/ta_mosc.py
mkdir -p scripts tools
cp RDSKD_UTANet_Code_5fold/scripts/run_rdskd_utanet_5fold.sh ./scripts/run_rdskd_utanet_5fold.sh
cp RDSKD_UTANet_Code_5fold/tools/*.py ./tools/
```

## 4. Dataset layout

The 5-fold split root is assumed to be:

For GlaS and ISIC16, the expected layout is:

```text
original_5fold_splits/
└── <dataset>/
    ├── fold0/
    │   ├── train/images/
    │   ├── train/masks/
    │   ├── val/images/
    │   └── val/masks/
    ├── fold1/
    ├── fold2/
    ├── fold3/
    └── fold4/
```

For Synapse, the expected layout is:

```text
original_5fold_splits/
└── synapse/
    ├── fold0/
    │   ├── train_npz/
    │   └── val_npz/
    ├── fold1/
    ├── fold2/
    ├── fold3/
    └── fold4/
```

## 5. Required Stage-1 teacher checkpoints

RDSKD-UTANet uses the first-stage original-skip UTANet as the teacher and as the initialization checkpoint of the second-stage student. Before running RDSKD-UTANet, the Stage-1 checkpoints should exist.

If your Stage-1 folder names are different, modify `BASELINE_ROOT` or `stage1_name` in `scripts/run_rdskd_utanet_5fold.sh`.

## 6. Dataset-specific parameters

| Dataset | Task type | Num classes | TopK | Batch size | Normalize | Input size |
|---|---:|---:|---:|---:|---|---:|
| GlaS | binary | 1 | 3 | 4 | ImageNet | 224×224 |
| ISIC16 | binary | 1 | 3 | 16 | ImageNet | 224×224 |
| Synapse | multi-class | 9 | 4 | 8 | none | 224×224 |

Common training parameters:

| Parameter | Value |
|---|---:|
| epochs | 200 |
| optimizer | Adam, as implemented in `train_rdskd_utanet.py` |
| learning rate | 0.001 |
| weight decay | 1e-4 |
| early stopping patience | 50 |
| image size | 224 |
| mixed precision | enabled by `--amp` |
| augmentation | enabled by `--aug` |

RDSKD parameters:

| Parameter | Value |
|---|---:|
| knowledge distillation weight (`--kd_weight`) | 0.02 |
| distillation temperature (`--kd_temperature`) | 2.0 |
| KD warmup epochs | 0 |
| TA-MoSC auxiliary loss weight | 0.001 |
| confidence power | 1.0 |
| confidence threshold | 0.0 |
| class-balance power | 0.5 |
| reliability-aware KD | enabled |
| teacher-label agreement filtering | enabled |
| class-balanced weighting | enabled |
| background ignored for multi-class KD | enabled |

## 7. Quick run: all retained datasets and all folds

The final retained experiments can be launched by:

```bash
bash RDSKD_UTANet_Code_5fold/scripts/run_rdskd_utanet_5fold.sh
```

By default, the script runs:

```bash
DATASETS="glas isic16 synapse"
FOLDS="0 1 2 3 4"
```

## 8. Run a single dataset or a single fold

Run only GlaS:

```bash
DATASETS="glas" FOLDS="0 1 2 3 4" \
bash RDSKD_UTANet_Code_5fold/scripts/run_rdskd_utanet_5fold.sh
```

Run only ISIC16:

```bash
DATASETS="ISIC16" FOLDS="0 1 2 3 4" \
bash RDSKD_UTANet_Code_5fold/scripts/run_rdskd_utanet_5fold.sh
```

Run only Synapse:

```bash
DATASETS="Synapse" FOLDS="0 1 2 3 4" \
bash RDSKD_UTANet_Code_5fold/scripts/run_rdskd_utanet_5fold.sh
```

Run only ISIC16 fold0:

```bash
DATASETS="isic16" FOLDS="0" \
bash RDSKD_UTANet_Code_5fold/scripts/run_rdskd_utanet_5fold.sh
```

Run only Synapse fold0:

```bash
DATASETS="synapse" FOLDS="0" \
bash RDSKD_UTANet_Code_5fold/scripts/run_rdskd_utanet_5fold.sh
```


## 9. Output files

Each run saves results under:

```text
/root/autodl-fs/data/UTANet_01_storage/experiments_rdskd_utanet_5fold/<dataset>/<exp_name>/
```
If you have different file storage paths, you need to modify them yourself


Typical files include:

```text
best.pth
last.pth
args.json
summary.csv
training_log.csv
```

Synapse case-level HD95 evaluation, if enabled, is saved to:

```text
<exp_dir>/case_level_eval/summary.csv
```

The summary script writes collected 5-fold results to:

```text
/root/autodl-fs/data/UTANet_01_storage/paper_tables/rdskd_utanet_5fold/
```

## 11. Reproducibility notes

- The fold seed is set as `42 + fold` in the shell script.
- All folds use the same input size, optimizer settings, and early-stopping rule.
- The distillation constraint is used only during training. It does not add parameters, FLOPs, or inference-time branches to the final model.
- If a run has already produced `summary.csv`, the shell script skips it by default to avoid overwriting finished experiments.
