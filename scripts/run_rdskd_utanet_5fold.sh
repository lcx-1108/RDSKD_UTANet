#!/usr/bin/env bash
set -euo pipefail

# RDSKD-UTANet 5-fold training script.
# Run from the project root:
#   cd /root/UTANet_01/UTANet
#   bash RDSKD_UTANet_Code_5fold/scripts/run_rdskd_utanet_5fold.sh

cd /root/UTANet_01/UTANet

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

# Root paths. Modify these paths if your storage layout is different.
SPLIT_ROOT="${SPLIT_ROOT:-/root/autodl-fs/data/UTANet_01_storage/original_5fold_splits}"
BASELINE_ROOT="${BASELINE_ROOT:-/root/autodl-fs/data/UTANet_01_storage/experiments_original_utanet_5fold}"
EXP_ROOT="${EXP_ROOT:-/root/autodl-fs/data/UTANet_01_storage/experiments_rdskd_utanet_5fold}"
TABLE_DIR="${TABLE_DIR:-/root/autodl-fs/data/UTANet_01_storage/paper_tables/rdskd_utanet_5fold}"
LOG_DIR="${LOG_DIR:-/root/autodl-fs/data/UTANet_01_storage/logs/rdskd_utanet_5fold}"

# Datasets and folds. The paper's final retained comparison uses GlaS, ISIC16,
# and Synapse. MoNuSeg can still be run for completeness.
DATASETS="${DATASETS:-glas isic16 synapse}"
FOLDS="${FOLDS:-0 1 2 3 4}"

# Common training settings.
EPOCHS_RDSKD="${EPOCHS_RDSKD:-200}"
LR_RDSKD="${LR_RDSKD:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EARLY_STOP="${EARLY_STOP:-50}"
IMG_SIZE="${IMG_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# RDSKD settings used in the final experiments.
KD_WEIGHT="${KD_WEIGHT:-0.02}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
KD_WARMUP_EPOCHS="${KD_WARMUP_EPOCHS:-0}"
TAMOSC_AUX_WEIGHT="${TAMOSC_AUX_WEIGHT:-0.001}"
KD_CONF_POWER="${KD_CONF_POWER:-1.0}"
KD_CONF_MIN="${KD_CONF_MIN:-0.0}"
KD_CLASS_BALANCE_POWER="${KD_CLASS_BALANCE_POWER:-0.5}"

# Reliability-aware switches. Set to 0 to disable one component.
KD_RELIABLE="${KD_RELIABLE:-1}"
KD_AGREE_ONLY="${KD_AGREE_ONLY:-1}"
KD_CLASS_BALANCE="${KD_CLASS_BALANCE:-1}"
KD_IGNORE_BACKGROUND="${KD_IGNORE_BACKGROUND:-1}"

TAG="${TAG:-rdskd_kdw${KD_WEIGHT}_T${KD_TEMPERATURE}_agree_cb}"

mkdir -p "${EXP_ROOT}" "${TABLE_DIR}" "${LOG_DIR}"

set_dataset_cfg () {
  local dataset="$1"
  if [ "${dataset}" = "glas" ]; then
    NUM_CLASSES=1
    TOPK=3
    BATCH_SIZE=4
    NORMALIZE="imagenet"
  elif [ "${dataset}" = "isic16" ]; then
    NUM_CLASSES=1
    TOPK=3
    BATCH_SIZE=16
    NORMALIZE="imagenet"
  elif [ "${dataset}" = "monuseg" ]; then
    NUM_CLASSES=1
    TOPK=3
    BATCH_SIZE=4
    NORMALIZE="imagenet"
  elif [ "${dataset}" = "synapse" ]; then
    NUM_CLASSES=9
    TOPK=4
    BATCH_SIZE=8
    NORMALIZE="none"
  else
    echo "[ERROR] Unknown dataset: ${dataset}"
    exit 1
  fi
}

data_args_for_fold () {
  local dataset="$1"
  local fold="$2"
  if [ "${dataset}" = "synapse" ]; then
    DATA_ARGS=(
      --train_npz "${SPLIT_ROOT}/synapse/fold${fold}/train_npz"
      --val_npz   "${SPLIT_ROOT}/synapse/fold${fold}/val_npz"
    )
  else
    DATA_ARGS=(
      --train_images "${SPLIT_ROOT}/${dataset}/fold${fold}/train/images"
      --train_masks  "${SPLIT_ROOT}/${dataset}/fold${fold}/train/masks"
      --val_images   "${SPLIT_ROOT}/${dataset}/fold${fold}/val/images"
      --val_masks    "${SPLIT_ROOT}/${dataset}/fold${fold}/val/masks"
    )
  fi
}

make_rdskd_flags () {
  RDSKD_FLAGS=()
  if [ "${KD_RELIABLE}" = "1" ]; then RDSKD_FLAGS+=(--kd_reliable); fi
  if [ "${KD_AGREE_ONLY}" = "1" ]; then RDSKD_FLAGS+=(--kd_agree_only); fi
  if [ "${KD_CLASS_BALANCE}" = "1" ]; then RDSKD_FLAGS+=(--kd_class_balance); fi
  if [ "${KD_IGNORE_BACKGROUND}" = "1" ]; then RDSKD_FLAGS+=(--kd_ignore_background); fi
}

run_one () {
  local dataset="$1"
  local fold="$2"
  set_dataset_cfg "${dataset}"
  data_args_for_fold "${dataset}" "${fold}"
  make_rdskd_flags

  local seed=$((42 + fold))
  local stage1_name="${dataset}_fold${fold}_stage1_origin_skip"
  local teacher_ckpt="${BASELINE_ROOT}/${dataset}/${stage1_name}/best.pth"

  if [ ! -f "${teacher_ckpt}" ]; then
    echo "[ERROR] Missing Stage1 teacher checkpoint: ${teacher_ckpt}"
    echo "Please run the UTANet Stage1/original-skip 5-fold baseline first, or modify BASELINE_ROOT/stage1_name."
    exit 1
  fi

  local exp_name="${dataset}_fold${fold}_stage2_rdskd_topk${TOPK}_${TAG}"
  local exp_dir="${EXP_ROOT}/${dataset}/${exp_name}"
  local log_file="${LOG_DIR}/${dataset}_${exp_name}.log"

  if [ -f "${exp_dir}/summary.csv" ]; then
    echo "[SKIP] ${dataset} fold${fold}: ${exp_dir}/summary.csv exists."
    return 0
  fi

  echo "============================================================"
  echo "[RUN] RDSKD-UTANet | dataset=${dataset} | fold=${fold} | seed=${seed}"
  echo "num_classes=${NUM_CLASSES}, topk=${TOPK}, batch_size=${BATCH_SIZE}, normalize=${NORMALIZE}"
  echo "teacher/init checkpoint=${teacher_ckpt}"
  echo "kd_weight=${KD_WEIGHT}, T=${KD_TEMPERATURE}, reliable=${KD_RELIABLE}, agree_only=${KD_AGREE_ONLY}, class_balance=${KD_CLASS_BALANCE}, ignore_bg=${KD_IGNORE_BACKGROUND}"
  echo "output=${exp_dir}"
  echo "============================================================"

  "${PYTHON_BIN}" train_rdskd_utanet.py \
    --dataset "${dataset}" \
    "${DATA_ARGS[@]}" \
    --num_classes "${NUM_CLASSES}" \
    --topk "${TOPK}" \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LR_RDSKD}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS_RDSKD}" \
    --early_stop_patience "${EARLY_STOP}" \
    --seed "${seed}" \
    --aug \
    --normalize "${NORMALIZE}" \
    --amp \
    --require_tamosc \
    --train_only_tamosc \
    --teacher_ckpt "${teacher_ckpt}" \
    --init_ckpt "${teacher_ckpt}" \
    --kd_weight "${KD_WEIGHT}" \
    --kd_temperature "${KD_TEMPERATURE}" \
    --kd_warmup_epochs "${KD_WARMUP_EPOCHS}" \
    --kd_conf_power "${KD_CONF_POWER}" \
    --kd_conf_min "${KD_CONF_MIN}" \
    --kd_class_balance_power "${KD_CLASS_BALANCE_POWER}" \
    "${RDSKD_FLAGS[@]}" \
    --tamosc_aux_weight "${TAMOSC_AUX_WEIGHT}" \
    --exp_root "${EXP_ROOT}" \
    --exp_name "${exp_name}" 2>&1 | tee "${log_file}"

  if [ "${dataset}" = "synapse" ] && [ -f tools/eval_synapse_case_hd95_rdskd.py ]; then
    local eval_dir="${exp_dir}/case_level_eval"
    "${PYTHON_BIN}" tools/eval_synapse_case_hd95_rdskd.py \
      --ckpt "${exp_dir}/best.pth" \
      --val_npz "${SPLIT_ROOT}/synapse/fold${fold}/val_npz" \
      --out_dir "${eval_dir}" \
      --img_size "${IMG_SIZE}" \
      --num_classes 9 \
      --topk 4 \
      --batch_size 8 \
      --num_workers "${NUM_WORKERS}" \
      --normalize "${NORMALIZE}"
  fi
}

for dataset in ${DATASETS}; do
  for fold in ${FOLDS}; do
    run_one "${dataset}" "${fold}"
  done
done

if [ -f tools/summarize_rdskd_utanet_5fold.py ]; then
  "${PYTHON_BIN}" tools/summarize_rdskd_utanet_5fold.py \
    --datasets "${DATASETS}" \
    --folds "${FOLDS}" \
    --exp_root "${EXP_ROOT}" \
    --table_dir "${TABLE_DIR}" \
    --tag "${TAG}"
fi

echo "============================================================"
echo "RDSKD-UTANet 5-fold training finished."
echo "Experiment root: ${EXP_ROOT}"
echo "Log directory: ${LOG_DIR}"
echo "Summary directory: ${TABLE_DIR}"
echo "============================================================"
