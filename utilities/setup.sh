#!/bin/bash
# End-to-end pipeline for the autoPET V interactive experiment:
#   1. Convert DeepPSMA and merge into main dataset
#   2. Generate 3-strategy scribble heatmaps for all cases
#   3. nnU-Net preprocessing + ResEnc-M plans
#   4. Train a single fold (configurable GPU and fold)
set -euo pipefail

# ==============================================================
# Configuration
# ==============================================================
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${REPO_ROOT}/data/PSMA-FDG-PET-CT-Lesions_v2"
DEEPPSMA_DIR="${REPO_ROOT}/data"

NNUNET_RAW="${REPO_ROOT}/nnUNet_raw"
NNUNET_PREPROCESSED="${REPO_ROOT}/nnUNet_preprocessed"
NNUNET_RESULTS="${REPO_ROOT}/nnUNet_results"

DATASET_ID=998
DATASET_NAME="Dataset998_AutoPETV"
PLANS_NAME="nnUNetResEncUNetMPlans_40G"
GPU_MEMORY_TARGET=40
TRAINER_NAME="nnUNetTrainerAutoPETV"

# Default: train fold 0 on GPU 0.
# Override via environment: TRAIN_FOLD=1 TRAIN_GPU=1 bash setup_and_train.sh train
TRAIN_FOLD="${TRAIN_FOLD:-0}"
TRAIN_GPU="${TRAIN_GPU:-0}"

HEATMAP_WORKERS=32

export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

# Find the nnunetv2 trainer directory for copying custom trainer
NNUNET_TRAINER_DIR=$(python -c "import nnunetv2.training.nnUNetTrainer as m; import os; print(os.path.dirname(m.__file__))" 2>/dev/null || echo "")

# ==============================================================
# Helper
# ==============================================================
log() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
    echo ""
}

# ==============================================================
# Step 1: Convert DeepPSMA
# ==============================================================
do_convert() {
    log "Step 1: Converting DeepPSMA and merging into main dataset"
    python "${REPO_ROOT}/convert_and_merge_deeppsma.py" \
        --deeppsma_dir "${DEEPPSMA_DIR}" \
        --main_dataset "${DATA_DIR}" \
        --workers ${HEATMAP_WORKERS}
}

# ==============================================================
# Step 2: Generate heatmaps
# ==============================================================
do_heatmaps() {
    log "Step 2: Generating scribble heatmaps (3 strategies)"
    python "${REPO_ROOT}/generate_all_heatmaps.py" \
        --data_dir "${DATA_DIR}" \
        --workers ${HEATMAP_WORKERS}
}

# ==============================================================
# Step 3: nnU-Net preprocessing + plans
# ==============================================================
do_preprocess() {
    log "Step 3: nnU-Net directory setup and preprocessing"

    # Verify data
    for f in "imagesTr" "labelsTr" "splits_final.json"; do
        if [ ! -e "${DATA_DIR}/${f}" ]; then
            echo "[ERROR] ${f} not found in ${DATA_DIR}"
            exit 1
        fi
    done

    N_0002=$(find "${DATA_DIR}/imagesTr" -name "*_0002.nii.gz" 2>/dev/null | wc -l)
    if [ "$N_0002" -lt 100 ]; then
        echo "[ERROR] Only ${N_0002} heatmap files found. Run: bash $0 heatmaps"
        exit 1
    fi
    echo "[OK] Found ${N_0002} heatmap files"

    # Install custom trainer
    if [ -n "${NNUNET_TRAINER_DIR}" ] && [ -d "${NNUNET_TRAINER_DIR}" ]; then
        cp "${REPO_ROOT}/nnUNetTrainerAutoPETV.py" "${NNUNET_TRAINER_DIR}/"
        echo "[OK] Copied nnUNetTrainerAutoPETV.py to ${NNUNET_TRAINER_DIR}/"
    else
        echo "[WARN] Could not find nnunetv2 trainer directory."
        echo "       Manually copy nnUNetTrainerAutoPETV.py to your nnunetv2 installation."
    fi

    # Create nnUNet directories
    mkdir -p "${NNUNET_RAW}/${DATASET_NAME}"
    mkdir -p "${NNUNET_PREPROCESSED}"
    mkdir -p "${NNUNET_RESULTS}"

    ln -sfn "${DATA_DIR}/imagesTr" "${NNUNET_RAW}/${DATASET_NAME}/imagesTr"
    ln -sfn "${DATA_DIR}/labelsTr" "${NNUNET_RAW}/${DATASET_NAME}/labelsTr"
    cp "${DATA_DIR}/dataset.json" "${NNUNET_RAW}/${DATASET_NAME}/dataset.json"
    echo "[OK] Dataset symlinks created"

    export nnUNet_raw="${NNUNET_RAW}"
    export nnUNet_preprocessed="${NNUNET_PREPROCESSED}"
    export nnUNet_results="${NNUNET_RESULTS}"

    echo "  nnUNet_raw          = ${nnUNet_raw}"
    echo "  nnUNet_preprocessed = ${nnUNet_preprocessed}"
    echo "  nnUNet_results      = ${nnUNet_results}"

    echo ""
    echo ">>> Extracting dataset fingerprint..."
    nnUNetv2_extract_fingerprint -d ${DATASET_ID}

    echo ""
    echo ">>> Generating default plans..."
    nnUNetv2_plan_experiment -d ${DATASET_ID}

    echo ""
    echo ">>> Preprocessing (3d_fullres only)..."
    nnUNetv2_preprocess -d ${DATASET_ID} -c 3d_fullres

    echo ""
    echo ">>> Generating ResEnc-M plans (${GPU_MEMORY_TARGET}GB target)..."
    nnUNetv2_plan_experiment \
        -d ${DATASET_ID} \
        -pl nnUNetPlannerResEncM \
        -gpu_memory_target ${GPU_MEMORY_TARGET} \
        -overwrite_plans_name "${PLANS_NAME}"

    cp "${DATA_DIR}/splits_final.json" \
       "${NNUNET_PREPROCESSED}/${DATASET_NAME}/splits_final.json"
    echo "[OK] splits_final.json copied"
    echo "[OK] Preprocessing complete"
}

# ==============================================================
# Step 4: Training
# ==============================================================
do_train() {
    log "Training fold ${TRAIN_FOLD} on GPU ${TRAIN_GPU}"

    export nnUNet_raw="${NNUNET_RAW}"
    export nnUNet_preprocessed="${NNUNET_PREPROCESSED}"
    export nnUNet_results="${NNUNET_RESULTS}"

    PLANS_FILE="${NNUNET_PREPROCESSED}/${DATASET_NAME}/${PLANS_NAME}.json"
    if [ ! -f "${PLANS_FILE}" ]; then
        echo "[ERROR] Plans not found: ${PLANS_FILE}"
        echo "        Run: bash $0 setup"
        exit 1
    fi

    echo "  Trainer : ${TRAINER_NAME}"
    echo "  Plans   : ${PLANS_NAME}"
    echo "  Fold    : ${TRAIN_FOLD}"
    echo "  GPU     : ${TRAIN_GPU}"
    echo "  Results : ${NNUNET_RESULTS}"
    echo ""

    CUDA_VISIBLE_DEVICES=${TRAIN_GPU} nnUNetv2_train \
        ${DATASET_ID} 3d_fullres ${TRAIN_FOLD} \
        -tr "${TRAINER_NAME}" \
        -p "${PLANS_NAME}"

    echo ""
    echo "[OK] Training fold ${TRAIN_FOLD} complete."
}

# ==============================================================
# Dispatch
# ==============================================================
MODE="${1:-all}"

case "${MODE}" in
    convert)
        do_convert
        ;;
    heatmaps)
        do_heatmaps
        ;;
    setup)
        do_convert
        do_heatmaps
        do_preprocess
        ;;
    train)
        do_train
        ;;
    all)
        do_convert
        do_heatmaps
        do_preprocess
        do_train
        ;;
    *)
        echo "Usage: bash $0 [convert|heatmaps|setup|train|all]"
        echo ""
        echo "  convert   : Convert DeepPSMA and merge into main dataset"
        echo "  heatmaps  : Generate scribble heatmaps for all cases"
        echo "  setup     : convert + heatmaps + preprocessing (no training)"
        echo "  train     : Train a single fold"
        echo "  all       : setup + train (default)"
        echo ""
        echo "  Override fold/GPU: TRAIN_FOLD=1 TRAIN_GPU=1 bash $0 train"
        exit 1
        ;;
esac