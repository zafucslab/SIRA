#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
station="${1:-tonglu}"
RANDOM_SEED="${2:-2021}"
TRAIN_EPOCHS="${EPOCHS:-10}"
PYTHON="${PYTHON:-python}"
ROOT_PATH=dataset/processed
graph_path="graphs/${station}.csv"
[[ -f "$ROOT_PATH/$station.csv" && -f "$graph_path" ]] || { echo "Check station name and run bash scripts/prepare.sh first." >&2; exit 1; }
unset INIT_CHECKPOINT
export SIRA_FREEZE_BASE=0
export PRIOR_GRAPH_PATH="${graph_path}"
export PCFB_GRAPH_PATH="${graph_path}"
export PCFB_GATE_PATH="${graph_path}"
export PCFB_GATE_NORMALIZE="column"
export PCFB_GATE_POWER="1.0"
export PCFB_LAG="1"
export PCFB_TARGET_ONLY_FUSE="1"
export PCFB_FUSION_MODE="input"
export PCFB_BASE_SCALE="1.0"
export PCFB_RESIDUAL_SCALE="0.0"
export PCFB_STRENGTH="0.1"
export PCFB_ALPHA_MODE="variable"
export PCFB_ALPHA_INIT="0.02"
export PCFB_INIT_STD="0.0"
export DARA_BASE_CAUSAL_FUSION="0"
export DARA_DISABLE="0"
export DARA_DISABLE_CAUSAL="1"
export DARA_DISABLE_TREND="1"
export DARA_DISABLE_LOCAL="0"
export DARA_EXPERT_HIDDEN="32"
export DARA_ROUTER_HIDDEN="16"
export DARA_LOCAL_CONTEXT="48"
export DARA_TREND_KERNEL="25"
export DARA_NULL_BIAS="1.0"
export DARA_MAX_LATE_SCALE="0.25"
export DARA_CORRECTION_CLIP="0.5"
export DARA_HISTORY_LOSS_WEIGHT="0.10"
export SGR_LAGS="1,3,7,14,28"
export SGR_HIDDEN_SIZE="32"
export SGR_NULL_BIAS="2.0"
export SGR_MAX_SCALE="0.10"
export SGR_CORRECTION_CLIP="0.5"
export SGR_VAR_LAMBDA="0.01"
export SGR_LAG_LAMBDA="0.005"
export SGR_SELF_LOSS_WEIGHT="0.25"
export TRAIN_TARGET_ONLY="0"
export VALI_TARGET_ONLY="1"
export TEST_TARGET_ONLY="1"
export FORCE_TARGET_VALI="1"
export TARGET_AUX_LOSS_WEIGHT="0.0"
export SIRA_TIME_LOSS="mse"
export STAGE_LOSS_WEIGHTS="legacy"
run_stage() {
    "$PYTHON" -u -m exp.SIRA --random_seed "$RANDOM_SEED" --model SIRA "$@" \
    --data custom_masked  \
    --root_path "${ROOT_PATH}/"  \
    --data_path "${station}.csv"  \
    --mask_data_path "${station}_mask.csv"  \
    --features M  \
    --target soil_moisture_5cm  \
    --freq d  \
    --seq_len 192  \
    --label_len 0  \
    --pred_len 24  \
    --alpha_freq 0.0  \
    --revin 1  \
    --enc_in 10  \
    --dec_in 10  \
    --c_out 10  \
    --d_model 512  \
    --n_heads 8  \
    --e_layers 2  \
    --d_layers 3  \
    --d_ff 64  \
    --factor 1  \
    --embed timeF  \
    --dropout 0.5  \
    --level_dim 1  \
    --norm_name layer  \
    --n_block 1  \
    --pe 1  \
    --r_ema 0.996  \
    --num_workers 0  \
    --itr 1  \
    --gamma 0.1  \
    --train_epochs "${TRAIN_EPOCHS}"  \
    --batch_size 4  \
    --patience 5  \
    --learning_rate 0.002  \
    --des main  \
    --loss mse  \
    --lradj TST  \
    --pct_start 0.3  \
    --use_gpu True  \
    --gpu 0
}
base="${station}_s${RANDOM_SEED}_stage1"
final="${station}_s${RANDOM_SEED}_stage2"
export SGR_DISABLE=1
run_stage --is_training 1 --validation_only --model_id "$base"
checkpoints=(checkpoints/"${base}_SIRA_"*"/checkpoint.pth")
[[ ${#checkpoints[@]} == 1 && -f "${checkpoints[0]}" ]] || { echo "Expected one stage-one checkpoint" >&2; exit 1; }
export INIT_CHECKPOINT="${checkpoints[0]}" SIRA_FREEZE_BASE=1 SGR_DISABLE=0
run_stage --is_training 1 --validation_only --model_id "$final"
unset INIT_CHECKPOINT
export SIRA_FREEZE_BASE=0
run_stage --is_training 0 --model_id "$final"
