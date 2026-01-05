#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zfl/code/point3/data/real}
SEED=${SEED:-0}
STEPS=${STEPS:-50}
NUM_PER_CLASS=${NUM_PER_CLASS:-64}
GUIDANCE_WEIGHTS=${GUIDANCE_WEIGHTS:-"1.0 2.0"}

echo "Running A_base"
python -m scripts.train --exp A_base --root ${ROOT} --seed ${SEED} --epochs 1 --batch_size 4 --img_size 120 --clip_len 64
A_CKPT=$(ls -t outputs/runs/A_base*/ckpt/best.ckpt | head -n 1)
if [ -z "${A_CKPT}" ]; then
  A_CKPT=$(ls -t outputs/runs/train_A_base*/ckpt/best.ckpt | head -n 1 || true)
fi
if [ -z "${A_CKPT}" ]; then
  echo "A_base checkpoint not found"; exit 1; fi


echo "Running B_cond"
python -m scripts.train --exp B_cond --root ${ROOT} --seed ${SEED} --epochs 1 --batch_size 4 --img_size 120 --clip_len 64 --cond_drop 0.0
B_CKPT=$(ls -t outputs/runs/B_cond*/ckpt/best.ckpt | head -n 1)
if [ -z "${B_CKPT}" ]; then
  B_CKPT=$(ls -t outputs/runs/train_B_cond*/ckpt/best.ckpt | head -n 1 || true)
fi
if [ -z "${B_CKPT}" ]; then
  echo "B_cond checkpoint not found"; exit 1; fi


echo "Running C_film"
python -m scripts.train --exp C_film --root ${ROOT} --seed ${SEED} --epochs 1 --batch_size 4 --img_size 120 --clip_len 64 --cond_drop 0.0 --use_film 1
C_CKPT=$(ls -t outputs/runs/train_C_film*/ckpt/best.ckpt | head -n 1)
if [ -z "${C_CKPT}" ]; then
  C_CKPT=$(ls -t outputs/runs/C_film*/ckpt/best.ckpt | head -n 1 || true)
fi
if [ -z "${C_CKPT}" ]; then
  echo "C_film checkpoint not found"; exit 1; fi


echo "Running D_full (CrossAttn + FiLM, no CFG)"
python -m scripts.train --exp D_full --root ${ROOT} --seed ${SEED} --epochs 1 --batch_size 4 --img_size 120 --clip_len 64 --cond_drop 0.0 --use_film 1
D_CKPT=$(ls -t outputs/runs/train_D_full*/ckpt/best.ckpt | head -n 1)
if [ -z "${D_CKPT}" ]; then
  D_CKPT=$(ls -t outputs/runs/D_full*/ckpt/best.ckpt | head -n 1 || true)
fi
if [ -z "${D_CKPT}" ]; then
  echo "D_full checkpoint not found"; exit 1; fi

echo "Running E_full (CrossAttn + FiLM + CFG)"
python -m scripts.train --exp E_full --root ${ROOT} --seed ${SEED} --epochs 1 --batch_size 4 --img_size 120 --clip_len 64 --cond_drop 0.25 --use_film 1
E_CKPT=$(ls -t outputs/runs/train_E_full*/ckpt/best.ckpt | head -n 1)
if [ -z "${E_CKPT}" ]; then
  E_CKPT=$(ls -t outputs/runs/E_full*/ckpt/best.ckpt | head -n 1 || true)
fi
if [ -z "${E_CKPT}" ]; then
  echo "E_full checkpoint not found"; exit 1; fi

echo "Running F_freq (E_full + freq-band loss)"
python -m scripts.train --exp F_freq --root ${ROOT} --seed ${SEED} --epochs 1 --batch_size 4 --img_size 120 --clip_len 64 --cond_drop 0.25 --use_film 1 --freq_lambda 0.1
F_CKPT=$(ls -t outputs/runs/train_F_freq*/ckpt/best.ckpt | head -n 1)
if [ -z "${F_CKPT}" ]; then
  F_CKPT=$(ls -t outputs/runs/F_freq*/ckpt/best.ckpt | head -n 1 || true)
fi
if [ -z "${F_CKPT}" ]; then
  echo "F_freq checkpoint not found"; exit 1; fi

for w in ${GUIDANCE_WEIGHTS}; do
  echo "Sampling guidance w=${w}"
  python -m scripts.sample --exp D_full --ckpt ${D_CKPT} --root ${ROOT} --seed ${SEED} --steps ${STEPS} --cfg_w ${w} --num_per_class ${NUM_PER_CLASS} --run_name D_full_guided_w${w}
  python -m scripts.sample --exp E_full --ckpt ${E_CKPT} --root ${ROOT} --seed ${SEED} --steps ${STEPS} --cfg_w ${w} --num_per_class ${NUM_PER_CLASS} --run_name E_full_guided_w${w}
  python -m scripts.sample --exp F_freq --ckpt ${F_CKPT} --root ${ROOT} --seed ${SEED} --steps ${STEPS} --cfg_w ${w} --num_per_class ${NUM_PER_CLASS} --run_name F_freq_guided_w${w}
done

echo "Ablation finished"
