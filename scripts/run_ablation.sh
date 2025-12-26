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


echo "Running B_cond"
python -m scripts.train --exp B_cond --root ${ROOT} --seed ${SEED} --epochs 1 --batch_size 4 --img_size 120 --clip_len 64 --cond_drop 0.0
B_CKPT=$(ls -t outputs/runs/B_cond*/ckpt/best.ckpt | head -n 1)


echo "Running C_full"
python -m scripts.train --exp C_full --root ${ROOT} --seed ${SEED} --epochs 1 --batch_size 4 --img_size 120 --clip_len 64 --cond_drop 0.25 --use_film 1
C_CKPT=$(ls -t outputs/runs/C_full*/ckpt/best.ckpt | head -n 1)

for w in ${GUIDANCE_WEIGHTS}; do
  echo "Sampling guidance w=${w}"
  python -m scripts.sample --exp C_full --ckpt ${C_CKPT} --root ${ROOT} --seed ${SEED} --steps ${STEPS} --cfg_w ${w} --num_per_class ${NUM_PER_CLASS} --run_name C_full_guided_w${w}
done

echo "Ablation finished"
