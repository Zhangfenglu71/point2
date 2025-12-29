# Video-Conditioned Rectified Flow for Radar Micro-Doppler Generation

Minimal PyTorch project for video-conditioned rectified flow generation of real radar micro-Doppler spectrograms. The code focuses on three ablations:

- **A_base**: Rectified Flow backbone (unconditional option).
- **B_cond**: Rectified Flow + video conditioning.
- **C_full**: Rectified Flow + video conditioning + FiLM + CFG training/guidance.

## Data layout
```
/workspace/point2/data/{train,val,test}/
  radar/Sxx/<action>/*.jpg|png
  video/Sxx/<action>/*.mp4|avi|mov|mkv
```
Actions: `box`, `jump`, `run`, `walk`. Default subject split: train S01–S08, val S09, test S10 (configurable via CLI).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
训练的三个实验默认使用固定的 run 名称，生成的权重也固定命名为 `best.ckpt` / `last.ckpt` / `epoch_*.ckpt`：
```bash
# A_base（run_name 默认 train_A_base）
python -m scripts.train --exp A_base --root data --epochs 50 --batch_size 32 --seed 0

# B_cond（run_name 默认 train_B_cond）
python -m scripts.train --exp B_cond --root data --epochs 50 --batch_size 32 --seed 0 --cond_drop 0.0

# C_full（run_name 默认 train_C_full）
python -m scripts.train --exp C_full --root data --epochs 50 --batch_size 32 --seed 0 --cond_drop 0.25 --use_film 1
```
关键参数：`--root <数据根目录>`（默认 `repo_root/data`）、`--img_size 120`、`--clip_len 64`、`--batch_size 32`、`--epochs 50`（默认）、`--use_amp 1`（默认）、`--early_stop_patience 5`、`--early_stop_min_delta 1e-3`。  
输出目录固定为 `outputs/runs/train_<EXP>/{logs,ckpt,metrics}/`，其中权重在 `ckpt/best.ckpt`。

## Sampling
三种采样模式的输出 run 名称也固定为 `sample_<EXP>`，指向上面固定的训练权重：
```bash
# A_base
python -m scripts.sample --exp A_base \
  --ckpt outputs/runs/train_A_base/ckpt/best.ckpt \
  --run_name sample_A_base --steps 50 --seed 0

# B_cond
python -m scripts.sample --exp B_cond \
  --ckpt outputs/runs/train_B_cond/ckpt/best.ckpt \
  --run_name sample_B_cond --steps 50 --seed 0

# C_full（线性 CFG 调度示例）
python -m scripts.sample --exp C_full \
  --ckpt outputs/runs/train_C_full/ckpt/best.ckpt \
  --run_name sample_C_full \
  --schedule linear --cfg_w0 0.5 --cfg_w1 1.5 --steps 50 --seed 0
```
Samples are stored under `outputs/runs/sample_<EXP>/samples/<action>/` without overwriting.

## Evaluation
Use the fixed-name radar classifier to score generated samples（采样输出 run 名称固定为 `sample_<EXP>`）:
```bash
python -m scripts.eval_gen_with_cls --root outputs/runs/sample_C_full/samples \
  --cls_ckpt checkpoints/radar_cls_resnet18_best.pth --out_json outputs/runs/sample_C_full/metrics/eval.json
```

## One-click ablation
```bash
bash scripts/run_ablation.sh
```
Environment variables: `ROOT`, `SEED`, `STEPS`, `NUM_PER_CLASS`, `GUIDANCE_WEIGHTS`.

## Radar classifier for generation evaluation
Train a radar-only action classifier on the real spectrograms before scoring generated samples:
```bash
python -m scripts.train_classifier \
  --root data \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --scheduler_patience 2 \
  --early_stop_patience 5 \
  --freeze_backbone_epochs 2 \
  --class_weight_box 1.1 \
  --pretrained 1
```
Outputs (checkpoints, config, metrics) are stored under `outputs/classifier/<run_name>/`. The checkpoint `best.pth` is compatible with `scripts.eval_gen_with_cls`.

Evaluate a trained classifier (overall + per-action accuracy) on any split:
```bash
python -m scripts.eval_classifier --root data --split test \
  --ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/classifier/radar_cls_resnet18/metrics/test_eval.json
```

## Notes
- All entry points accept `--seed` for reproducibility.
- Guidance weights are never hard-coded; sweep via CLI or `run_ablation.sh`.
- No auxiliary classifiers or extra losses are included by default.
