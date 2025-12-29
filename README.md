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
```bash
python -m scripts.train --exp A_base --seed 0
python -m scripts.train --exp B_cond --seed 0
python -m scripts.train --exp C_full --seed 0 --cond_drop 0.25 --use_film 1
```
Key options: `--root <path to data root (default: repo_root/data)>`, `--img_size 120`, `--clip_len 64`, `--batch_size 32`, `--epochs 50` (default), `--run_name <custom>`, `--use_amp 1` (default), `--early_stop_patience 5`, `--early_stop_min_delta 1e-3`.

Outputs land in `outputs/runs/<run_name>/{logs,ckpt,samples,metrics}/`. Each run saves `config.json` with seed and git state.

## Sampling
Run names are fixed per实验配置，便于复现与后续评估：
```bash
# A_base
python -m scripts.sample --exp A_base --ckpt <path/to/ckpt.pth> --run_name sample_A_base

# B_cond
python -m scripts.sample --exp B_cond --ckpt <path/to/ckpt.pth> --run_name sample_B_cond

# C_full（可带线性 CFG 调度示例）
python -m scripts.sample --exp C_full --ckpt <path/to/ckpt.pth> --run_name sample_C_full \
  --schedule linear --cfg_w0 0.5 --cfg_w1 1.5 --steps 50 --seed 0
```
Samples are stored under `outputs/runs/sample_<EXP>/samples/<action>/` without overwriting.

## Evaluation
Use the fixed-name radar classifier to score generated samples (matching the above fixed sample run names):
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
