# Video-Conditioned Rectified Flow for Radar Micro-Doppler Generation

Minimal PyTorch project for video-conditioned rectified flow generation of real radar micro-Doppler spectrograms. The code focuses on four ablations:

- **A_base**: Rectified Flow backbone (unconditional option).
- **B_cond**: Rectified Flow + video conditioning.
- **C_film**: Rectified Flow + video conditioning + FiLM (no CFG).
- **D_full**: Rectified Flow + video conditioning + FiLM + CFG training/guidance.

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
训练的四个实验默认使用固定的 run 名称，权重固定命名为 `best.ckpt` / `last.ckpt` / `epoch_*.ckpt`（默认 batch size=128，其他参数用脚本默认即可）：
```bash
# A_base（run_name 默认 train_A_base）
python -m scripts.train --exp A_base

# B_cond（run_name 默认 train_B_cond）
python -m scripts.train --exp B_cond

# C_film（run_name 默认 train_C_film，FiLM + 条件分支，无 CFG）
python -m scripts.train --exp C_film

# D_full（run_name 默认 train_D_full，等价于旧的 C_full；命令 --exp C_full 作为别名仍可用）
python -m scripts.train --exp D_full
# 兼容命令（别名）：python -m scripts.train --exp C_full
```
输出目录固定为 `outputs/runs/train_<EXP>/{logs,ckpt,metrics}/`，其中权重在 `ckpt/best.ckpt`。

## Sampling
四种采样模式的输出 run 名称固定为 `sample_<EXP>`，指向上面固定的训练权重（参数用默认即可）：
```bash
# A_base
python -m scripts.sample --exp A_base --ckpt outputs/runs/train_A_base/ckpt/best.ckpt

# B_cond
python -m scripts.sample --exp B_cond --ckpt outputs/runs/train_B_cond/ckpt/best.ckpt

# C_film（FiLM + 条件分支，不计算 CFG）
python -m scripts.sample --exp C_film --ckpt outputs/runs/train_C_film/ckpt/best.ckpt

# D_full（等价于旧的 C_full；默认 CFG w=3，如需线性 CFG 可按需追加调度参数）
python -m scripts.sample --exp D_full --ckpt outputs/runs/train_D_full/ckpt/best.ckpt --cfg_w 3
# 兼容命令（别名）：python -m scripts.sample --exp C_full --ckpt outputs/runs/train_D_full/ckpt/best.ckpt --cfg_w 3
```
Samples are stored under `outputs/runs/sample_<EXP>/samples/<action>/` without overwriting.

## Evaluation
训练与采样的 run 名称目前写死为 `train_<EXP>` / `sample_<EXP>`（输出结构见上文），评估时请保持相同命名，否则需要自行修改路径。
使用固定名称的雷达分类器对采样结果进行打分：
```bash
python -m scripts.eval_gen_with_cls --root outputs/runs/sample_A_base/samples \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth --out_json outputs/runs/sample_A_base/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_B_cond/samples \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth --out_json outputs/runs/sample_B_cond/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_D_full/samples \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth --out_json outputs/runs/sample_D_full/metrics/eval.json
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
