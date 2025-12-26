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
```bash
python -m scripts.sample --exp C_full --ckpt <path> --cfg_w <float> --steps 50 --seed 0 --num_per_class 64
python -m scripts.sample --exp C_full --ckpt <path> --schedule linear --cfg_w0 <float> --cfg_w1 <float> --steps 50 --seed 0
```
Samples are stored under `outputs/runs/<run_name>/samples/<action>/` without overwriting.

## Evaluation
Use a pretrained radar classifier to score generated samples:
```bash
python -m scripts.eval_gen_with_cls --root outputs/runs/<run_name>/samples \
  --cls_ckpt checkpoints/radar_cls_resnet18_best.pth --out_json outputs/runs/<run_name>/metrics/eval.json
```

## One-click ablation
```bash
bash scripts/run_ablation.sh
```
Environment variables: `ROOT`, `SEED`, `STEPS`, `NUM_PER_CLASS`, `GUIDANCE_WEIGHTS`.

## Notes
- All entry points accept `--seed` for reproducibility.
- Guidance weights are never hard-coded; sweep via CLI or `run_ablation.sh`.
- No auxiliary classifiers or extra losses are included by default.
