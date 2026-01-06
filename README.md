# Video-Conditioned Rectified Flow for Radar Micro-Doppler Generation

Minimal PyTorch project for video-conditioned rectified flow generation of real radar micro-Doppler spectrograms. The code focuses on five ablations plus two structured-loss extensions:

- **A_base**: Rectified Flow backbone (unconditional option).
- **B_cond**: Rectified Flow + video conditioning.
- **C_film**: Rectified Flow + video conditioning + FiLM (no CFG).
- **D_full**: C_film + cross attention (no CFG).
- **E_full**: FiLM + cross attention + CFG training/guidance. The old alias `C_full` maps here (legacy CFG variant).
- **F_freq**: E_full + optional frequency-band energy consistency loss (disabled by default).
- **G_grad**: E_full + optional spectral gradient structure consistency loss (disabled by default unless set for this exp).
- **H_taware**: F_freq + G_grad with per-sample t-aware mixing between frequency and gradient losses.
- **K_color**: H_taware variant that keeps color (3-channel) radar spectrograms instead of converting to grayscale.

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
- 优化器/调度：AdamW + 余弦学习率调度（默认），可选线性 warmup（`--warmup_epochs`），训练中自动跟踪 EMA 权重（`--ema_decay`）。
- 阶段式损失：可通过 `--contrast_start_epoch` / `--adv_start_epoch` 指定对比/对抗分支启用的 epoch，满足“先重建+频域+时序，稳定后再启用对比/对抗”的需求。
- 训练的 run 名称固定为 `train_<EXP>`，权重命名为 `best.ckpt` / `last.ckpt` / `epoch_*.ckpt`（默认 batch size=128，其他参数用脚本默认即可）。
- 保存的模型权重已默认切换为 EMA 版本，供采样和评估直接使用（同时在 checkpoint 中保留 EMA 状态以便恢复）。

### 推荐的阶段式训练示例（E_full：FiLM + CrossAttn + CFG）
以下命令先进行 5 个 epoch 的 warmup（只训练重建/频域/时序），第 6 个 epoch 起加入 InfoNCE 对比损失，第 10 个 epoch 起加入对抗分支，并开启 EMA：
```bash
python -m scripts.train --exp E_full \
  --epochs 50 \
  --batch_size 128 \
  --lr 3e-4 \
  --lr_scheduler cosine \
  --warmup_epochs 5 \
  --freq_lambda 0.1 \
  --temporal_lambda 0.05 \
  --infonce_lambda 0.1 --contrast_start_epoch 6 \
  --action_adv 1 --adv_lambda 0.5 --adv_start_epoch 10 \
  --ema_decay 0.9999
```

### 最终实验一键指令（可直接复制执行）
下列指令完成“训练→采样→评估”的全流程（假设数据放在 `data/`，使用 E_full 配置与上面的阶段式超参，评估使用默认分类器权重）。如需切换不同实验，只要改 `--exp`、`--ckpt` 路径和对应的 `sample_<EXP>`/`train_<EXP>` 名称。
```bash
# 1) 训练（阶段式重建→对比→对抗，含余弦LR + EMA）
python -m scripts.train --exp E_full \
  --epochs 50 \
  --batch_size 128 \
  --lr 3e-4 \
  --lr_scheduler cosine \
  --warmup_epochs 5 \
  --freq_lambda 0.1 \
  --temporal_lambda 0.05 \
  --infonce_lambda 0.1 --contrast_start_epoch 6 \
  --action_adv 1 --adv_lambda 0.5 --adv_start_epoch 10 \
  --ema_decay 0.9999

# 2) 采样（固定 run_name=sample_E_full，读训练得到的 best.ckpt）
python -m scripts.sample --exp E_full \
  --ckpt outputs/runs/train_E_full/ckpt/best.ckpt \
  --cfg_w 3

# 3) 评估（使用预训练雷达分类器，输出 JSON 指标；会自动汇总 ResNet18 + 其他可用分类器）
# 若目录下存在 EfficientNet-B0 / ConvNeXt-Tiny / Swin-Tiny 的权重，会自动一并评估并写入 JSON。
# 如需显式追加自定义分类器，可叠加 --extra_cls_ckpt arch=/path/to/ckpt.pth。
python -m scripts.eval_gen_with_cls --root outputs/runs/sample_E_full/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_E_full/metrics/eval.json

# 4) （可选）分类器自训练与评测，如需重训或验证分类器
python -m scripts.train_classifier --root data --epochs 30 --batch_size 32 --lr 1e-4 --weight_decay 5e-4 --scheduler_patience 2 --early_stop_patience 5 --freeze_backbone_epochs 2 --class_weight_box 1.1 --pretrained 1
python -m scripts.eval_classifier --root data --split test \
  --ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/classifier/radar_cls_resnet18/metrics/test_eval.json
```

### 其他实验（保持原有结构）
```bash
# A_base（run_name 默认 train_A_base）
python -m scripts.train --exp A_base

# B_cond（run_name 默认 train_B_cond）
python -m scripts.train --exp B_cond

# C_film（run_name 默认 train_C_film，FiLM + 条件分支，无 CFG）
python -m scripts.train --exp C_film

# D_full（run_name 默认 train_D_full，新版：FiLM + CrossAttn，无 CFG）
python -m scripts.train --exp D_full

# E_full（run_name 默认 train_E_full，FiLM + CrossAttn + CFG，旧版 D_full；命令 --exp C_full 作为别名）
python -m scripts.train --exp E_full
# 兼容命令（别名）：python -m scripts.train --exp C_full

# F_freq（E_full 基础上可选频带能量一致性，默认关闭）
python -m scripts.train --exp F_freq --freq_lambda 0.1

# G_grad（E_full 结构 + 梯度统计一致性，可独立开关权重）
python -m scripts.train --exp G_grad --grad_lambda 0.05

# H_taware（F_freq + G_grad，按 t 自适应权重混合）
python -m scripts.train --exp H_taware --freq_lambda 0.1 --grad_lambda 0.05 --taware 1 --t_low 0.3 --t_high 0.7

# K_color（H_taware 基础上保留彩色雷达谱图，输入三通道）
python -m scripts.train --exp K_color --freq_lambda 0.1 --grad_lambda 0.05 --taware 1 --t_low 0.3 --t_high 0.7 --radar_channels 3
```
输出目录固定为 `outputs/runs/train_<EXP>/{logs,ckpt,metrics}/`，其中权重在 `ckpt/best.ckpt`。

## Sampling
各实验的采样模式输出 run 名称固定为 `sample_<EXP>`，指向上面固定的训练权重（参数用默认即可）：
```bash
# A_base
python -m scripts.sample --exp A_base --ckpt outputs/runs/train_A_base/ckpt/best.ckpt

# B_cond
python -m scripts.sample --exp B_cond --ckpt outputs/runs/train_B_cond/ckpt/best.ckpt

# C_film（FiLM + 条件分支，不计算 CFG）
python -m scripts.sample --exp C_film --ckpt outputs/runs/train_C_film/ckpt/best.ckpt

# D_full（FiLM + CrossAttn，无 CFG）
python -m scripts.sample --exp D_full --ckpt outputs/runs/train_D_full/ckpt/best.ckpt

# E_full（FiLM + CrossAttn + CFG，旧别名 C_full，默认 CFG w=3，如需线性 CFG 可按需追加调度参数）
python -m scripts.sample --exp E_full --ckpt outputs/runs/train_E_full/ckpt/best.ckpt --cfg_w 3
# 兼容命令（别名）：python -m scripts.sample --exp C_full --ckpt outputs/runs/train_E_full/ckpt/best.ckpt --cfg_w 3

# F_freq（与 E_full 采样路径一致，训练包含可选频带 loss，采样默认不变）
python -m scripts.sample --exp F_freq --ckpt outputs/runs/train_F_freq/ckpt/best.ckpt --cfg_w 3

# G_grad（与 E_full 采样路径一致，训练额外加入梯度 loss）
python -m scripts.sample --exp G_grad --ckpt outputs/runs/train_G_grad/ckpt/best.ckpt --cfg_w 3

# H_taware（与 E_full 采样路径一致，训练包含 t-aware 结构 loss）
python -m scripts.sample --exp H_taware --ckpt outputs/runs/train_H_taware/ckpt/best.ckpt --cfg_w 3

# K_color（与 H_taware 相同结构/采样路径，使用彩色雷达输入）
python -m scripts.sample --exp K_color --ckpt outputs/runs/train_K_color/ckpt/best.ckpt --cfg_w 3
```
Samples are stored under `outputs/runs/sample_<EXP>/samples/<action>/` without overwriting.

## Evaluation
训练与采样的 run 名称目前写死为 `train_<EXP>` / `sample_<EXP>`（输出结构见上文），评估时请保持相同命名，否则需要自行修改路径。
使用固定名称的雷达分类器对采样结果进行打分（默认会尝试加载 ResNet18 + EfficientNet-B0 + ConvNeXt-Tiny + Swin-Tiny，全部结果保存在同一个 JSON 中）：
```bash
python -m scripts.eval_gen_with_cls --root outputs/runs/sample_A_base/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_A_base/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_B_cond/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_B_cond/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_C_film/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_C_film/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_D_full/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_D_full/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_E_full/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_E_full/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_F_freq/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_F_freq/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_G_grad/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_G_grad/metrics/eval.json

python -m scripts.eval_gen_with_cls --root outputs/runs/sample_H_taware/samples \
  --cls_arch resnet18 \
  --cls_ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/runs/sample_H_taware/metrics/eval.json
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

### Multi-arch radar classifiers (EfficientNet-B0 / ConvNeXt-Tiny / Swin-Tiny)
一次性顺序训练三种 timm 分类器（默认监控 `val_loss`，固定 run name 避免覆盖现有 ResNet18）：
```bash
python -m scripts.train_radar_cls_multi \
  --root data \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --img_size 120 \
  --num_workers 4 \
  --seed 0 \
  --pretrained 1 \
  --scheduler_patience 2 \
  --early_stop_patience 5 \
  --hf_hub_download_timeout 60
```
输出分别位于：
- `outputs/classifier/radar_cls_efficientnet_b0/`
- `outputs/classifier/radar_cls_convnext_tiny/`
- `outputs/classifier/radar_cls_swin_tiny_patch4_window7_224/`

最终汇总指标写入 `outputs/classifier/radar_cls_multi_summary.json`。单独评测某个模型可用（示例）：
```bash
python -m scripts.eval_classifier --root data --split test \
  --ckpt outputs/classifier/radar_cls_efficientnet_b0/ckpt/best.pth \
  --out_json outputs/classifier/radar_cls_efficientnet_b0/metrics/test_eval.json

# 对真实数据一次性跑完全部分类器的评测（ResNet18 + EfficientNet-B0 + ConvNeXt-Tiny + Swin-Tiny）
python -m scripts.eval_classifier --root data --split test \
  --ckpt outputs/classifier/radar_cls_resnet18/ckpt/best.pth \
  --out_json outputs/classifier/radar_cls_resnet18/metrics/test_eval.json

python -m scripts.eval_classifier --root data --split test \
  --ckpt outputs/classifier/radar_cls_efficientnet_b0/ckpt/best.pth \
  --out_json outputs/classifier/radar_cls_efficientnet_b0/metrics/test_eval.json

python -m scripts.eval_classifier --root data --split test \
  --ckpt outputs/classifier/radar_cls_convnext_tiny/ckpt/best.pth \
  --out_json outputs/classifier/radar_cls_convnext_tiny/metrics/test_eval.json

python -m scripts.eval_classifier --root data --split test \
  --ckpt outputs/classifier/radar_cls_swin_tiny_patch4_window7_224/ckpt/best.pth \
  --out_json outputs/classifier/radar_cls_swin_tiny_patch4_window7_224/metrics/test_eval.json
```

> **Note:** 若所在环境无法从 HuggingFace 下载预训练权重（超时/离线），可直接添加 `--pretrained 0`，或保留 `--pretrained 1` 让脚本自动在下载失败后退回随机初始化继续训练。脚本会按每个模型的默认输入尺寸自动设置 `img_size`（例如 Swin-Tiny 为 224），`--img_size` 仅在模型未提供默认尺寸时作为兜底。

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
