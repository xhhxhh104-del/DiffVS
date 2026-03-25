# DiffVS: Marker-wise Conditioned Latent Diffusion for Virtual Multiplex Staining

本仓库当前实现按论文描述的核心流程搭建：
1) H&E / mIHC 原始 `.tif` 数据读取（HEMIT 原始格式）；
2) 统一转灰度单 marker 监督；
3) Latent Diffusion 主干（Autoencoder + Marker-conditioned U-Net）；
4) 两阶段训练（扩散训练 + 单步采样微调）；
5) 单步推理生成每个 marker 的预测图。

> 论文：**Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model** (AAAI 2026)

---

## 项目结构

```text
DiffVS/
├── configs/
│   └── train_example.yaml
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── infer.py
├── src/diffvs/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── infer.py
│   ├── data/
│   │   └── hemit.py
│   ├── diffusion/
│   │   └── scheduler.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   └── unet.py
│   └── engine/
│       ├── trainer.py
│       └── inferencer.py
└── pyproject.toml
```

---

## 数据格式（HEMIT 原始 tif）

该实现默认按“同一父目录中 HE 与 mIHC 配对”的方式检索：

```text
<root>/case_xxx/
  ├── ...HE....tif
  ├── ...mIHC..._CD3.tif
  ├── ...mIHC..._CD20.tif
  └── ...
```

其中 marker 名通过文件名 split 得到（由配置控制）：
- `marker_from_name_sep`
- `marker_from_name_index`

> mIHC 与 HE 在训练时都会被转换为灰度并归一化到 `[0,1]`，用于单 marker 监督。

---

## 训练流程（按论文思路）

### Stage-1: Diffusion training
- 在 latent 空间对 target marker latent 加噪；
- U-Net 预测噪声；
- 损失：`noise MSE + latent x0 L1`。

### Stage-2: One-step fine-tuning
- 从高噪声 latent 出发，执行单步反演；
- 解码到像素空间；
- 损失：`pixel L1 + pixel MSE`。

---

## 快速开始

### 1) 安装

```bash
python -m pip install -e .
```

### 2) 修改配置

编辑 `configs/train_example.yaml`，重点改：
- `data.root`
- `data.he_glob`
- `data.mihc_glob`
- marker 提取规则

### 3) 扫描并导出 manifest（可选，但推荐）

```bash
python scripts/preprocess.py --config configs/train_example.yaml --output outputs/manifest.json
```

### 4) 训练

```bash
python scripts/train.py --config configs/train_example.yaml
```

### 5) 推理

```bash
python scripts/infer.py \
  --config configs/train_example.yaml \
  --checkpoint outputs/checkpoints/onestep_epoch_020.pt \
  --he-tif /path/to/one_he.tif \
  --output-dir outputs/infer_case_001
```

推理会输出：
- `pred_<marker>.npy`
- `pred_<marker>.tif`
- `markers.json`

---

## 说明

- 当前实现重点是“完整训练/推理工程骨架 + 论文流程一致的数据流”，方便你先在 HEMIT 原始 tif 上复现实验。
- 如需进一步完全对齐论文数值结果，下一步建议继续补：
  - 论文同款 backbone 与超参；
  - 数据划分策略与评估脚本；
  - 指标复现（marker-wise 定量评估）。
