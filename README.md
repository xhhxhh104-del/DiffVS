# Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model

**AAAI 2026 Accepted**

This repository contains the official implementation of the paper:
> **Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model**  
> Hyun-Jic Oh, Junsik Kim, Zhiyi Shi, Yichen Wu, Yu-An Chen, Peter K. Sorger, Hanspeter Pfister, Won-Ki Jeong

## [Overview]

![Overview Figure](Figure/overview.png)

We propose a marker-wise conditioned latent diffusion framework that generates virtual multiplex (mIF/mIHC) marker channels directly from corresponding H&E images while sharing a single unified architecture across all markers.
The model supports marker-by-marker synthesis, accommodates heterogeneous marker intensity distributions, and is fine-tuned for single-step inference to improve both visual fidelity and runtime efficiency.

---

## Project structure (starter implementation)

```text
DiffVS/
├── Figure/
│   └── overview.png
├── configs/
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── infer.py
├── src/diffvs/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── infer.py
└── pyproject.toml
```

### What is implemented

- **Preprocessing**: convert image files to normalized grayscale `.npy` tensors.
- **Data loader**: simple marker-wise directory layout backed by NumPy files.
- **Training**: compact marker-conditioned denoiser with noise-prediction objective.
- **Inference**: marker-by-marker synthesis from H&E using iterative denoising (single-step by default).

### Expected dataset layout

```text
data_root/
├── sample_000/
│   ├── he.npy
│   ├── marker_CD3.npy
│   └── marker_CD20.npy
└── sample_001/
    ├── he.npy
    ├── marker_CD3.npy
    └── marker_CD20.npy
```

### Quick start

```bash
python -m pip install -e .

# train
python scripts/train.py \
  --data-root /path/to/data_root \
  --markers CD3 CD20 \
  --epochs 20 \
  --batch-size 4 \
  --save-dir checkpoints

# inference
python scripts/infer.py \
  --checkpoint checkpoints/epoch_020.pt \
  --he-path /path/to/sample_000/he.npy \
  --output-dir outputs \
  --steps 1
```

---

### Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{oh2025virtual,
  title   = {Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model},
  author  = {Oh, Hyun-Jic and Kim, Junsik and Shi, Zhiyi and Wu, Yichen and Chen, Yu-An and Sorger, Peter K and Pfister, Hanspeter and Jeong, Won-Ki},
  journal = {arXiv preprint arXiv:2508.14681},
  year    = {2025}
}
```

## Acknowledgements

Our implementation, training scripts, and evaluation pipelines heavily draw inspiration from [Marigold](https://github.com/prs-eth/Marigold) and [diffusion-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft), and we gratefully acknowledge their authors for releasing high-quality code and models.
