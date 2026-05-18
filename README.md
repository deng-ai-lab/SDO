<p align="center">
  <img src="assets/SDO.png" alt="SDO Logo" height="200">
</p>

<h1 align="center">You Only Look One Step: Accelerating Backpropagation in Diffusion Sampling with Gradient Shortcuts</h1>

## 📌 Overview
Shortcut Diffusion Optimization (SDO) is a lightweight, high-performance approach to optimizing diffusion sampling with a one-step gradient shortcut. It targets fast latent optimization while preserving semantic guidance, and can be extended to network parameter tuning for alignment tasks.


<p align="center">
  <img src="assets/method2_00.png" height="170">
</p>

## 🧾 Abstract
We present SDO, a diffusion optimization method that accelerates backpropagation by restricting gradient flow to a single denoising step while keeping later steps fixed. This shortcut preserves the core guidance signal yet cuts memory and compute overhead. We demonstrate SDO on style-guided generation by optimizing the latent variable to match reference-style statistics measured by CLIP feature Gram matrices, while keeping text conditioning fixed.

## 🧩 Method
SDO runs a diffusion sampler in two phases:
1. A short no-gradient warmup to reach a reasonable latent state.
2. An optimization phase where only the first denoising step retains gradient flow, while later steps are evaluated without gradients.

This produces a lightweight training graph with a tractable memory footprint while still providing an effective signal to update the latent variable.

## 🛠 Implementation Notes
This repository includes a runnable reference implementation in `style_guide_implict.py`, which:
- Uses Stable Diffusion v1.4 for the denoising backbone.
- Applies classifier-free guidance with a fixed prompt.
- Optimizes a latent `z` to minimize the Gram-matrix style distance between the generated output and a reference image, using CLIP features.

It also includes a CLIP-guided latent optimization variant in `clip_guide_implict.py`, which:
- Uses a guided-diffusion UNet backbone and DDIM sampling.
- Optimizes a latent `z` with a CLIP text-image alignment loss plus a pixel similarity term.

An aesthetic-optimized variant in `aesthetic_implict.py`:
- Uses Stable Diffusion v1.4 and DDIM inversion to initialize the latent.
- Optimizes the latent to maximize (or target) an aesthetic score from CLIP-based heads.

## ✅ Requirements
- Python 3.8+
- PyTorch with CUDA (recommended)
- diffusers, transformers, clip, torchvision, numpy, Pillow, matplotlib

## 🚀 Usage
Basic example:
```bash
python style_guide_implict.py \
  --input_image style_image/xing.jpg \
  --prompt "A portrait of Caleb, a character from the Critical Role series." \
  --logdir ./results_sdo \
  --run_name style_run
```

All variants accept `--input_image` to override `--data_root/--dataset/--img`.

CLIP-guided variant:
```bash
python clip_guide_implict.py \
  --data_root ./data \
  --dataset ffhq_eval \
  --img 98 \
  --prompt "A photo of a smile face." \
  --logdir ./results_sdo \
  --run_name ffhq_eval/00098
```

Aesthetic variant:
```bash
python aesthetic_implict.py \
  --data_root ./data \
  --dataset bedroom \
  --img 80 \
  --aesthetic_target 10 \
  --logdir ./results_sdo \
  --run_name bedroom/00080
```

Optional parameters:
```bash
--custom_steps 50 --pre_steps 45 --optim_steps 200 --guidance_scale 5.0 --lr 0.01
```

## 🧾 Outputs
The script writes:
- `gt.png`: the reference style image
- `rec_img_*.png`: intermediate reconstructions per optimization step
- `loss.png`: the optimization loss curve

## 🧪 Results
<p align="center">
  <img src="assets/intro2_00.png" height="525">
</p>


## 📚 Citation
If you use this code or idea, please cite the project:
```bibtex
@article{dou2025you,
  title={You Only Look One Step: Accelerating Backpropagation in Diffusion Sampling with Gradient Shortcuts},
  author={Dou, Hongkun and Li, Zeyu and Jiang, Xingyu and Li, Hongjue and Yang, Lijun and Yao, Wen and Deng, Yue},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

## 🙏 Acknowledgements
This code builds on Stable Diffusion, diffusers, and CLIP.
