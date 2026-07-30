# UAV Image Restoration (Classical, from scratch)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

Classical (non-deep-learning) image restoration implemented from scratch with
NumPy. This project accompanies the report
*Classical Image Restoration for UAV Imaging* (see `docs/`) and is split into
two parts:

1. **Denoising** - synthesize Gaussian and salt-and-pepper (impulse) noise,
   then remove it with hand-coded **median**, **averaging**, **Gaussian**, and
   **bilateral** filters. Quality is measured with PSNR.

2. **Bandwidth-limited reconstruction** - build an image pyramid
   (512 -> 256 -> 128), transmit only a budgeted subset of patches, and
   reconstruct the full-resolution image via interpolation. A **baseline**
   (random patch selection + bilinear interpolation) is compared against
   **ours** (gradient / FFT-energy / Laplacian-pyramid patch selection plus a
   custom forward interpolation), scored by PSNR.

All numerical work is hand-written in NumPy. OpenCV is used only for image I/O
and color conversion, and Matplotlib only for visualization.

## Dataset

The original notebooks used two local test images (`lena.bmp` for denoising and
a UAV frame `4611.png` for reconstruction). Neither is redistributed here.

To keep the pipeline **reproducible with no external data**, `run_all.py`
synthesizes a deterministic 512x512 test image (fixed seed) and runs both scripts
on it with `--seed 0` — the committed results under `results/` are produced this
way and are byte-stable across runs. You can still pass your own image:

```bash
python scripts/01_denoising.py --image data/lena.bmp
python scripts/02_reconstruction.py --image data/4611.png
```

Any RGB image works. For the reconstruction demo a roughly square image
(e.g. 512x512) matches the 512 -> 256 -> 128 pyramid most cleanly.

## Structure

```
uav-image-restoration/
├── restoration/               # reusable library (NumPy implementations)
│   ├── __init__.py
│   ├── metrics.py             # psnr()
│   ├── noise.py               # g_noise() Gaussian, i_noise() impulse
│   ├── denoise.py             # median / averaging / gaussian / bilateral filters
│   ├── pyramid.py             # downsampling(), expand(), assemble()
│   ├── interpolate.py         # bilinear(), bicubic_upsampling(), cubic_weight(),
│   │                          #   restoration1/2/3()
│   └── patch_select.py        # random / gradient / frequency (FFT) / laplacian choose
├── scripts/
│   ├── 01_denoising.py        # denoising: noise synthesis + classical filters + PSNR
│   └── 02_reconstruction.py   # bandwidth-limited reconstruction (baseline vs ours + PSNR)
├── run_all.py                 # synthesize a 512x512 image + run both scripts -> results/
├── results/                   # committed artifacts: logs, figures, synthetic input
│   └── notebook_reference/    # figures/logs preserved from the original notebooks
├── docs/
│   └── Classical Image Restoration for UAV Imaging.pdf
├── data/                      # optional: your own test image(s) here (git-ignored)
├── requirements.txt
├── RESULTS.md
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.8+ and the packages in `requirements.txt`
(`numpy`, `opencv-python`, `matplotlib`).

## Usage

Reproduce all committed results on a synthetic image (no data needed):

```bash
python run_all.py        # writes results/ (logs + figure), see RESULTS.md
```

Or run the scripts individually. Denoising demo (synthesizes noise, denoises,
prints PSNR, shows figures):

```bash
python scripts/01_denoising.py --image data/lena.bmp
# save the result figure and skip the interactive window:
python scripts/01_denoising.py --image data/lena.bmp --save result.png --no-show
```

Reconstruction demo (baseline vs ours, prints a PSNR table):

```bash
python scripts/02_reconstruction.py --image data/4611.png
python scripts/02_reconstruction.py --image data/4611.png --patch-size 4
```

Using the library directly:

```python
import cv2
from restoration import g_noise, median_filter, psnr

img = cv2.cvtColor(cv2.imread("data/lena.bmp"), cv2.COLOR_BGR2RGB)
noisy = g_noise(img, 50)
clean = median_filter(noisy, 5)
print(psnr(img, clean))
```

## Notes

- The filters and interpolators are written for clarity, not speed: they use
  explicit Python loops over pixels, so they can be slow on large images.
- The `psnr` function assumes a max value of 1.0 for floating-point images and
  255.0 for integer images.
- The `psnr` implementation was duplicated across the two original notebooks;
  it is defined once here in `restoration/metrics.py`.
- The original notebooks imported `torch`, but it was never used; that import
  has been dropped in the modular code.
- The patch-selection scoring functions convert to grayscale via OpenCV's
  `COLOR_BGR2GRAY`, matching the original notebooks.
- Noise synthesis and random patch selection are stochastic; both scripts now
  accept `--seed` (used by `run_all.py`) so the committed results are
  reproducible.
- The original notebooks have been removed; their embedded figures/logs are
  preserved under `results/notebook_reference/`.
