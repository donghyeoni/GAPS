# UAV Image Restoration (Classical, from scratch)

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

The experiments use two single test images, loaded from local paths:

- `lena.bmp`  - used by the denoising part.
- `4611.png`  - a UAV/aerial image used by the reconstruction part.

**These images are not included in this repository.** Drop your own image into
the `data/` folder and point the scripts at it with `--image`, for example:

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
│   ├── 01_denoising.py        # reproduces Notebook 1
│   └── 02_reconstruction.py   # reproduces Notebook 2 (baseline vs ours + PSNR)
├── notebooks/                 # original Colab notebooks (unmodified)
│   ├── Img_Project_1.ipynb
│   └── Img_Project_2.ipynb
├── docs/
│   └── Classical Image Restoration for UAV Imaging.pdf
├── data/                      # put your own test image(s) here (git-ignored)
├── requirements.txt
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

Denoising demo (synthesizes noise, denoises, prints PSNR, shows figures):

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
- Results depend on NumPy's random state (noise synthesis and random patch
  selection are stochastic); set a seed if you need reproducible numbers.
