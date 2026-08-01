# Results

Produced by a single reproducible command (no external data — a deterministic
512x512 synthetic image is generated, both scripts run with `--seed 0`):

```bash
python run_all.py
```

Artifacts under [`results/`](results/). The synthetic input is
[`results/input_synthetic.png`](results/input_synthetic.png).

## 1. Denoising (PSNR, dB — higher is better)

Classical filters applied to synthesized noise. Log:
[`results/01_denoising.log`](results/01_denoising.log).

| Filter | Noise | PSNR (dB) |
| --- | --- | --- |
| Median | Gaussian σ=50 | 15.60 |
| Averaging | impulse p=0.10 | 22.64 |
| Gaussian | impulse p=0.10 | 22.64 |
| Bilateral | Gaussian σ=50 | 17.11 |

![denoising](results/01_denoising.png)

## 2. Bandwidth-limited reconstruction (PSNR, dB)

Image pyramid 512 → 256 → 128 with a budgeted subset of patches transmitted per
level, then reconstructed. Log:
[`results/02_reconstruction.log`](results/02_reconstruction.log).

| Pipeline | 256 level | 512 level |
| --- | --- | --- |
| Baseline (random patches + bilinear) | 35.07 | 27.82 |
| **Ours** (gradient patches + custom fill) | **40.89** | **28.86** |

The gradient-based patch selection with the custom forward interpolation beats
the random-patch bilinear baseline at both pyramid levels — most clearly at the
256 level (+5.8 dB).

## Original notebook figures

Figures/logs embedded in the original notebooks are preserved under
[`results/notebook_reference/`](results/notebook_reference/) for provenance —
`Img_Project_1__*` (4 figures: noise synthesis, and per-filter PSNR
comparisons) and `Img_Project_2__*` (7 figures: the 512/256/128 pyramid, the
hand-written bicubic interpolation, entropy-ranked top-64 grid selection and
the reconstructions), alongside the two run logs. These
are the *original* runs on the project's own imagery; the tables above come
from the reproducible synthetic pipeline, so the numbers are not directly
comparable.
