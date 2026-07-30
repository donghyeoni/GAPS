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
[`results/notebook_reference/`](results/notebook_reference/) for provenance.
