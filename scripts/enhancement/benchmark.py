"""Full benchmark: every selection x fill combination, one PSNR table.

``random + bilinear`` is the baseline; every other combination is ours.
Also renders the README figures: the 128 -> 256 -> 512 progression
(baseline vs the best ours) and the selected-patches comparison.

Note: everything is hand-written NumPy with per-pixel Python loops — the full
grid takes tens of minutes on a typical machine. For a single combination use
the per-method scripts in this folder.

Usage
-----
    python scripts/enhancement/benchmark.py --seed 0 --save results/reconstruction_progression.png --save-patches results/patch_selection.png
"""

from _common import (
    BASELINE,
    FILLS,
    SELECTIONS,
    build_pyramid,
    load_rgb,
    parse_args,
    run_pipeline,
)


def main():
    args = parse_args("Bandwidth-limited reconstruction benchmark (full grid).",
                      with_figures=True)
    image_original = load_rgb(args.image)
    image_512, image_256, image_128, image_high_up = build_pyramid(image_original)

    print(f"Pyramid sizes: {image_512.shape} -> {image_256.shape} -> {image_128.shape}")
    print(f"Patch size: {args.patch_size}\n")

    results = {}
    for sel_name in SELECTIONS:
        for fill_mode in FILLS:
            results[(sel_name, fill_mode)] = run_pipeline(
                sel_name, fill_mode, image_512, image_256, image_128,
                image_high_up, args.patch_size, args.seed)
            r = results[(sel_name, fill_mode)]
            tag = " (baseline)" if (sel_name, fill_mode) == BASELINE else ""
            print(f"  {sel_name:<10} + {fill_mode:<9} : "
                  f"256 {r['p256']:6.2f}  512 {r['p512']:6.2f}{tag}")

    # Summary table (256 / 512 per cell).
    print("\nPSNR (dB) -- selection x fill, '256 / 512', higher is better:")
    print(f"  {'':<11}" + "".join(f"{f:>16}" for f in FILLS))
    for sel_name in SELECTIONS:
        row = f"  {sel_name:<11}"
        for fill_mode in FILLS:
            r = results[(sel_name, fill_mode)]
            row += f"{r['p256']:7.2f} /{r['p512']:6.2f} "
        print(row)

    # Best non-baseline combination by 512-level PSNR.
    best_key = max((k for k in results if k != BASELINE),
                   key=lambda k: results[k]["p512"])
    base = results[BASELINE]
    best = results[best_key]
    print(f"\nBaseline (random + bilinear): 256 {base['p256']:.2f}  512 {base['p512']:.2f}")
    print(f"Best ours ({best_key[0]} + {best_key[1]}): "
          f"256 {best['p256']:.2f}  512 {best['p512']:.2f}")

    if args.save or args.save_patches:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    if args.save:
        # 128 -> 256 -> 512 progression, one row per method.
        rows = [
            ("Baseline (random + bilinear)", base),
            (f"Ours ({best_key[0]} + {best_key[1]})", best),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.4),
                                 gridspec_kw={"width_ratios": [1, 2, 4]})
        for r, (name, res) in enumerate(rows):
            panels = [
                (image_128, f"{name}\n128 base (transmitted in full)"),
                (res["re_256"], f"256 reconstruction\n{res['p256']:.2f} dB"),
                (res["re_512"], f"512 reconstruction\n{res['p512']:.2f} dB"),
            ]
            for c, (img, title) in enumerate(panels):
                axes[r, c].imshow(img)
                axes[r, c].set_title(title, fontsize=10)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")

    if args.save_patches:
        rows = [
            ("Baseline (random)", base),
            (f"Ours ({best_key[0]})", best),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(9, 9.4))
        for r, (name, res) in enumerate(rows):
            axes[r, 0].imshow(res["sel_256"])
            axes[r, 0].set_title(f"{name}\n256 level (1/4 of pixels)", fontsize=11)
            axes[r, 0].axis("off")
            axes[r, 1].imshow(res["sel_512"])
            axes[r, 1].set_title(f"{name}\n512 level (1/16 of pixels)", fontsize=11)
            axes[r, 1].axis("off")
        fig.tight_layout()
        fig.savefig(args.save_patches, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save_patches}")


if __name__ == "__main__":
    main()
