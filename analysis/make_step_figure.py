#!/usr/bin/env python3
"""
Create a paper figure showing step-by-step outputs per lens system.

Rows (default order):
1) lens_only
2) sources_only
3) lens_sources
4) field_only
5) final

Each row is shown as an RGB composite. The script auto-selects an
"interesting" lens (highest total flux) unless --sample-id is provided.

Usage:
  python analysis/make_step_figure.py \
      --input-dir outputs/custom_YYYYMMDD_HHMMSS \
      --output figures/step_outputs_example.png

Optional:
  --sample-id 123 (uses cosmos_lens_000123.npz)
  --ncols 1 (default 1)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

STEPS_DEFAULT = ["lens_only", "sources_only", "lens_sources", "field_only", "final"]
BANDS = ["F115W", "F150W", "F277W", "F444W"]


def _make_rgb_from_stack(stack_4, p=99.5):
    """Create an RGB image from a 4-band stack (F115W, F150W, F277W, F444W)."""
    f115, f150, f277, f444 = stack_4
    r = f444
    g = 0.5 * (f150 + f277)
    b = f115
    rgb = np.stack([r, g, b], axis=-1)
    vmax = np.percentile(rgb, p)
    if vmax <= 0:
        return np.zeros_like(rgb, dtype=np.float32)
    rgb = np.clip(rgb / vmax, 0, 1)
    return rgb.astype(np.float32)


def _load_npz(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as data:
        result = {k: data[k] for k in data.files}
    if "metadata" in result:
        try:
            result["metadata"] = json.loads(str(result["metadata"]))
        except Exception:
            pass
    return result


def _select_interesting_lens(unified_dir: Path):
    candidates = sorted(unified_dir.glob("cosmos_lens_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No lens samples found in {unified_dir}")

    best_path = None
    best_flux = -np.inf
    for p in candidates:
        data = _load_npz(p)
        if "image_final" not in data:
            continue
        total_flux = float(np.sum(data["image_final"]))
        if total_flux > best_flux:
            best_flux = total_flux
            best_path = p
    if best_path is None:
        raise RuntimeError("No valid lens samples with image_final found")
    return best_path


def _get_step_stack(data, step):
    if step == "final":
        return data.get("image_final")
    key = f"image_{step}"
    return data.get(key)


def main():
    parser = argparse.ArgumentParser(description="Generate step-output figure")
    parser.add_argument("--input-dir", required=True, help="Output run directory")
    parser.add_argument("--output", required=True, help="Output figure path")
    parser.add_argument("--sample-id", type=int, default=None, help="Lens id (integer)")
    parser.add_argument("--order", nargs="+", default=STEPS_DEFAULT, help="Step order")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    unified_dir = input_dir / "unified_npz"
    if not unified_dir.exists():
        raise FileNotFoundError(f"Missing unified_npz directory: {unified_dir}")

    if args.sample_id is not None:
        npz_path = unified_dir / f"cosmos_lens_{args.sample_id:06d}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Sample not found: {npz_path}")
    else:
        npz_path = _select_interesting_lens(unified_dir)

    data = _load_npz(npz_path)

    rows = []
    labels = []
    for step in args.order:
        stack = _get_step_stack(data, step)
        if stack is None:
            continue
        rgb = _make_rgb_from_stack(stack)
        rows.append(rgb)
        labels.append(step.replace("_", " "))

    if not rows:
        raise RuntimeError("No step images available in this sample")

    fig, axes = plt.subplots(len(rows), 1, figsize=(6, 2.4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for ax, img, label in zip(axes, rows, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(npz_path.name.replace(".npz", ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
