#!/usr/bin/env python3
"""Side-by-side Euclid RGB: eummy vs Trilogy (from existing unified_npz).

Usage:
  python scripts/local/compare_eummy_trilogy_rgb.py outputs/euclid_paper_physics_1arcmin_50
  python scripts/local/compare_eummy_trilogy_rgb.py outputs/euclid_paper_physics_1arcmin_50 --n 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.euclid_eummy_rgb import create_euclid_eummy_rgb, eummy_params_from_config  # noqa: E402
from src.euclid_trilogy_rgb import create_euclid_trilogy_rgb, trilogy_params_from_config  # noqa: E402


def _meta(data) -> dict:
    meta = data["metadata"]
    if isinstance(meta, np.ndarray):
        meta = meta.item()
    if isinstance(meta, str):
        meta = json.loads(meta)
    return meta if isinstance(meta, dict) else {}


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n", type=int, default=8, help="How many systems to compare")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    cfg_path = args.config or (run_dir / "run_config.yaml")
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path if not cfg_path.exists() else cfg_path
    if not cfg_path.exists():
        cfg_path = REPO / "configs" / "euclid_paper_physics_1arcmin.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    out_dir = run_dir / "jpg_rgb_eummy_vs_trilogy"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted((run_dir / "unified_npz").glob("PRISM_lens_*.npz"))[: args.n]
    if not files:
        raise SystemExit(f"No NPZ in {run_dir / 'unified_npz'}")

    e_kw = eummy_params_from_config(cfg)
    t_kw = trilogy_params_from_config(cfg)
    print(f"[CMP] eummy params: {e_kw}")
    print(f"[CMP] trilogy params: {t_kw}")
    print(f"[CMP] {len(files)} → {out_dir}")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()

    for npz in files:
        data = np.load(npz, allow_pickle=True)
        meta = _meta(data)
        bands = list(meta.get("bands") or cfg.get("bands") or [])
        images = {b: np.asarray(data["image_final"][i], dtype=np.float64) for i, b in enumerate(bands)}

        rgb_e = create_euclid_eummy_rgb(images, **e_kw)
        rgb_t = create_euclid_trilogy_rgb(images, **t_kw)

        he, we = rgb_e.shape[:2]
        canvas = Image.new("RGB", (we * 2 + 4, he + 28), (0, 0, 0))
        canvas.paste(Image.fromarray(_to_u8(rgb_e)), (0, 28))
        canvas.paste(Image.fromarray(_to_u8(rgb_t)), (we + 4, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), "eummy", fill=(255, 220, 100), font=font)
        draw.text((we + 12, 4), "Trilogy", fill=(120, 200, 255), font=font)
        draw.text((we // 2 - 40, 4), npz.stem, fill=(200, 200, 200), font=font)

        out = out_dir / f"{npz.stem}_eummy_vs_trilogy.jpg"
        canvas.save(out, quality=95)
        print(f"  wrote {out.name}")

        # Also save trilogy-only RGB for inspection
        Image.fromarray(_to_u8(rgb_t)).save(
            run_dir / "jpg_rgb" / f"{npz.stem}_trilogy.jpg", quality=95
        )

    print("[DONE]")


if __name__ == "__main__":
    main()
