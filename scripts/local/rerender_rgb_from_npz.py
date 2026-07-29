#!/usr/bin/env python3
"""Re-render jpg_rgb panels from existing unified_npz (no re-simulation).

Usage:
  python scripts/local/rerender_rgb_from_npz.py outputs/euclid_paper_physics_1arcmin_50
  python scripts/local/rerender_rgb_from_npz.py outputs/euclid_paper_physics_1arcmin_50 \\
      --config configs/euclid_paper_physics_1arcmin.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.jwst_lens_simulator as sim  # noqa: E402


def _load_meta(data) -> dict:
    meta = data["metadata"]
    if isinstance(meta, np.ndarray):
        meta = meta.item()
    if isinstance(meta, str):
        meta = json.loads(meta)
    return meta if isinstance(meta, dict) else {}


def rerender_one(npz_path: Path, out_jpg: Path, telescope: str) -> None:
    data = np.load(npz_path, allow_pickle=True)
    meta = _load_meta(data)
    bands = list(meta.get("bands") or sim.CONFIG.get("bands") or [])
    if not bands:
        raise RuntimeError(f"No bands in {npz_path.name}")

    images = {b: np.asarray(data["image_final"][i], dtype=np.float64) for i, b in enumerate(bands)}

    arc_images = None
    if "image_lens_sources" in data.files and "image_lens_only" in data.files:
        arc_images = {
            "lens_sources": {b: data["image_lens_sources"][i] for i, b in enumerate(bands)},
            "lens_only": {b: data["image_lens_only"][i] for i, b in enumerate(bands)},
        }

    panel = sim.create_jwst_panel_rgb(
        images, bands=bands, telescope=telescope, arc_images=arc_images,
    )
    if panel is None:
        raise RuntimeError(f"panel RGB failed for {npz_path.name}")

    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)).save(
        out_jpg, quality=95, optimize=True,
    )

    # Also write RGB-only for quick inspection
    rgb = sim.create_jwst_rgb(images, bands=bands, telescope=telescope, arc_images=arc_images)
    if rgb is not None:
        rgb_only = out_jpg.with_name(out_jpg.stem + "_RGBonly.jpg")
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
            rgb_only, quality=95, optimize=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="Simulation output directory")
    ap.add_argument(
        "--config", type=Path, default=None,
        help="YAML config (default: run_dir/run_config.yaml or physics config)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Only first N files (debug)")
    args = ap.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    cfg_path = args.config
    if cfg_path is None:
        cand = run_dir / "run_config.yaml"
        cfg_path = cand if cand.exists() else REPO / "configs" / "euclid_paper_physics_1arcmin.yaml"
    elif not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path

    cfg = yaml.safe_load(cfg_path.read_text())
    sim.CONFIG.update(cfg)
    telescope = str(cfg.get("telescope", "euclid")).lower()

    npz_dir = run_dir / "unified_npz"
    out_dir = run_dir / "jpg_rgb"
    files = sorted(npz_dir.glob("PRISM_lens_*.npz"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No NPZ in {npz_dir}")

    print(f"[RGB] config={cfg_path}")
    print(f"[RGB] rgb params={cfg.get('output', {}).get('rgb', {})}")
    print(f"[RGB] re-rendering {len(files)} → {out_dir}")

    for i, npz in enumerate(files, 1):
        out = out_dir / f"{npz.stem}.jpg"
        rerender_one(npz, out, telescope)
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {npz.name}")

    print("[DONE]")


if __name__ == "__main__":
    main()
