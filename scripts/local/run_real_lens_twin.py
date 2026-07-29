#!/usr/bin/env python3
"""Simulate a realistic twin of real COSMOS-Web lens COSJ100011+023531.

Produces JWST (9″ stamp, matches real_jwst_rgb.jpg FOV) and/or Euclid (1′)
mocks with binary SIE, θ_E≈2.25″, z≈0.706, then builds a side-by-side
comparison with the real JWST cutout.

Usage:
  python scripts/local/run_real_lens_twin.py --telescope jwst
  python scripts/local/run_real_lens_twin.py --telescope euclid
  python scripts/local/run_real_lens_twin.py --telescope both
  python scripts/local/run_real_lens_twin.py --telescope jwst --compare-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
REAL_RGB = REPO / "outputs" / "cosmos_field_euclid" / "real_jwst_rgb.jpg"
REAL_REF = "COSJ100011+023531"


def _run_sim(telescope: str, n: int, seed: int) -> Path:
    if telescope == "jwst":
        cfg = REPO / "configs" / "twin_cosj100011_jwst.yaml"
        out = REPO / "outputs" / "twin_cosj100011_jwst"
        n_field = 8
    else:
        cfg = REPO / "configs" / "twin_cosj100011_euclid.yaml"
        out = REPO / "outputs" / "twin_cosj100011_euclid"
        n_field = 40

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "prism.core.simulator",
        "--config", str(cfg),
        "--cosmos_catalog", "data/cosmos_web_lens_structural_properties.csv",
        "--lens_analysis_catalog", "data/lens_analysis_catalog.csv",
        "--merged_field_catalog", "data/merged_lens_field_catalog.csv",
        "--output_dir", str(out),
        "--n_lenses", str(n),
        "--n_non_lenses", "0",
        "--n_field_max", str(n_field),
        "--variations_per_base", "1",
        "--seed", str(seed),
        "--add_artifacts",
        "--save_intermediate",
        "--no_date_suffix",
    ]
    log = out / "run.log"
    print(f"[RUN] {telescope} → {out}")
    with open(log, "w") as fh:
        subprocess.check_call(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
    print(f"[DONE] {telescope}  log={log}")
    return out


def _load_rgb_from_npz(npz_path: Path, telescope: str) -> np.ndarray:
    sys.path.insert(0, str(REPO))
    import yaml
    import prism.core.simulator as sim

    cfg_name = "twin_cosj100011_jwst.yaml" if telescope == "jwst" else "twin_cosj100011_euclid.yaml"
    cfg = yaml.safe_load((REPO / "configs" / cfg_name).read_text())
    sim.CONFIG.update(cfg)

    data = np.load(npz_path, allow_pickle=True)
    meta = data["metadata"]
    if isinstance(meta, np.ndarray):
        meta = meta.item()
    if isinstance(meta, str):
        meta = json.loads(meta)
    bands = list(meta.get("bands", cfg["bands"]))
    images = {b: data["image_final"][i] for i, b in enumerate(bands)}
    arc = None
    if "image_lens_sources" in data.files:
        arc = {
            "lens_sources": {b: data["image_lens_sources"][i] for i, b in enumerate(bands)},
            "lens_only": {b: data["image_lens_only"][i] for i, b in enumerate(bands)},
        }
    rgb = sim.create_jwst_rgb(images, bands=bands, telescope=telescope, arc_images=arc)
    return np.clip(rgb, 0, 1)


def _arc_score(npz_path: Path) -> float:
    d = np.load(npz_path, allow_pickle=True)
    if "image_lens_sources" not in d.files:
        return float(np.sum(d["image_final"]))
    res = np.maximum(d["image_lens_sources"][0] - d["image_lens_only"][0], 0)
    return float(res.sum())


def _pick_best(sim_dir: Path) -> Path:
    scored = []
    for p in sorted((sim_dir / "unified_npz").glob("PRISM_lens_*.npz")):
        scored.append((_arc_score(p), p))
    scored.sort(reverse=True)
    return scored[0][1]


def _to_u8(rgb: np.ndarray, size: int | None = None) -> Image.Image:
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    im = Image.fromarray(arr)
    if size is not None and im.size[0] != size:
        im = im.resize((size, size), Image.Resampling.LANCZOS)
    return im


def make_comparison(jwst_dir: Path | None, euclid_dir: Path | None, out_path: Path) -> Path:
    panels = []
    labels = []

    if REAL_RGB.exists():
        panels.append(Image.open(REAL_RGB).convert("RGB").resize((360, 360), Image.Resampling.LANCZOS))
        labels.append(f"REAL JWST\n{REAL_REF}\nF444/F277/F115 · 9″")
    if jwst_dir and (jwst_dir / "unified_npz").exists():
        best = _pick_best(jwst_dir)
        rgb = _load_rgb_from_npz(best, "jwst")
        panels.append(_to_u8(rgb, 360))
        labels.append(f"SIM JWST twin\n{best.stem}\nbinary SIE · θ_E≈2.25″")
        _to_u8(rgb).save(jwst_dir / "best_rgb.jpg", quality=95)
    if euclid_dir and (euclid_dir / "unified_npz").exists():
        best = _pick_best(euclid_dir)
        rgb = _load_rgb_from_npz(best, "euclid")
        # Show central 15″ crop (~150 px of 600) to match real stamp scale
        h, w = rgb.shape[:2]
        half = int(7.5 / 0.10)  # 7.5" radius → 15" FOV
        cy, cx = h // 2, w // 2
        crop = rgb[cy - half: cy + half, cx - half: cx + half]
        panels.append(_to_u8(crop, 360))
        labels.append(f"SIM Euclid twin\n{best.stem}\n15″ crop of 1′ field")
        _to_u8(rgb).save(euclid_dir / "best_rgb_full1arcmin.jpg", quality=95)
        _to_u8(crop).save(euclid_dir / "best_rgb_15arcsec_crop.jpg", quality=95)

    if not panels:
        raise SystemExit("Nothing to compare")

    label_h = 70
    gap = 8
    W = len(panels) * 360 + (len(panels) - 1) * gap
    H = 360 + label_h + 40
    canvas = Image.new("RGB", (W, H), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    draw.text((10, 8), "Real COSMOS-Web JWST vs realistic twins (binary SIE, θ_E≈2.25″, z≈0.7)", fill=(230, 230, 230), font=font)

    for i, (im, lab) in enumerate(zip(panels, labels)):
        x = i * (360 + gap)
        canvas.paste(im, (x, 36))
        for j, line in enumerate(lab.split("\n")):
            draw.text((x + 8, 36 + 360 + 4 + j * 16), line, fill=(200, 200, 210), font=font_sm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    print(f"[COMPARE] {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telescope", choices=["jwst", "euclid", "both"], default="both")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=100011)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    jwst_dir = REPO / "outputs" / "twin_cosj100011_jwst"
    euclid_dir = REPO / "outputs" / "twin_cosj100011_euclid"

    if not args.compare_only:
        if args.telescope in ("jwst", "both"):
            jwst_dir = _run_sim("jwst", args.n, args.seed)
        if args.telescope in ("euclid", "both"):
            euclid_dir = _run_sim("euclid", args.n, args.seed + 1)

    out = REPO / "outputs" / "twin_cosj100011_comparison" / "real_vs_twins.jpg"
    make_comparison(
        jwst_dir if jwst_dir.exists() else None,
        euclid_dir if euclid_dir.exists() else None,
        out,
    )


if __name__ == "__main__":
    main()
