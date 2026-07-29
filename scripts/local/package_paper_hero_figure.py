#!/usr/bin/env python3
"""Package a 1′ Euclid paper-hero simulation into euclid_final-style figure assets.

Selects the best candidate lens (strongest visible arcs + healthy flux) from a
simulation output directory, then writes publication-quality band JPGs, RGB,
and copies kappa/flexion diagnostic panels.

Usage:
  python scripts/local/package_paper_hero_figure.py outputs/euclid_paper_hero_1arcmin
  python scripts/local/package_paper_hero_figure.py outputs/euclid_paper_hero_1arcmin --prefix paper_hero
  python scripts/local/package_paper_hero_figure.py outputs/euclid_paper_hero_1arcmin --lens-id 3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism.core.simulator import (  # noqa: E402
    create_jwst_panel_rgb,
    create_jwst_rgb,
    normalize_for_display_astronomical,
)

EUCLID_BANDS = ["EUCLID_VIS", "EUCLID_Y", "EUCLID_J", "EUCLID_H"]
BAND_SUFFIX = {
    "EUCLID_VIS": ("vis", "VIS"),
    "EUCLID_Y": ("Y", "Y"),
    "EUCLID_J": ("J", "J"),
    "EUCLID_H": ("H", "H"),
}


def _load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "metadata" in out:
        meta = out["metadata"]
        if isinstance(meta, np.ndarray):
            meta = meta.item()
        if isinstance(meta, str):
            meta = json.loads(meta)
        out["metadata"] = meta
    return out


def _band_dict(stack: np.ndarray, bands: list[str]) -> dict[str, np.ndarray]:
    return {b: stack[i] for i, b in enumerate(bands)}


def _arc_score(npz: dict) -> float:
    """Higher = more visible lensed arcs in the composite."""
    bands = list(npz.get("metadata", {}).get("bands", EUCLID_BANDS))
    if "image_lens_sources" not in npz or "image_lens_only" not in npz:
        return float(np.sum(npz["image_final"]))

    ls = _band_dict(npz["image_lens_sources"], bands)
    lo = _band_dict(npz["image_lens_only"], bands)
    vis = ls.get("EUCLID_VIS", ls[bands[0]])
    residual = np.maximum(vis - lo.get("EUCLID_VIS", lo[bands[0]]), 0.0)

    ny, nx = residual.shape
    yy, xx = np.ogrid[:ny, :nx]
    cy, cx = ny // 2, nx // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inner = (r > 0.08 * min(ny, nx)) & (r < 0.45 * min(ny, nx))
    arc_flux = float(np.sum(residual[inner]))

    total_flux = float(np.sum(np.maximum(vis, 0)))
    theta_e = float(npz.get("metadata", {}).get("theta_E", 1.0) or 1.0)
    return arc_flux + 0.15 * total_flux + 50.0 * theta_e


def _find_candidates(sim_dir: Path) -> list[tuple[float, Path, int]]:
    npz_dir = sim_dir / "unified_npz"
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"No unified_npz in {sim_dir}")
    scored = []
    for path in sorted(npz_dir.glob("PRISM_lens_*.npz")):
        m = re.search(r"_(\d{6})\.npz$", path.name)
        if not m:
            continue
        lens_id = int(m.group(1))
        try:
            npz = _load_npz(path)
            scored.append((_arc_score(npz), path, lens_id))
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _save_band_jpg(image: np.ndarray, out_path: Path, *, quality: int = 98) -> None:
    gray = normalize_for_display_astronomical(
        image, noise_level=0.28, sat_percent=0.008, channel_name="",
    )
    arr = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path, quality=quality, optimize=True)


def package_hero(sim_dir: Path, out_dir: Path, prefix: str, lens_id: int | None) -> Path:
    sim_dir = sim_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _find_candidates(sim_dir)
    if not candidates:
        raise RuntimeError(f"No lens NPZ files found under {sim_dir}")

    if lens_id is None:
        score, npz_path, lens_id = candidates[0]
        print(f"[SELECT] lens_id={lens_id:06d}  arc_score={score:.3e}  ({npz_path.name})")
    else:
        npz_path = sim_dir / "unified_npz" / f"PRISM_lens_GR_{lens_id:06d}.npz"
        if not npz_path.exists():
            matches = list((sim_dir / "unified_npz").glob(f"PRISM_lens_*_{lens_id:06d}.npz"))
            if not matches:
                raise FileNotFoundError(f"No NPZ for lens_id={lens_id}")
            npz_path = matches[0]
        print(f"[SELECT] manual lens_id={lens_id:06d}  ({npz_path.name})")

    npz = _load_npz(npz_path)
    bands = list(npz.get("metadata", {}).get("bands", EUCLID_BANDS))
    final = _band_dict(npz["image_final"], bands)

    arc_images = None
    if "image_lens_sources" in npz and "image_lens_only" in npz:
        arc_images = {
            "lens_sources": _band_dict(npz["image_lens_sources"], bands),
            "lens_only": _band_dict(npz["image_lens_only"], bands),
        }

    rgb = create_jwst_rgb(final, bands=bands, telescope="euclid", arc_images=arc_images)
    panel = create_jwst_panel_rgb(final, bands=bands, telescope="euclid", arc_images=arc_images)

    for band in bands:
        low, up = BAND_SUFFIX[band]
        np.save(out_dir / f"{prefix}_{up}.npy", final[band].astype(np.float32))
        _save_band_jpg(final[band], out_dir / f"{prefix}_{low}.jpg")

    if rgb is not None:
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
            out_dir / f"{prefix}_rgb.jpg", quality=98, optimize=True,
        )
    if panel is not None:
        Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)).save(
            out_dir / f"{prefix}_panel.jpg", quality=98, optimize=True,
        )

    kid = f"{lens_id:06d}"
    kappa_dir = sim_dir / "kappa_maps"
    for src_name, dst_name in [
        (f"{kid}_kappa_panel.jpg", f"lens_{prefix}_kappa_panel.jpg"),
        (f"{kid}_flexion_panel.jpg", f"lens_{prefix}_flexion_panel.jpg"),
        (f"{kid}_kappa.jpg", f"lens_{prefix}_kappa.jpg"),
        (f"{kid}_kappa.npy", f"lens_{prefix}_kappa.npy"),
        (f"{kid}_kappa_data.npz", f"lens_{prefix}_kappa_data.npz"),
        (f"{kid}_ext1.0arcmin_kappa_panel.jpg", f"lens_{prefix}_ext1arcmin_kappa_panel.jpg"),
        (f"{kid}_ext1.0arcmin_flexion_panel.jpg", f"lens_{prefix}_ext1arcmin_flexion_panel.jpg"),
    ]:
        src = kappa_dir / src_name
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)

    # Mirror euclid_final naming (final_* + lens_*)
    for name in out_dir.glob(f"{prefix}_*"):
        alt = name.name.replace(f"{prefix}_", "final_", 1)
        shutil.copy2(name, out_dir / alt)

    meta_path = out_dir / f"{prefix}_README.json"
    meta_path.write_text(json.dumps({
        "source_npz": str(npz_path),
        "lens_id": lens_id,
        "metadata": npz.get("metadata", {}),
        "bands": bands,
        "fov_arcmin": 1.0,
        "pixel_scale_arcsec": 0.10,
        "n_pixels": int(final[bands[0]].shape[0]),
    }, indent=2, default=str))

    print(f"[DONE] figures -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path, help="Simulation output directory")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Package destination (default: <sim_dir>/paper_figures)")
    parser.add_argument("--prefix", default="hero", help="Filename prefix (default: hero)")
    parser.add_argument("--lens-id", type=int, default=None, help="Force a specific lens index")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.sim_dir / "paper_figures")
    package_hero(args.sim_dir, out_dir, args.prefix, args.lens_id)


if __name__ == "__main__":
    main()
