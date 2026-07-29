#!/usr/bin/env python3
"""
Clean near-empty samples from a JWST mock lens output directory.

- Scans unified_npz (preferred) for image_final (4-band) arrays
- Flags samples with low total flux (and optional low peak)
- Optionally removes matching files across jpg_rgb, unified_npz, unified_npy
- Can inspect a single base ID and report 4-band statistics
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

BANDS = ["F115W", "F150W", "F277W", "F444W"]
SKIP_PREFIXES = ("ANNOTATED_", "REFERENCE_")


def load_thresholds(config_path: Optional[Path]) -> Tuple[float, Optional[float]]:
    if not config_path or not config_path.exists():
        return 1.0e-7, None
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    output_cfg = (cfg.get("output") or {})
    min_total = float(output_cfg.get("min_total_flux", 1.0e-7))
    min_peak = output_cfg.get("min_peak_flux")
    min_peak = float(min_peak) if min_peak is not None else None
    return min_total, min_peak


def iter_npz_samples(unified_npz: Path) -> Iterable[Path]:
    for p in sorted(unified_npz.glob("*.npz")):
        if p.name.startswith(SKIP_PREFIXES):
            continue
        yield p


def iter_npy_samples(unified_npy: Path) -> Iterable[Path]:
    for p in sorted(unified_npy.glob("*.npy")):
        if p.name.startswith(SKIP_PREFIXES):
            continue
        yield p


def compute_flux_stats(image_final: np.ndarray) -> Dict[str, float]:
    pos = np.clip(image_final, 0.0, None)
    total_flux = float(pos.sum())
    max_flux = float(image_final.max())
    mean_flux = float(image_final.mean())
    return {
        "total_flux": total_flux,
        "max_flux": max_flux,
        "mean_flux": mean_flux,
    }


def is_empty_sample(stats: Dict[str, float], min_total_flux: float, min_peak_flux: Optional[float]) -> bool:
    if stats["total_flux"] < min_total_flux:
        return True
    if min_peak_flux is not None and stats["max_flux"] < min_peak_flux:
        return True
    return False


def related_paths(base: str, output_dir: Path) -> List[Path]:
    candidates = [
        output_dir / "jpg_rgb" / f"{base}.jpg",
        output_dir / "unified_npz" / f"{base}.npz",
        output_dir / "unified_npy" / f"{base}.npy",
        output_dir / "npy" / f"{base}.npy",
    ]
    return [p for p in candidates if p.exists()]


def load_image_final(base: str, output_dir: Path) -> Optional[np.ndarray]:
    npz_path = output_dir / "unified_npz" / f"{base}.npz"
    if npz_path.exists():
        npz = np.load(npz_path)
        return npz.get("image_final")

    npy_path = output_dir / "unified_npy" / f"{base}.npy"
    if npy_path.exists():
        stacked = np.load(npy_path)
        # stacked format: steps x 5 channels (4 bands + rgb gray) concatenated
        # take last step (final): last 5 channels, then first 4 bands
        if stacked.ndim == 3 and stacked.shape[0] >= 5:
            final_block = stacked[-5:]
            return final_block[:4]

    legacy_npy = output_dir / "npy" / f"{base}.npy"
    if legacy_npy.exists():
        arr = np.load(legacy_npy)
        return arr if arr.ndim == 3 else None

    return None


def inspect_single(base: str, output_dir: Path) -> int:
    image_final = load_image_final(base, output_dir)
    if image_final is None:
        print(f"[ERROR] Missing image data for base: {base}")
        return 1

    if image_final.shape[0] != 4:
        print(f"[WARNING] Unexpected band count: {image_final.shape}")

    stats = compute_flux_stats(image_final)
    print(f"Base: {base}")
    print(f"Shape: {image_final.shape}")
    print(f"Total flux: {stats['total_flux']:.6e}")
    print(f"Max flux: {stats['max_flux']:.6e}")
    print(f"Mean flux: {stats['mean_flux']:.6e}")

    for i, band in enumerate(BANDS):
        if i >= image_final.shape[0]:
            break
        band_img = image_final[i]
        print(
            f"  {band}: sum={band_img.clip(0).sum():.6e}, "
            f"max={band_img.max():.6e}, mean={band_img.mean():.6e}"
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean empty/blank JWST samples")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory containing unified_npz/jpg_rgb (e.g., outputs/spikes_verified_fix)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config yaml to read min_total_flux (default: configs/default_config.yaml)",
    )
    parser.add_argument(
        "--min-total-flux",
        type=float,
        default=None,
        help="Override min_total_flux threshold",
    )
    parser.add_argument(
        "--min-peak-flux",
        type=float,
        default=None,
        help="Optional peak threshold (max pixel value)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; do not delete files",
    )
    parser.add_argument(
        "--check-base",
        default=None,
        help="Inspect a single base id (e.g., PRISM_lens_SF_epoch02_000021)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional CSV report output path",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        print(f"[ERROR] Output dir not found: {output_dir}")
        return 1

    if args.check_base:
        return inspect_single(args.check_base, output_dir)

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    if config_path is None:
        default_cfg = Path(__file__).resolve().parents[2] / "configs" / "default_config.yaml"
        config_path = default_cfg if default_cfg.exists() else None

    min_total_flux, cfg_peak = load_thresholds(config_path)
    min_total_flux = args.min_total_flux if args.min_total_flux is not None else min_total_flux
    min_peak_flux = args.min_peak_flux if args.min_peak_flux is not None else cfg_peak

    unified_npz = output_dir / "unified_npz"
    unified_npy = output_dir / "unified_npy"
    if not unified_npz.exists() and not unified_npy.exists():
        print(f"[ERROR] No unified_npz or unified_npy folder found in: {output_dir}")
        return 1

    report_rows = []
    to_delete = []

    if unified_npz.exists():
        sample_iter = ((p.stem, p) for p in iter_npz_samples(unified_npz))
    else:
        sample_iter = ((p.stem, p) for p in iter_npy_samples(unified_npy))

    for base, sample_path in sample_iter:
        try:
            image_final = load_image_final(base, output_dir)
            if image_final is None:
                continue
            stats = compute_flux_stats(image_final)
            empty = is_empty_sample(stats, min_total_flux, min_peak_flux)
        except Exception as e:
            print(f"[WARNING] Failed to read {sample_path}: {e}")
            continue

        report_rows.append({
            "base": base,
            "total_flux": stats["total_flux"],
            "max_flux": stats["max_flux"],
            "mean_flux": stats["mean_flux"],
            "is_empty": int(empty),
        })

        if empty:
            to_delete.append(base)

    print(f"Scanned: {len(report_rows)} samples")
    print(f"Empty: {len(to_delete)} samples")
    print(f"min_total_flux={min_total_flux:.3e}, min_peak_flux={min_peak_flux}")

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()) if report_rows else [])
            if report_rows:
                writer.writeheader()
                writer.writerows(report_rows)
        print(f"Report saved: {report_path}")

    if args.dry_run:
        print("Dry run: no files deleted")
        return 0

    deleted = 0
    for base in to_delete:
        for p in related_paths(base, output_dir):
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                print(f"[WARNING] Failed to delete {p}: {e}")

    print(f"Deleted files: {deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
