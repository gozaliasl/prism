"""
Load GalaxyGenius mock JWST panels from published combined figures.

Zhou et al. (2025, A&A 700, A120; arXiv:2506.15060) Figure 6 (TNG) and
Figure 10b (EAGLE) show SKIRT radiative-transfer + mock-observation images.
We crop individual band panels from the GalaxyGenius repository assets.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_CATALOG = PACKAGE_DIR / "galaxygenius_catalog.yaml"
DEFAULT_FIGURES_DIR = (
    REPO_ROOT / "analysis" / "galaxy_morphology" / "reports" / "figures"
)

BAND_COLUMNS = ["F070W", "F150W", "F200W", "F182M", "F356W", "F444W"]

# Default JWST NIRCam RGB mapping (short → long wavelength) for combined figures.
# GalaxyGenius uses F070W/F150W/F2100W for true-color RGB; F444W substitutes here.
DEFAULT_RGB_BAND_COLS = (0, 1, 5)  # B=F070W, G=F150W, R=F444W
DEFAULT_RGB_BANDS = tuple(BAND_COLUMNS[i] for i in DEFAULT_RGB_BAND_COLS)

# Crop boxes (x0, y0, x1, y1) tuned to GalaxyGenius combined PNG layouts.
_ASSET_GRIDS: dict[str, dict[str, Any]] = {
    "tng_jwst_combined": {
        "filename": "galaxygenius_JWST_combined.png",
        "url": (
            "https://raw.githubusercontent.com/xczhou-astro/galaxyGenius/"
            "main/assets/JWST_combined.png"
        ),
        "rows": [(40, 265), (345, 575), (680, 875), (960, 1085)],
        "cols": [
            (78, 298), (320, 608), (630, 918), (940, 1228),
            (1250, 1538), (1560, 1848),
        ],
        "row_meta": [
            {"subhalo_id": 0, "simulation": "TNG100", "snapshot": 94, "redshift": 0.06},
            {"subhalo_id": 31, "simulation": "TNG100", "snapshot": 94, "redshift": 0.06},
            {
                "subhalo_id": 253881, "simulation": "TNG100", "snapshot": 94,
                "redshift": 0.06, "view": "face-on",
            },
            {
                "subhalo_id": 253881, "simulation": "TNG100", "snapshot": 94,
                "redshift": 0.06, "view": "edge-on",
            },
        ],
    },
    "eagle_jwst_combined": {
        "filename": "galaxygenius_JWST_eagle_combined.png",
        "url": (
            "https://raw.githubusercontent.com/xczhou-astro/galaxyGenius/"
            "main/assets/JWST_eagle_combined.png"
        ),
        "rows": [(40, 280), (345, 575)],
        "cols": [
            (78, 308), (330, 578), (600, 878), (900, 1168),
            (1190, 1478), (1500, 1788),
        ],
        "row_meta": [
            {
                "subhalo_id": 20695120, "simulation": "EAGLE", "snapshot": 27,
                "redshift": 0.101,
            },
            {
                "subhalo_id": 21109761, "simulation": "EAGLE", "snapshot": 27,
                "redshift": 0.101,
            },
        ],
    },
}


def load_catalog(catalog_path: Path | None = None) -> dict[str, Any]:
    path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG
    with open(path) as f:
        return yaml.safe_load(f)


def morphology_order(catalog: dict[str, Any] | None = None) -> list[str]:
    cat = catalog or load_catalog()
    return list(cat["morphologies"].keys())


def ensure_asset(asset_key: str, figures_dir: Path | None = None) -> Path:
    """Return local path to a combined figure, downloading if missing."""
    import urllib.request

    grid = _ASSET_GRIDS[asset_key]
    out_dir = Path(figures_dir) if figures_dir else DEFAULT_FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / grid["filename"]
    if not path.exists():
        urllib.request.urlretrieve(grid["url"], path)
    return path


@lru_cache(maxsize=8)
def _load_rgb(asset_key: str, figures_dir: str) -> np.ndarray:
    path = ensure_asset(asset_key, Path(figures_dir))
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def extract_panel(
    asset_key: str,
    row: int,
    band_col: int,
    *,
    figures_dir: Path | None = None,
) -> np.ndarray:
    """Return a grayscale panel in linear-ish display units (0–255)."""
    grid = _ASSET_GRIDS[asset_key]
    rows = grid["rows"]
    cols = grid["cols"]
    if not (0 <= row < len(rows)):
        raise IndexError(f"row {row} out of range for {asset_key} ({len(rows)} rows)")
    if not (0 <= band_col < len(cols)):
        raise IndexError(
            f"band_col {band_col} out of range for {asset_key} ({len(cols)} cols)"
        )

    fig_dir = str(Path(figures_dir) if figures_dir else DEFAULT_FIGURES_DIR)
    rgb = _load_rgb(asset_key, fig_dir)
    cx0, cx1 = cols[band_col]
    ry0, ry1 = rows[row]
    patch = rgb[ry0:ry1, cx0:cx1]
    # Luminance from displayed RGB (paper figures are log-stretched electron counts).
    gray = 0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]
    return gray


def load_morphology_panel(
    morph_type: str,
    *,
    catalog_path: Path | None = None,
    figures_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load one morphology panel and metadata from the GalaxyGenius catalog.

    Returns
    -------
    image : ndarray
        Grayscale panel (display units, already mock-observed in the paper assets).
    meta : dict
        Subhalo ID, band, simulation, etc.
    """
    catalog = load_catalog(catalog_path)
    entry = catalog["morphologies"][morph_type]
    asset_key = entry["asset"]
    row = int(entry["row"])
    band_col = int(entry["band_col"])
    band = entry.get("band", BAND_COLUMNS[band_col])

    image = extract_panel(asset_key, row, band_col, figures_dir=figures_dir)

    grid = _ASSET_GRIDS[asset_key]
    row_meta = dict(grid["row_meta"][row])
    meta = {
        "morph_type": morph_type,
        "label": entry.get("label", morph_type),
        "asset": asset_key,
        "row": row,
        "band_col": band_col,
        "band": band,
        "notes": entry.get("notes", ""),
        **row_meta,
        **{k: entry[k] for k in ("subhalo_id", "simulation", "snapshot", "redshift", "view")
           if k in entry},
    }
    return image, meta


def _align_band_planes(planes: list[np.ndarray]) -> list[np.ndarray]:
    """Center-crop multiple band planes to their common field of view."""
    h = min(p.shape[0] for p in planes)
    w = min(p.shape[1] for p in planes)
    aligned = []
    for plane in planes:
        ph, pw = plane.shape
        y0 = (ph - h) // 2
        x0 = (pw - w) // 2
        aligned.append(plane[y0:y0 + h, x0:x0 + w])
    return aligned


def load_morphology_rgb(
    morph_type: str,
    *,
    rgb_band_cols: tuple[int, int, int] = DEFAULT_RGB_BAND_COLS,
    catalog_path: Path | None = None,
    figures_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load an RGB stack for one morphology from three JWST bands in the same row.

    Returns
    -------
    rgb : ndarray, shape (H, W, 3)
        Channel order is (R, G, B) with bands given by ``rgb_band_cols``.
    meta : dict
        Morphology metadata plus ``rgb_bands`` list.
    """
    catalog = load_catalog(catalog_path)
    entry = catalog["morphologies"][morph_type]
    asset_key = entry["asset"]
    row = int(entry["row"])

    b_col, g_col, r_col = rgb_band_cols
    planes = _align_band_planes([
        extract_panel(asset_key, row, r_col, figures_dir=figures_dir),
        extract_panel(asset_key, row, g_col, figures_dir=figures_dir),
        extract_panel(asset_key, row, b_col, figures_dir=figures_dir),
    ])
    rgb = np.stack(planes, axis=-1)

    grid = _ASSET_GRIDS[asset_key]
    row_meta = dict(grid["row_meta"][row])
    rgb_bands = [BAND_COLUMNS[r_col], BAND_COLUMNS[g_col], BAND_COLUMNS[b_col]]
    meta = {
        "morph_type": morph_type,
        "label": entry.get("label", morph_type),
        "asset": asset_key,
        "row": row,
        "rgb_bands": rgb_bands,
        "notes": entry.get("notes", ""),
        **row_meta,
        **{k: entry[k] for k in ("subhalo_id", "simulation", "snapshot", "redshift", "view")
           if k in entry},
    }
    return rgb, meta


def stretch_rgb(
    rgb: np.ndarray,
    *,
    percentile: float = 99.5,
    gamma: float = 0.85,
) -> np.ndarray:
    """Per-channel percentile stretch to [0, 1] for display."""
    out = np.zeros_like(rgb, dtype=np.float64)
    for i in range(3):
        plane = np.clip(rgb[:, :, i], 0, None)
        scale = float(np.percentile(plane, percentile))
        if scale <= 0:
            scale = 1.0
        out[:, :, i] = np.clip(plane / scale, 0, 1) ** gamma
    return out


def panel_annotation(meta: dict[str, Any]) -> str:
    """Compact annotation string for showcase panels."""
    parts = [
        f"{meta.get('simulation', '?')} subhalo {meta.get('subhalo_id', '?')}",
        f"z={meta.get('redshift', '?')}",
    ]
    if meta.get("rgb_bands"):
        r, g, b = meta["rgb_bands"]
        parts.append(f"RGB: R={r}, G={g}, B={b}")
    else:
        parts.append(str(meta.get("band", "F150W")))
    if meta.get("view"):
        parts.append(str(meta["view"]))
    parts.append("GalaxyGenius + SKIRT")
    return "\n".join(parts)
