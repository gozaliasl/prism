#!/usr/bin/env python3
"""Build final_B_fullmaps-style 8-panel Euclid figures from a simulation run.

Layout (matches outputs/euclid_final/final_B_fullmaps.jpg):
  Top:    colour (H/J/VIS) | VIS | NISP-Y | NISP-H
  Bottom: κ | |μ| | |F| | |G|
  All panels 60″×60″ with 5″ scale bar and orange dashed θ_E circle.

Usage:
  python scripts/local/make_euclid_fullmaps_figure.py outputs/euclid_paper_finalB_style
  python scripts/local/make_euclid_fullmaps_figure.py outputs/euclid_paper_finalB_style --top 6
  python scripts/local/make_euclid_fullmaps_figure.py outputs/euclid_paper_finalB_style --lens-id 7
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prism.core.simulator import (  # noqa: E402
    create_jwst_rgb,
    normalize_for_display_astronomical,
)

EUCLID_BANDS = ["EUCLID_VIS", "EUCLID_Y", "EUCLID_J", "EUCLID_H"]


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


def _asinh_stretch(im: np.ndarray, lo_pct=1.0, hi_pct=99.7, soft=0.15) -> np.ndarray:
    lo, hi = np.percentile(im, [lo_pct, hi_pct])
    x = np.clip((im - lo) / max(hi - lo, 1e-12), 0, 1)
    return np.arcsinh(x / soft) / np.arcsinh(1.0 / soft)


def _gray_rgb(im: np.ndarray) -> np.ndarray:
    g = _asinh_stretch(im)
    return np.stack([g, g, g], axis=-1)


def _cmap_rgb(im: np.ndarray, cmap_name: str, lo_pct=1.0, hi_pct=99.7) -> np.ndarray:
    g = _asinh_stretch(im, lo_pct=lo_pct, hi_pct=hi_pct)
    cmap = plt.get_cmap(cmap_name)
    return cmap(g)[..., :3]


def _map_stretch(im: np.ndarray, cmap_name: str, vmin=None, vmax=None, log=False) -> np.ndarray:
    x = np.asarray(im, dtype=np.float64)
    if log:
        x = np.log10(np.clip(np.abs(x), 1e-3, None))
    if vmin is None:
        vmin = float(np.percentile(x, 1))
    if vmax is None:
        vmax = float(np.percentile(x, 99.5))
    g = np.clip((x - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    return plt.get_cmap(cmap_name)(g)[..., :3]


def _ring_score(npz: dict) -> float:
    meta = npz.get("metadata", {})
    te = float(meta.get("theta_E", 1.0) or 1.0)
    if "image_lens_sources" not in npz or "image_lens_only" not in npz:
        return te
    res = np.maximum(npz["image_lens_sources"][0] - npz["image_lens_only"][0], 0.0)
    ny, nx = res.shape
    yy, xx = np.ogrid[:ny, :nx]
    r = np.sqrt((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) * 0.10
    annulus = (r > 0.55 * te) & (r < 1.45 * te)
    return float(res[annulus].sum()) + 40.0 * te


def _find_candidates(sim_dir: Path) -> list[tuple[float, Path, int, str]]:
    scored = []
    for path in sorted((sim_dir / "unified_npz").glob("PRISM_lens_*.npz")):
        m = re.search(r"PRISM_lens_([A-Z]+)_(\d{6})\.npz$", path.name)
        if not m:
            continue
        klass, lid = m.group(1), int(m.group(2))
        try:
            scored.append((_ring_score(_load_npz(path)), path, lid, klass))
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _add_scale_bar(ax, extent, length=5.0):
    x0 = extent[0] + 0.08 * (extent[1] - extent[0])
    y0 = extent[3] - 0.10 * (extent[3] - extent[2])
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=2.2, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 1.2, f'{length:.0f}"', color="white",
            ha="center", va="bottom", fontsize=8, fontweight="bold")


def _add_theta_e_circle(ax, theta_e: float):
    if theta_e <= 0:
        return
    ax.add_patch(Circle((0, 0), theta_e, fill=False, ec="orange",
                        ls="--", lw=1.4, alpha=0.95))


def _class_label(klass: str, meta: dict) -> str:
    mapping = {
        "GR": "Group SIE+NFW",
        "BR": "Binary SIE+SIE",
        "SF": "Single Field SIE",
    }
    return mapping.get(klass, meta.get("lens_system_class", klass) or klass)


def make_fullmaps(
    npz_path: Path,
    kappa_npz: Path,
    out_path: Path,
    *,
    label: str | None = None,
    klass: str = "GR",
) -> Path:
    npz = _load_npz(npz_path)
    meta = npz.get("metadata", {}) or {}
    bands = list(meta.get("bands", EUCLID_BANDS))
    final = _band_dict(npz["image_final"], bands)

    arc_images = None
    if "image_lens_sources" in npz and "image_lens_only" in npz:
        arc_images = {
            "lens_sources": _band_dict(npz["image_lens_sources"], bands),
            "lens_only": _band_dict(npz["image_lens_only"], bands),
        }

    # Temporarily boost arc visibility for the paper colour panel
    rgb = create_jwst_rgb(final, bands=bands, telescope="euclid", arc_images=arc_images)
    if rgb is None:
        rgb = _gray_rgb(final[bands[0]])

    kdat = np.load(kappa_npz, allow_pickle=True)
    kappa = np.asarray(kdat["kappa"], dtype=np.float64)
    mu = np.asarray(kdat["mu"], dtype=np.float64)
    F_mag = np.asarray(kdat["F_mag"], dtype=np.float64)
    G_mag = np.asarray(kdat["G_mag"], dtype=np.float64)
    extent = np.asarray(kdat["extent"], dtype=float)
    theta_e = float(meta.get("theta_E", kdat["theta_E_eff"] if "theta_E_eff" in kdat.files else 0.0) or 0.0)
    theta_e_eff = float(kdat["theta_E_eff"]) if "theta_E_eff" in kdat.files else theta_e
    # Prefer catalog θ_E for the overlay (matches final_B annotation style)
    te_draw = theta_e if theta_e > 0 else theta_e_eff

    vis = final.get("EUCLID_VIS", final[bands[0]])
    yb = final.get("EUCLID_Y", final[bands[min(1, len(bands) - 1)]])
    hb = final.get("EUCLID_H", final[bands[-1]])

    panels = [
        (rgb, "colour (H/J/VIS)", None),
        (_gray_rgb(vis), "VIS", None),
        (_cmap_rgb(yb, "YlOrBr_r"), "NISP-Y", None),
        (_cmap_rgb(hb, "hot"), "NISP-H", None),
        (_map_stretch(kappa, "inferno", vmin=0, vmax=max(2.0, np.percentile(kappa, 95))), r"$\kappa$", None),
        (_map_stretch(mu, "coolwarm", vmin=-1, vmax=3, log=True), r"$|\mu|$ (log)", None),
        (_map_stretch(F_mag, "plasma", vmin=0, vmax=np.percentile(F_mag, 99)), r"$|F|$ 1st flexion", None),
        (_map_stretch(G_mag, "inferno", vmin=0, vmax=np.percentile(G_mag, 99)), r"$|G|$ 2nd flexion", None),
    ]

    zl = float(meta.get("lens_redshift", meta.get("zl", 0.4)) or 0.4)
    zs = float(meta.get("source_redshift", meta.get("zs", 2.0)) or 2.0)
    sigma = meta.get("sigma_kms") or meta.get("velocity_dispersion")
    sigma_s = f"{float(sigma):.0f}" if sigma not in (None, "", "None") else "~300"
    sys_name = label or _class_label(klass, meta)

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.6), dpi=160)
    fig.patch.set_facecolor("white")

    for ax, (img, title, _) in zip(axes.ravel(), panels):
        ax.imshow(img, extent=extent, origin="lower", interpolation="nearest")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xlabel(r'$\Delta$RA ["]', fontsize=8)
        ax.set_ylabel(r'$\Delta$Dec ["]', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        _add_scale_bar(ax, extent, 5.0)
        _add_theta_e_circle(ax, te_draw)

    header = (
        f"{sys_name}   |   z={zl:.2f}   σ≈{sigma_s} km/s   "
        f"θ_E={te_draw:.1f}\"   |   Euclid full detector chain   "
        f"ZP(VIS)=25.58   FOV=60\"×60\""
    )
    fig.suptitle(header, fontsize=11, fontweight="bold", y=0.995)
    fig.text(0.5, 0.01,
             f"source z={zs:.2f}   ·   orange dashed circle: θ_E   ·   "
             f"κ effective θ_E={theta_e_eff:.2f}\"",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=[0.01, 0.03, 0.99, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[FULLMAPS] {out_path}")
    return out_path


def package_example(sim_dir: Path, out_dir: Path, path: Path, lid: int, klass: str) -> Path:
    """Also write band/RGB/kappa assets in euclid_final naming."""
    from scripts.local.package_paper_hero_figure import package_hero  # type: ignore

    prefix = f"ex{lid:02d}_{klass.lower()}"
    package_hero(sim_dir, out_dir / prefix, prefix, lid)
    kappa = sim_dir / "kappa_maps" / f"{lid:06d}_kappa_data.npz"
    full = make_fullmaps(
        path, kappa, out_dir / prefix / f"{prefix}_fullmaps.jpg",
        klass=klass,
    )
    # Convenient top-level copy
    Image.open(full).save(out_dir / f"{prefix}_fullmaps.jpg", quality=95)
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top", type=int, default=6, help="How many best examples")
    parser.add_argument("--lens-id", type=int, default=None)
    args = parser.parse_args()

    sim_dir = args.sim_dir.resolve()
    out_dir = (args.out_dir or (sim_dir / "EXAMPLES")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _find_candidates(sim_dir)
    if not candidates:
        raise SystemExit(f"No NPZ found in {sim_dir}/unified_npz")

    if args.lens_id is not None:
        selected = [c for c in candidates if c[2] == args.lens_id]
        if not selected:
            raise SystemExit(f"lens-id {args.lens_id} not found")
    else:
        selected = candidates[: args.top]

    print(f"[SELECT] packaging {len(selected)} examples -> {out_dir}")
    index_rows = []
    for score, path, lid, klass in selected:
        kappa = sim_dir / "kappa_maps" / f"{lid:06d}_kappa_data.npz"
        if not kappa.exists():
            print(f"[WARN] missing kappa for {lid:06d}")
            continue
        prefix = f"ex{lid:02d}_{klass.lower()}"
        dest = out_dir / f"{prefix}_fullmaps.jpg"
        make_fullmaps(path, kappa, dest, klass=klass)
        # Copy RGB strip for quick browsing
        rgb_strip = sim_dir / "jpg_rgb" / path.name.replace(".npz", ".jpg")
        if rgb_strip.exists():
            Image.open(rgb_strip).save(out_dir / f"{prefix}_bands.jpg", quality=92)
        index_rows.append((score, lid, klass, dest.name))

    # Simple HTML chooser
    html = ["<html><head><meta charset='utf-8'><title>final_B-style examples</title>",
            "<style>body{font-family:system-ui;margin:24px;background:#111;color:#eee}",
            "img{max-width:100%;border:1px solid #333;margin:8px 0 24px}",
            "h2{margin-top:32px}</style></head><body>",
            "<h1>Euclid final_B-style examples</h1>",
            f"<p>Source: <code>{sim_dir}</code></p>"]
    for score, lid, klass, name in index_rows:
        html.append(f"<h2>{name}  (id={lid:06d}, {klass}, ring_score={score:.0f})</h2>")
        html.append(f"<img src='{name}'/>")
        bands = name.replace("_fullmaps.jpg", "_bands.jpg")
        if (out_dir / bands).exists():
            html.append(f"<img src='{bands}'/>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))
    print(f"[DONE] open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
