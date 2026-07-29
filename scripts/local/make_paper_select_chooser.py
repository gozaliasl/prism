#!/usr/bin/env python3
"""Build a paper-selection chooser grid from Euclid 1′ RGB-only JPGs.

Usage:
  python scripts/local/make_paper_select_chooser.py outputs/euclid_paper_select_50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]


def _font(size: int = 14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _arc_score(npz: Path) -> float:
    d = np.load(npz, allow_pickle=True)
    if "image_lens_sources" not in d.files or "image_lens_only" not in d.files:
        return float(np.sum(d["image_final"]))
    res = np.maximum(d["image_lens_sources"][0] - d["image_lens_only"][0], 0)
    return float(res.sum())


def _rgb_only(panel_jpg: Path, rgb_only: Path, size: int = 280) -> Image.Image:
    if rgb_only.exists():
        im = Image.open(rgb_only).convert("RGB")
    else:
        im = Image.open(panel_jpg).convert("RGB")
        w, h = im.size
        # 5-panel strip → take rightmost panel
        if w > h * 1.5:
            pw = w // 5
            im = im.crop((4 * pw, 0, 5 * pw, h))
            # drop label strip if present
            if im.size[1] > im.size[0]:
                im = im.crop((0, im.size[1] - im.size[0], im.size[0], im.size[1]))
    return im.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell", type=int, default=280)
    args = ap.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    jpg_dir = run_dir / "jpg_rgb"
    npz_dir = run_dir / "unified_npz"
    out_dir = run_dir / "CHOOSER"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_path = run_dir / "cosmos_training_catalog_lens_and_nonlens.csv"
    cat = pd.read_csv(cat_path) if cat_path.exists() else None

    # Score + sort
    scored = []
    for npz in sorted(npz_dir.glob("PRISM_lens_*.npz")):
        scored.append((_arc_score(npz), npz))
    scored.sort(reverse=True)

    rows_meta = []
    tiles = []
    font = _font(13)
    font_sm = _font(11)
    cell = args.cell
    label_h = 36

    for rank, (score, npz) in enumerate(scored, 1):
        stem = npz.stem
        panel = jpg_dir / f"{stem}.jpg"
        rgb_only = jpg_dir / f"{stem}_RGBonly.jpg"
        if not panel.exists() and not rgb_only.exists():
            continue
        im = _rgb_only(panel, rgb_only, size=cell)
        tile = Image.new("RGB", (cell, cell + label_h), (20, 20, 20))
        tile.paste(im, (0, label_h))
        draw = ImageDraw.Draw(tile)

        cls = "?"
        thE = "?"
        nf = "?"
        if cat is not None and "lens_id" in cat.columns:
            # match by index in filename ..._000048
            try:
                idx = int(stem.split("_")[-1])
                if idx < len(cat):
                    row = cat.iloc[idx]
                    cls = str(row.get("lens_system_class", "?"))[:10]
                    thE = f"{float(row.get('theta_E', 0)):.1f}\""
                    nf = str(int(row.get("n_field_galaxies", 0)))
            except Exception:
                pass

        draw.text((4, 2), f"#{rank}  {stem}", fill=(255, 220, 100), font=font_sm)
        draw.text((4, 18), f"{cls}  θE={thE}  Nf={nf}", fill=(180, 200, 255), font=font_sm)
        tiles.append(tile)
        rows_meta.append({
            "rank": rank, "stem": stem, "arc_score": score,
            "class": cls, "theta_E": thE, "n_field": nf,
        })

    if not tiles:
        raise SystemExit(f"No RGB tiles in {jpg_dir}")

    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * (cell + label_h)), (0, 0, 0))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(tile, (c * cell, r * (cell + label_h)))

    grid_path = out_dir / "chooser_grid.jpg"
    grid.save(grid_path, quality=92, optimize=True)

    # Also write ranked RGB-only copies
    ranked_dir = out_dir / "ranked_rgb"
    ranked_dir.mkdir(exist_ok=True)
    for m, tile in zip(rows_meta, tiles):
        # extract image part
        rgb = tile.crop((0, label_h, cell, cell + label_h))
        rgb.save(ranked_dir / f"rank{m['rank']:02d}_{m['stem']}.jpg", quality=95)

    pd.DataFrame(rows_meta).to_csv(out_dir / "chooser_ranking.csv", index=False)
    (out_dir / "chooser_ranking.json").write_text(json.dumps(rows_meta, indent=2))

    print(f"[CHOOSER] {len(tiles)} systems → {grid_path}")
    print(f"[CHOOSER] ranked RGB → {ranked_dir}")
    print(f"[CHOOSER] catalog → {out_dir / 'chooser_ranking.csv'}")


if __name__ == "__main__":
    main()
