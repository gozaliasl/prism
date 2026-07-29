"""
Euclid Q1 strong-lens catalogue integration for JELSIM.

Loads modelling results and PSF metadata from data/euclid_q1_psf/ (and optional
external CSV copies) to drive population priors and empirical PSF assignment
when telescope == 'euclid'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _REPO_ROOT / 'data' / 'euclid_q1_psf'

_EUCLID_BANDS = ('EUCLID_VIS', 'EUCLID_Y', 'EUCLID_J', 'EUCLID_H')
_EUCLID_LOWER = tuple(b.lower() for b in _EUCLID_BANDS)

# Empirical lens colors relative to VIS (AB), from Q1 MGE modelling (N~322).
_LENS_COLOR_VS_VIS = {
    'euclid_vis': 0.0,
    'euclid_y': -1.52,
    'euclid_j': -1.87,
    'euclid_h': -2.24,
}
_SOURCE_COLOR_VS_VIS = {
    'euclid_vis': 0.0,
    'euclid_y': 0.19,
    'euclid_j': -0.19,
}


def is_euclid_q1_psf_data(psf_data: dict) -> bool:
    """True if psf_data keys are Q1 empirical tiles (Q1_*)."""
    if not psf_data:
        return False
    keys = list(psf_data.keys())
    return keys and keys[0].startswith('Q1_')


def euclid_q1_enabled(config: dict) -> bool:
    cfg = config.get('euclid_q1', {})
    return bool(cfg.get('enabled', False)) and config.get('telescope', 'jwst').lower() == 'euclid'


def get_euclid_lower_bands(config: dict) -> list[str]:
    tel = config.get('telescope_configs', {}).get('euclid', {})
    bands = tel.get('bands', list(_EUCLID_BANDS))
    return [b.lower() for b in bands]


class EuclidQ1Catalog:
    """In-memory Q1 modelling sample for population draws."""

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA
        self.psf_catalog = self._read_csv('psf_catalog.csv')
        self.mass = self._read_csv_optional('modeling_lens_mass.csv')
        self.sersic = self._read_csv_optional('modeling_lens_sersic.csv')
        self.mge_mag = self._read_csv_optional('modeling_mge_magnitude.csv')
        self._merged = self._build_merged()
        self.psf_tiles = (
            sorted(self.psf_catalog['tile'].dropna().unique().tolist())
            if self.psf_catalog is not None and 'tile' in self.psf_catalog.columns
            else []
        )

    def _read_csv(self, name: str) -> pd.DataFrame | None:
        path = self.data_dir / name
        if path.exists():
            return pd.read_csv(path)
        return None

    def _read_csv_optional(self, name: str) -> pd.DataFrame | None:
        path = self.data_dir / name
        if path.exists():
            return pd.read_csv(path)
        # Fall back to bundled copies symlinked/copied alongside psf_catalog
        alt = self.data_dir.parent / 'euclid_q1_catalog' / name
        if alt.exists():
            return pd.read_csv(alt)
        return None

    def _build_merged(self) -> pd.DataFrame | None:
        if self.mass is None or self.mass.empty:
            return None
        df = self.mass.copy()
        if self.sersic is not None and not self.sersic.empty:
            s_cols = ['id_str', 'effective_radius_median_pdf', 'sersic_index_median_pdf',
                      'ell_comps_0_median_pdf', 'ell_comps_1_median_pdf']
            s_cols = [c for c in s_cols if c in self.sersic.columns]
            df = df.merge(self.sersic[s_cols], on='id_str', how='left', suffixes=('', '_ser'))
        if self.mge_mag is not None and not self.mge_mag.empty:
            m_cols = [c for c in self.mge_mag.columns if c.endswith('_median_pdf')
                      and ('magnitude' in c or 'magnification' in c)]
            m_cols = ['id_str'] + m_cols
            m_cols = [c for c in m_cols if c in self.mge_mag.columns]
            df = df.merge(self.mge_mag[m_cols], on='id_str', how='left')
        if self.psf_catalog is not None and 'tile' in self.psf_catalog.columns:
            df = df.merge(
                self.psf_catalog[['id_str', 'tile', 'grade']],
                on='id_str', how='left',
            )
        # Keep rows with valid theta_E
        if 'einstein_radius_median_pdf' in df.columns:
            df = df[df['einstein_radius_median_pdf'].fillna(0) > 0].reset_index(drop=True)
        return df

    @property
    def n_systems(self) -> int:
        return 0 if self._merged is None else len(self._merged)

    def sample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self._merged is None or self.n_systems == 0:
            return rng.integers(0, 1, size=n)
        return rng.integers(0, self.n_systems, size=n)

    def sample_row(self, rng: np.random.Generator) -> dict | None:
        if self._merged is None or self.n_systems == 0:
            return None
        return self._merged.iloc[int(rng.integers(0, self.n_systems))].to_dict()

    def sample_physics_batch(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw lens physics arrays of length n from the Q1 modelling sample."""
        if self._merged is None or self.n_systems == 0:
            return {}
        idx = self.sample_indices(n, rng)
        m = self._merged.iloc[idx]
        e1 = m['ell_comps_0_median_pdf'].fillna(0).to_numpy()
        e2 = m['ell_comps_1_median_pdf'].fillna(0).to_numpy()
        g1 = m['shear_gamma_1_median_pdf'].fillna(0).to_numpy()
        g2 = m['shear_gamma_2_median_pdf'].fillna(0).to_numpy()
        shear = np.sqrt(g1 ** 2 + g2 ** 2)
        shear_phi = 0.5 * np.arctan2(g2, g1)
        out = {
            'theta_E': m['einstein_radius_median_pdf'].to_numpy(),
            'lens_e1': e1,
            'lens_e2': e2,
            'shear_gamma1': g1,
            'shear_gamma2': g2,
            'shear_g': shear,
            'shear_phi': shear_phi,
        }
        if 'effective_radius_median_pdf' in m.columns:
            out['lens_radius'] = m['effective_radius_median_pdf'].to_numpy()
        if 'sersic_index_median_pdf' in m.columns:
            out['n_rest'] = m['sersic_index_median_pdf'].to_numpy()
        if 'tile' in m.columns:
            out['euclid_psf_tile'] = m['tile'].astype(str).to_numpy()
        return out

    def sample_magnitudes_batch(
            self, n: int, rng: np.random.Generator, bands: list[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return (lens_mags, source_mags) dicts keyed by lower-case band name."""
        bands = bands or list(_EUCLID_LOWER)
        if self._merged is None or self.n_systems == 0:
            return {}, {}

        idx = self.sample_indices(n, rng)
        m = self._merged.iloc[idx]

        band_map = {
            'euclid_vis': 'vis_lens_magnitude_ab_median_pdf',
            'euclid_y': 'nir_y_lens_magnitude_ab_median_pdf',
            'euclid_j': 'nir_j_lens_magnitude_ab_median_pdf',
            'euclid_h': 'nir_h_lens_magnitude_ab_median_pdf',
        }
        src_map = {
            'euclid_vis': 'vis_source_magnitude_ab_median_pdf',
            'euclid_y': 'nir_y_source_magnitude_ab_median_pdf',
            'euclid_j': 'nir_j_source_magnitude_ab_median_pdf',
            'euclid_h': 'nir_h_source_magnitude_ab_median_pdf',
        }

        lens_out: dict[str, np.ndarray] = {}
        src_out: dict[str, np.ndarray] = {}
        for band in bands:
            lcol = band_map.get(band)
            scol = src_map.get(band)
            if lcol and lcol in m.columns:
                lens_out[band] = pd.to_numeric(m[lcol], errors='coerce').fillna(21.5).to_numpy()
            if scol and scol in m.columns:
                src_out[band] = pd.to_numeric(m[scol], errors='coerce').fillna(22.5).to_numpy()
        return lens_out, src_out

    def assign_psf_tile(self, rng: np.random.Generator, lens_id: str | None = None) -> str | None:
        if not self.psf_tiles:
            return None
        if lens_id and self.psf_catalog is not None and 'id_str' in self.psf_catalog.columns:
            match = self.psf_catalog[self.psf_catalog['id_str'].astype(str) == str(lens_id)]
            if not match.empty and 'tile' in match.columns:
                return str(match.iloc[0]['tile'])
        return str(rng.choice(self.psf_tiles))

    def write_psf_assignment(self, lens_ids: list[str], out_path: Path | str) -> None:
        """Create psf_assignment.csv mapping lens_id → Q1 PSF tile."""
        rng = np.random.default_rng(42)
        rows = []
        for lid in lens_ids:
            tile = self.assign_psf_tile(rng, lens_id=None)
            rows.append({'lens_id': lid, 'tile': tile, 'psf_source': 'euclid_q1'})
        pd.DataFrame(rows).to_csv(out_path, index=False)


_catalog_cache: EuclidQ1Catalog | None = None


def get_euclid_q1_catalog(config: dict) -> EuclidQ1Catalog | None:
    global _catalog_cache
    if not euclid_q1_enabled(config):
        return None
    data_dir = config.get('euclid_q1', {}).get('data_dir', str(_DEFAULT_DATA))
    if _catalog_cache is None or str(_catalog_cache.data_dir) != str(data_dir):
        _catalog_cache = EuclidQ1Catalog(data_dir)
        print(f"[EUCLID Q1] Loaded catalogue: { _catalog_cache.n_systems} modelled systems, "
              f"{len(_catalog_cache.psf_tiles)} PSF tiles from {data_dir}")
    return _catalog_cache


def apply_euclid_q1_photometry(df: pd.DataFrame, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Overwrite lens/source magnitudes in catalog with Q1 empirical draws."""
    cat = get_euclid_q1_catalog(config)
    if cat is None or df.empty:
        return df

    bands = get_euclid_lower_bands(config)
    phot = config.get('photometry', {})
    n = len(df)
    lens_mags, src_mags = cat.sample_magnitudes_batch(n, rng, bands)

    out = df.copy()
    min_delta = float(phot.get('min_source_fainter_than_lens_mag', 1.0))

    for band in bands:
        if band in lens_mags:
            out[f'lens_mag_{band}'] = np.clip(
                lens_mags[band],
                phot.get('lens_mag_min', 18.0),
                phot.get('lens_mag_max', 24.0),
            )
        if band in src_mags:
            src = src_mags[band]
            if band in lens_mags:
                src = np.maximum(src, lens_mags[band] + min_delta)
            out[f'source_mag_{band}'] = np.clip(
                src,
                phot.get('source_mag_min', 19.0),
                phot.get('source_mag_max', 27.0),
            )
    return out


def apply_euclid_q1_physics(df: pd.DataFrame, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Inject Q1-sampled theta_E, ellipticity, shear, Sersic into catalog."""
    cat = get_euclid_q1_catalog(config)
    if cat is None or df.empty:
        return df

    cfg = config.get('euclid_q1', {})
    if not cfg.get('use_population_priors', True):
        return df

    n = len(df)
    phys = cat.sample_physics_batch(n, rng)
    if not phys:
        return df

    out = df.copy()
    if cfg.get('resample_theta_E', True) and 'theta_E' in phys:
        out['theta_E'] = phys['theta_E']
    if 'lens_e1' in phys and 'lens_e2' in phys:
        out['lens_e1'] = phys['lens_e1']
        out['lens_e2'] = phys['lens_e2']
        out['lens_axis_ratio'] = np.clip(
            (1 - np.sqrt(phys['lens_e1'] ** 2 + phys['lens_e2'] ** 2)) / 
            (1 + np.sqrt(phys['lens_e1'] ** 2 + phys['lens_e2'] ** 2) + 1e-9),
            0.2, 1.0,
        )
    if cfg.get('use_q1_shear', True):
        out['shear_gamma1'] = phys['shear_gamma1']
        out['shear_gamma2'] = phys['shear_gamma2']
        out['shear_g'] = phys['shear_g']
        out['shear_phi'] = phys['shear_phi']
    if 'lens_radius' in phys:
        out['lens_radius'] = phys['lens_radius']
    if 'n_rest' in phys:
        out['n_rest'] = phys['n_rest']
    if 'euclid_psf_tile' in phys:
        out['euclid_psf_tile'] = phys['euclid_psf_tile']
    return out


def euclid_color_offsets_vs_vis(band_lower: str, component: str = 'lens') -> float:
    table = _LENS_COLOR_VS_VIS if component == 'lens' else _SOURCE_COLOR_VS_VIS
    return float(table.get(band_lower, 0.0))
