"""IllustrisTNG (TNG100-1) galaxy selector -- foundation for "TNG Mode".

Picks physically realistic subhalos from the public IllustrisTNG API by
target redshift, stellar mass, and (optionally) group environment, returning
their physical properties (stellar mass, SFR/sSFR, half-mass radius, gas/
stellar metallicity, group multiplicity / environment class). This is the
selection layer only -- it does not render images. The returned properties
are intended to drive the *parameters* of the existing Sersic/multi-component
light models (``src/galaxy_morphology``) and, for a small subset, the
GalaxyGenius/SKIRT stamp pipeline (``src/galaxygenius_stamps.py``).

Requires ``TNG_API_KEY`` (IllustrisTNG API token) in the environment.
Results are cached as JSON under ``data/tng_catalogs/`` to avoid repeated
API calls.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

TNG_API_BASE = "https://www.tng-project.org/api/"
TNG_VERSION = "TNG100-1"
LITTLE_H = 0.6774

# Local particle-cutout filename prefixes per simulation. TNG50-1 has ~16x
# better star-particle mass resolution than TNG100-1 (at the cost of a
# ~9.7x smaller box volume / fewer high-mass halos), so it is used to get
# better-resolved morphology for low-mass (logM* < ~10) field/source/
# companion galaxies where TNG100-1 cutouts are too sparse (~250 particles
# at logM*=9.5) to render smooth light profiles.
SIM_FILE_PREFIX = {
    "TNG100-1": "TNG_100",
    "TNG50-1": "TNG_50",
}

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "tng_catalogs"

# Directory holding locally-downloaded TNG particle cutouts
# (TNG_100_snap_{snapshot}_subhalo_{subhalo_id}.h5), fetched via
# scripts/fetch_galaxygenius_subhalo_particles.py /
# scripts/batch_fetch_galaxygenius_stamps.py. Used by the TNG particle-driven
# morphology mode (src/galaxy_morphology/tng_particle_light.py).
PARTICLE_DATA_DIR = Path("/Volumes/extHD/galaxygenius_build/workspace/data")


_PARTICLE_COUNT_CACHE: dict[str, int] = {}


def _star_particle_count(path: Path) -> int:
    """Number of PartType4 (star) particles in a local cutout, cached."""
    key = str(path)
    if key in _PARTICLE_COUNT_CACHE:
        return _PARTICLE_COUNT_CACHE[key]
    import h5py
    with h5py.File(path, "r") as f:
        n = f["PartType4"]["Coordinates"].shape[0] if "PartType4" in f else 0
    _PARTICLE_COUNT_CACHE[key] = n
    return n


def local_particle_path(snapshot: int, subhalo_id: int, min_particles: int | None = None,
                         sim: str = "TNG100-1") -> Path | None:
    """Path to the locally-downloaded particle cutout for ``(snapshot,
    subhalo_id)`` in simulation ``sim`` (``"TNG100-1"`` or ``"TNG50-1"``), or
    ``None`` if it hasn't been downloaded.

    If ``min_particles`` is given, also returns ``None`` when the cutout has
    fewer than that many star particles -- most TNG100-1 local cutouts
    (median ~250 particles) are too sparse for smooth particle-rendered
    morphology and produce a "bead field" of discrete clumps rather than a
    continuous light profile; such cutouts should fall back to TNG-informed
    Sersic. TNG50-1 cutouts at the same stellar mass have ~16x more
    particles and are far less likely to be filtered out by this check.
    """
    prefix = SIM_FILE_PREFIX[sim]
    path = PARTICLE_DATA_DIR / f"{prefix}_snap_{int(snapshot)}_subhalo_{int(subhalo_id)}.h5"
    if not path.exists():
        return None
    if min_particles is not None and _star_particle_count(path) < min_particles:
        return None
    return path

# TNG100-1 snapshot number -> redshift, for snapshots spanning z=0 to z~6.5
# (covers the lens/source redshift range used by jwst_lens_simulator.py).
SNAPSHOT_REDSHIFTS = {
    12: 6.492, 13: 6.011, 14: 5.847, 15: 5.530, 16: 5.228, 17: 4.996,
    18: 4.665, 19: 4.428, 20: 4.177, 21: 4.008, 22: 3.709, 23: 3.491,
    24: 3.283, 25: 3.008, 26: 2.896, 27: 2.733, 28: 2.577, 29: 2.444,
    30: 2.316, 31: 2.208, 32: 2.103, 33: 2.002, 34: 1.904, 35: 1.823,
    36: 1.744, 37: 1.667, 38: 1.604, 39: 1.531, 40: 1.496, 41: 1.414,
    42: 1.358, 43: 1.302, 44: 1.248, 45: 1.206, 46: 1.155, 47: 1.114,
    48: 1.074, 49: 1.036, 50: 0.997, 51: 0.951, 52: 0.923, 53: 0.887,
    54: 0.851, 55: 0.817, 56: 0.791, 57: 0.757, 58: 0.733, 59: 0.700,
    60: 0.676, 61: 0.645, 62: 0.621, 63: 0.599, 64: 0.576, 65: 0.546,
    66: 0.525, 67: 0.503, 68: 0.482, 69: 0.461, 70: 0.440, 71: 0.420,
    72: 0.400, 73: 0.380, 74: 0.361, 75: 0.348, 76: 0.329, 77: 0.310,
    78: 0.298, 79: 0.273, 80: 0.261, 81: 0.244, 82: 0.226, 83: 0.214,
    84: 0.197, 85: 0.180, 86: 0.169, 87: 0.153, 88: 0.142, 89: 0.126,
    90: 0.110, 91: 0.099, 92: 0.084, 93: 0.074, 94: 0.059, 95: 0.049,
    96: 0.034, 97: 0.024, 98: 0.010, 99: 0.000,
}

# Environment classification is based on the number of subhalos in the same
# FoF group with stellar mass above this threshold (logM > 8.0), NOT the
# group's total ``child_subhalos.count`` -- the latter includes thousands of
# tiny dark-matter-only subhalos even for an "isolated" central, so it is
# useless as an isolated/pair/group/rich-group discriminator.
ENV_MIN_COMPANION_LOGM = 8.0
ENV_PAIR_MAX_COUNT = 2
ENV_GROUP_MAX_COUNT = 5


def nearest_snapshot(redshift: float) -> int:
    """TNG100-1 snapshot number whose redshift is closest to ``redshift``."""
    return min(SNAPSHOT_REDSHIFTS, key=lambda s: abs(SNAPSHOT_REDSHIFTS[s] - redshift))


def _headers() -> dict:
    key = os.environ.get("TNG_API_KEY")
    if not key:
        raise RuntimeError("Set TNG_API_KEY in the environment (IllustrisTNG API token).")
    return {"api-key": key}


def _get(url: str, params: dict | None = None, max_retries: int = 5) -> dict:
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required for live IllustrisTNG API lookups. "
            "Install it with: pip install requests  "
            "(local-catalog / particle-cutout mode does not need the API.)"
        ) from exc
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=_headers(), params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_subhalo_summaries(
    snapshot_id: int, min_logM: float, max_logM: float, limit: int = 500, sim: str = "TNG100-1"
) -> list[dict]:
    """Bulk catalog query: ``[{id, sfr, mass_log_msun, url}, ...]`` for
    subhalos with ``min_logM <= log10(M_stars/Msun) <= max_logM``.

    The TNG API's ``mass_stars__lt`` filter is unreliable when combined with
    ``mass_stars__gt`` (empirically verified: it has no additional effect),
    so only the lower bound is applied server-side (ordered ascending by
    stellar mass, to bias the page toward the requested range), and the
    upper bound is applied client-side on ``mass_log_msun``.

    Cached to ``data/tng_catalogs/``.
    """
    cache_path = CACHE_DIR / (
        f"{sim}_snap{snapshot_id}_logM{min_logM:.2f}-{max_logM:.2f}.json"
    )
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    min_mass = 10 ** min_logM / 1e10 * LITTLE_H
    url = f"{TNG_API_BASE}{sim}/snapshots/{snapshot_id}/subhalos/"
    params = {
        "mass_stars__gt": min_mass,
        "order_by": "mass_stars",
        "limit": limit,
    }
    data = _get(url, params=params)
    results = [r for r in data.get("results", []) if r["mass_log_msun"] <= max_logM]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results))
    return results


def fetch_all_subhalo_summaries(
    snapshot_id: int, min_logM: float, max_logM: float, page_limit: int = 500, sim: str = "TNG100-1"
) -> list[dict]:
    """Like ``fetch_subhalo_summaries``, but paginates through *all* matching
    subhalos (not just the first ``page_limit``) using the API's DRF-style
    ``next`` pagination links. Used by ``scripts/build_tng_local_catalog.py``
    for the offline pre-fetch -- not used in the per-simulation hot path.

    Cached (full concatenated result) to ``data/tng_catalogs/``.
    """
    cache_path = CACHE_DIR / (
        f"{sim}_snap{snapshot_id}_allsummaries_logM{min_logM:.2f}-{max_logM:.2f}.json"
    )
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    min_mass = 10 ** min_logM / 1e10 * LITTLE_H
    url = f"{TNG_API_BASE}{sim}/snapshots/{snapshot_id}/subhalos/"
    params = {
        "mass_stars__gt": min_mass,
        "order_by": "mass_stars",
        "limit": page_limit,
        "offset": 0,
    }

    results: list[dict] = []
    while True:
        data = _get(url, params=params)
        page = data.get("results", [])
        results.extend(page)
        if not data.get("next") or not page:
            break
        params["offset"] += page_limit

    results = [r for r in results if r["mass_log_msun"] <= max_logM]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results))
    return results


def fetch_subhalo_detail(snapshot_id: int, subhalo_id: int, sim: str = "TNG100-1") -> dict:
    """Full subhalo record (mass, SFR, size, metallicity, grnr, ...). Cached."""
    cache_path = CACHE_DIR / f"{sim}_snap{snapshot_id}_subhalo{subhalo_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    url = f"{TNG_API_BASE}{sim}/snapshots/{snapshot_id}/subhalos/{subhalo_id}/"
    data = _get(url)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data


def fetch_group_massive_subhalo_count(
    snapshot_id: int, halo_id: int, min_logM: float = ENV_MIN_COMPANION_LOGM, sim: str = "TNG100-1"
) -> int:
    """Number of subhalos in FoF group ``halo_id`` with
    ``log10(M_stars/Msun) > min_logM`` (includes the central/primary itself).
    Cached."""
    cache_path = CACHE_DIR / (
        f"{sim}_snap{snapshot_id}_halo{halo_id}_ncompanions_logM{min_logM:.1f}.json"
    )
    if cache_path.exists():
        return json.loads(cache_path.read_text())["count"]

    min_mass = 10 ** min_logM / 1e10 * LITTLE_H
    url = f"{TNG_API_BASE}{sim}/snapshots/{snapshot_id}/subhalos/"
    params = {"grnr": halo_id, "mass_stars__gt": min_mass, "limit": 1}
    data = _get(url, params=params)
    count = int(data.get("count", 1))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"count": count}))
    return count


def classify_environment(group_massive_count: int) -> str:
    """``isolated`` / ``pair`` / ``group`` / ``rich_group`` from the number
    of stellar-mass-significant subhalos (``log10 M* > ENV_MIN_COMPANION_LOGM``)
    sharing the same FoF group, including the galaxy itself.
    """
    if group_massive_count <= 1:
        return "isolated"
    if group_massive_count <= ENV_PAIR_MAX_COUNT:
        return "pair"
    if group_massive_count <= ENV_GROUP_MAX_COUNT:
        return "group"
    return "rich_group"


def detail_to_record(snap: int, snap_z: float, detail: dict, group_massive_count: int,
                      sim: str = "TNG100-1") -> dict:
    """Convert a raw TNG API subhalo ``detail`` record into the standard
    physical-properties dict returned by ``select_tng_galaxy`` /
    ``select_tng_galaxy_local``."""
    mass_msun = 10 ** detail["mass_log_msun"]
    sfr = float(detail.get("sfr", 0.0))
    return {
        "sim": sim,
        "snapshot": snap,
        "snapshot_redshift": snap_z,
        "subhalo_id": int(detail["id"]),
        "halo_id": int(detail["grnr"]),
        "stellar_mass_logmsun": float(detail["mass_log_msun"]),
        "sfr_msun_per_yr": sfr,
        "ssfr_per_yr": sfr / mass_msun if mass_msun > 0 else 0.0,
        "halfmassrad_stars_kpc": float(detail["halfmassrad_stars"]) / LITTLE_H,
        "gas_mass_msun": float(detail["mass_gas"]) / LITTLE_H * 1e10,
        "gas_metallicity": float(detail.get("gasmetallicity", 0.0)),
        "star_metallicity": float(detail.get("starmetallicity", 0.0)),
        "environment": classify_environment(group_massive_count),
        "group_massive_subhalo_count": group_massive_count,
        "primary_flag": int(detail.get("primary_flag", 0)),
    }


# ---------------------------------------------------------------------------
# Local-catalog selection (no network calls)
# ---------------------------------------------------------------------------
# Built offline by scripts/build_tng_local_catalog.py into a single Parquet
# file with one row per pre-fetched subhalo, columns matching the keys of
# the dict returned by select_tng_galaxy(). select_tng_galaxy_local() does
# the same (z, logM[, environment]) matching purely against this table, so
# simulation runs never hit the (slow, rate-limited) TNG API.
_LOCAL_CATALOG_CACHE: dict[str, "object"] = {}
_PARTICLE_SUBHALO_SETS: dict[str, frozenset] = {}


def _particle_subhalo_set(sim: str) -> frozenset:
    """Return frozenset of (snapshot, subhalo_id) tuples for subhalos that
    have a local particle file, loaded from the pre-built particle catalog.

    The particle catalog is written by the offline build step:
    ``python3 -c "... filter_to_particle_files ..."`` which saves
    ``/Volumes/extHD/tng_local_catalog/{sim_lower}_particle_catalog.parquet``.
    Falls back to an empty set if the file doesn't exist (in that case
    ``require_local_particles`` filtering silently passes all rows through,
    preserving old behavior rather than crashing).
    """
    import pandas as pd

    if sim in _PARTICLE_SUBHALO_SETS:
        return _PARTICLE_SUBHALO_SETS[sim]

    sim_key = sim.lower().replace("-", "").replace("_", "")  # "tng1001" / "tng501"
    # canonical filename mapping
    _fname_map = {
        "tng1001": "tng100-1_particle_catalog.parquet",
        "tng501": "tng50-1_particle_catalog.parquet",
    }
    fname = _fname_map.get(sim_key)
    result: frozenset = frozenset()
    if fname:
        cat_path = Path("/Volumes/extHD/tng_local_catalog") / fname
        if cat_path.exists():
            df = pd.read_parquet(cat_path, columns=["snapshot", "subhalo_id"])
            result = frozenset(zip(df["snapshot"].astype(int), df["subhalo_id"].astype(int)))
    _PARTICLE_SUBHALO_SETS[sim] = result
    return result


def load_local_catalog(path: str | Path):
    """Load (and cache in-process) the local TNG subhalo catalog Parquet
    file built by ``scripts/build_tng_local_catalog.py``. Returns ``None``
    if ``path`` doesn't exist."""
    import pandas as pd

    path = str(path)
    if path in _LOCAL_CATALOG_CACHE:
        return _LOCAL_CATALOG_CACHE[path]
    if not Path(path).exists():
        _LOCAL_CATALOG_CACHE[path] = None
        return None
    df = pd.read_parquet(path)
    _LOCAL_CATALOG_CACHE[path] = df
    return df


def select_tng_galaxy_local(
    target_z: float,
    target_logM: float,
    rng,
    catalog,
    logM_tol: float = 0.3,
    environment: str | None = None,
    max_attempts: int = 10,
    require_local_particles: bool = False,
    min_particles: int | None = None,
    exclude_subhalos: set | None = None,
    sfr_class: str | None = None,
    delta_z_window: float = 0.4,
) -> dict | None:
    """Like ``select_tng_galaxy``, but selects from a pre-fetched local
    catalog DataFrame (``catalog``, as returned by ``load_local_catalog``)
    with zero network calls.

    Pools candidates from **all snapshots within ``delta_z_window``** of
    ``target_z`` (not just the single nearest snapshot) so that a batch of
    lenses at similar redshifts draws from many different TNG subhalos
    rather than repeatedly picking from the same tiny snapshot pool.

    Parameters
    ----------
    exclude_subhalos : set of (sim, snapshot, subhalo_id) tuples, optional
        Subhalos already used in this batch run.  Matching rows are removed
        from candidates so the same galaxy is never reused for two different
        roles (lens/source/field) or two different systems.
    sfr_class : 'star_forming' | 'quiescent' | None
        If given, restricts to subhalos with log10(sSFR/yr⁻¹) > −10.5
        (star-forming) or < −11.5 (quiescent).  When ``None`` all SFR
        classes are eligible.
    delta_z_window : float
        Half-width in redshift used to pool nearby snapshots.  Candidates
        from all snapshots with |z_snap − target_z| < delta_z_window are
        merged before filtering by mass/environment/SFR.
    """
    import pandas as pd

    if catalog is None or len(catalog) == 0:
        return None

    # ------------------------------------------------------------------
    # Pool candidates from all snapshots within delta_z_window of target_z
    # ------------------------------------------------------------------
    available_snaps = catalog["snapshot"].unique()
    nearby_snaps = [
        s for s in available_snaps
        if abs(SNAPSHOT_REDSHIFTS.get(int(s), 0.0) - target_z) <= delta_z_window
    ]
    # Always include at least the nearest snapshot as a safety fallback.
    if not nearby_snaps:
        nearby_snaps = [min(available_snaps, key=lambda s: abs(SNAPSHOT_REDSHIFTS.get(int(s), 0.0) - target_z))]

    snap_catalog = catalog[catalog["snapshot"].isin(nearby_snaps)].copy()

    # ------------------------------------------------------------------
    # Filter to subhalos with local particle files (if required)
    # ------------------------------------------------------------------
    if require_local_particles:
        if "sim" in snap_catalog.columns:
            sim_col = snap_catalog["sim"].fillna("TNG100-1").astype(str)
            sim_col = sim_col.where(~sim_col.str.lower().isin({"nan", "none", ""}), "TNG100-1")
        else:
            sim_col = pd.Series("TNG100-1", index=snap_catalog.index)

        unique_sims = sim_col.unique()
        psets = {s: _particle_subhalo_set(s) for s in unique_sims}

        has_particles = pd.Series([
            (int(snap_catalog.at[i, "snapshot"]), int(snap_catalog.at[i, "subhalo_id"]))
            in psets.get(sim_col.at[i], frozenset())
            for i in snap_catalog.index
        ], index=snap_catalog.index)
        snap_catalog = snap_catalog[has_particles]

    # ------------------------------------------------------------------
    # Exclude subhalos already used in this batch run
    # ------------------------------------------------------------------
    if exclude_subhalos and len(snap_catalog) > 0:
        sim_col_ex = (
            snap_catalog["sim"].fillna("TNG100-1").astype(str)
            if "sim" in snap_catalog.columns
            else pd.Series("TNG100-1", index=snap_catalog.index)
        )
        not_used = pd.Series([
            (str(sim_col_ex.at[i]),
             int(snap_catalog.at[i, "snapshot"]),
             int(snap_catalog.at[i, "subhalo_id"])) not in exclude_subhalos
            for i in snap_catalog.index
        ], index=snap_catalog.index)
        filtered = snap_catalog[not_used]
        # Only apply exclusion if it leaves at least one candidate.
        if len(filtered) > 0:
            snap_catalog = filtered

    # ------------------------------------------------------------------
    # SFR stratification: star_forming / quiescent / any
    # ------------------------------------------------------------------
    if sfr_class is not None and "ssfr_per_yr" in snap_catalog.columns and len(snap_catalog) > 0:
        log_ssfr = snap_catalog["ssfr_per_yr"].clip(lower=1e-14).apply(
            lambda x: float(__import__("math").log10(max(x, 1e-14)))
        )
        if sfr_class == "star_forming":
            sfr_mask = log_ssfr > -10.5
        else:  # quiescent
            sfr_mask = log_ssfr < -11.5
        sfr_filtered = snap_catalog[sfr_mask]
        if len(sfr_filtered) > 0:
            snap_catalog = sfr_filtered

    # ------------------------------------------------------------------
    # Progressive mass + environment relaxation to guarantee a match
    # ------------------------------------------------------------------
    tol_multipliers = [1.0, 2.0, 4.0, 8.0] if require_local_particles else [1.0, 2.0, 4.0]
    candidates = snap_catalog.iloc[0:0]
    for relax_environment in ([False, True] if environment is not None else [False]):
        for mult in tol_multipliers:
            tol = logM_tol * mult
            mask = (
                (snap_catalog["stellar_mass_logmsun"] >= target_logM - tol)
                & (snap_catalog["stellar_mass_logmsun"] <= target_logM + tol)
            )
            if environment is not None and not relax_environment:
                mask &= snap_catalog["environment"] == environment
            candidates = snap_catalog[mask]
            if len(candidates) > 0:
                break
        if len(candidates) > 0:
            break

    if len(candidates) == 0:
        return None

    # When particle morphology is enabled, prefer subhalos whose cutouts have
    # enough star particles to render smoothly (see local_particle_path).
    if require_local_particles and min_particles is not None and len(candidates) > 0:
        sim_col_c = (
            candidates["sim"].fillna("TNG100-1").astype(str)
            if "sim" in candidates.columns
            else pd.Series("TNG100-1", index=candidates.index)
        )
        dense_idx = [
            i for i in candidates.index
            if local_particle_path(
                int(candidates.at[i, "snapshot"]),
                int(candidates.at[i, "subhalo_id"]),
                min_particles=min_particles,
                sim=str(sim_col_c.at[i]),
            ) is not None
        ]
        if dense_idx:
            candidates = candidates.loc[dense_idx]

    # TNG50-1 has ~16× better star-particle resolution at fixed stellar mass;
    # prefer it for low-mass sources/field galaxies when both sims match.
    if len(candidates) > 0 and target_logM < 10.5 and "sim" in candidates.columns:
        sim_col_pref = candidates["sim"].fillna("TNG100-1").astype(str).str.upper()
        tng50_rows = candidates[sim_col_pref.eq("TNG50-1")]
        if len(tng50_rows) > 0:
            candidates = tng50_rows

    row = candidates.iloc[int(rng.integers(len(candidates)))]
    return {
        "sim": str(row.get("sim", "TNG100-1") or "TNG100-1"),
        "snapshot": int(row["snapshot"]),
        "snapshot_redshift": float(row["snapshot_redshift"]),
        "subhalo_id": int(row["subhalo_id"]),
        "halo_id": int(row["halo_id"]),
        "stellar_mass_logmsun": float(row["stellar_mass_logmsun"]),
        "sfr_msun_per_yr": float(row["sfr_msun_per_yr"]),
        "ssfr_per_yr": float(row["ssfr_per_yr"]),
        "halfmassrad_stars_kpc": float(row["halfmassrad_stars_kpc"]),
        "gas_mass_msun": float(row["gas_mass_msun"]),
        "gas_metallicity": float(row["gas_metallicity"]),
        "star_metallicity": float(row["star_metallicity"]),
        "environment": str(row["environment"]),
        "group_massive_subhalo_count": int(row["group_massive_subhalo_count"]),
        "primary_flag": int(row["primary_flag"]),
    }


def select_tng_galaxy(
    target_z: float,
    target_logM: float,
    rng,
    logM_tol: float = 0.3,
    environment: str | None = None,
    max_attempts: int = 10,
) -> dict | None:
    """Pick a TNG100-1 subhalo near ``(target_z, target_logM)``.

    ``environment``, if given, restricts to subhalos whose
    ``classify_environment`` result matches (``isolated``/``pair``/``group``/
    ``rich_group``); up to ``max_attempts`` random candidates are tried.

    Returns a dict of physical properties, or ``None`` if no candidate
    (matching ``environment``, if given) is found in ``max_attempts`` tries.
    """
    snap = nearest_snapshot(target_z)
    snap_z = SNAPSHOT_REDSHIFTS[snap]

    candidates = fetch_subhalo_summaries(snap, target_logM - logM_tol, target_logM + logM_tol)
    if not candidates:
        return None

    tried = set()
    for _ in range(min(max_attempts, len(candidates))):
        remaining = [c for c in candidates if c["id"] not in tried]
        if not remaining:
            break
        c = remaining[int(rng.integers(len(remaining)))]
        tried.add(c["id"])

        detail = fetch_subhalo_detail(snap, c["id"])
        group_massive_count = fetch_group_massive_subhalo_count(snap, detail["grnr"])
        env = classify_environment(group_massive_count)

        if environment is not None and env != environment:
            continue

        return detail_to_record(snap, snap_z, detail, group_massive_count)

    return None
