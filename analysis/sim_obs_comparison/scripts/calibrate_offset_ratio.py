#!/usr/bin/env python3
"""
Calibrate the source-position offset_ratio distribution
(offset = offset_ratio * theta_E) used in
src/jwst_lens_simulator.py::create_parameter_variations() against the
target quad-image fraction from the real 435-lens COSMOS-Web sample (~0.86).

For each candidate distribution, draws SIE+SHEAR lens parameters matching
the ranges used elsewhere in create_parameter_variations()
(lens_q ~ clip(lognormal(ln 0.7, 0.1), 0.2, 1), shear ~ U(0.01, 0.05)),
places a source at offset_ratio * theta_E and a random angle, and solves the
lens equation with lenstronomy's LensEquationSolver to count images.

Background: see analysis/sim_obs_comparison/reports/phase2_noise_and_color_fixes.md,
section 10. Beta(1, 8) * 0.6 (mean offset_ratio ~0.065) gives ~83-89% quad+
(n_img >= 4), matching real. The previous Beta(1.5, 4) * 0.6 (mean ~0.16)
gave only ~51%.

Usage:
    conda run -n astro-clean python calibrate_offset_ratio.py
"""

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import collections

lens_model = LensModel(["SIE", "SHEAR"])
solver = LensEquationSolver(lens_model)


def run(offset_sampler, n_trials=300, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        theta_E = rng.uniform(0.3, 1.5)
        q = np.clip(rng.lognormal(np.log(0.7), 0.1), 0.2, 1.0)
        phi = rng.uniform(0, np.pi)
        e1 = (1 - q) / (1 + q) * np.cos(2 * phi)
        e2 = (1 - q) / (1 + q) * np.sin(2 * phi)
        shear_g = rng.uniform(0.01, 0.05)
        shear_phi = rng.uniform(0, np.pi)
        g1 = shear_g * np.cos(2 * shear_phi)
        g2 = shear_g * np.sin(2 * shear_phi)
        kwargs = [
            {"theta_E": theta_E, "e1": e1, "e2": e2, "center_x": 0, "center_y": 0},
            {"gamma1": g1, "gamma2": g2, "ra_0": 0, "dec_0": 0},
        ]

        offset_ratio = offset_sampler(rng)
        angle = rng.uniform(0, 2 * np.pi)
        offset = offset_ratio * theta_E
        sx, sy = offset * np.cos(angle), offset * np.sin(angle)

        xs, ys = solver.image_position_from_source(
            sx, sy, kwargs, min_distance=0.01, search_window=theta_E * 4
        )
        results.append((offset_ratio, q, len(xs), theta_E))

    arr = np.array(results)
    quad_frac = (arr[:, 2] >= 4).mean()
    return quad_frac, arr[:, 0].mean(), collections.Counter(arr[:, 2].astype(int))


SAMPLERS = {
    "Beta(1.5,4)*0.6 (mean0.16, previous)": lambda rng: rng.beta(1.5, 4.0) * 0.6,
    "Beta(1,5)*0.6 (mean0.083)": lambda rng: rng.beta(1, 5.0) * 0.6,
    "Beta(1,8)*0.6 (mean0.065, current)": lambda rng: rng.beta(1, 8.0) * 0.6,
    "Beta(1,5)*0.4 (mean0.056)": lambda rng: rng.beta(1, 5.0) * 0.4,
    "Beta(0.7,5)*0.5": lambda rng: rng.beta(0.7, 5.0) * 0.5,
    "uniform(0,0.15)": lambda rng: rng.uniform(0, 0.15),
}


def main():
    print(f"{'sampler':<35} {'quad+ frac':>10} {'mean offset_ratio':>18}  n_img distribution")
    for name, sampler in SAMPLERS.items():
        for seed in (42,):
            frac, mean_r, counts = run(sampler, seed=seed)
            print(f"{name:<35} {frac:>10.3f} {mean_r:>18.4f}  {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
