"""
Compute the four observables for the Kerr family and for every hairy black
hole in the parameter grid (paper Secs. II.C and III.A).

    python -m pipeline.scan            # skips files that already exist
    python -m pipeline.scan --force    # recompute everything (~5 min)

Writes
    data/kerr_reference.npz   spins (40,), observables (40, 4)
    data/hairy_scan.npz       spins, hair_charges, hair_strengths,
                              observables (n_alpha, n_ell0, n_a, 4), R_A (n_alpha, n_ell0, n_a)
Observable order is config.OBSERVABLES. Cells without a horizon, photon
orbits or converged QNM are NaN.
"""

import argparse

import numpy as np

from physics import HairyKerr, shadow_observables, solve_qnm
from .config import (CF_TOLERANCE, DATA_DIR, HAIR_CHARGES, HAIR_STRENGTHS,
                     KERR_SPINS, SPINS)

KERR_FILE = DATA_DIR / "kerr_reference.npz"
SCAN_FILE = DATA_DIR / "hairy_scan.npz"


def observables(a, alpha=0.0, ell_0=1.0):
    """[R_s, delta_s, omega_R, omega_I] and R_A for one black hole.

    Raises ValueError if the black hole has no horizon or photon orbits,
    or if the QNM solver does not converge.
    """
    shadow = shadow_observables(HairyKerr(a=a, alpha=alpha, ell_0=ell_0))
    omega, info = solve_qnm(a=a, alpha=alpha, ell_0=ell_0)
    if not info["cf_residual"] < CF_TOLERANCE:
        raise ValueError("QNM did not converge")
    return [shadow["R_s"], shadow["delta_s"], omega.real, -omega.imag], shadow["R_A"]


def scan_kerr():
    print(f"Kerr family: {len(KERR_SPINS)} spins")
    obs = np.array([observables(a)[0] for a in KERR_SPINS])
    np.savez(KERR_FILE, spins=KERR_SPINS, observables=obs)


def scan_hairy():
    shape = (len(HAIR_STRENGTHS), len(HAIR_CHARGES), len(SPINS))
    obs = np.full(shape + (4,), np.nan)
    R_A = np.full(shape, np.nan)

    for k, alpha in enumerate(HAIR_STRENGTHS):
        print(f"alpha = {alpha:.2f}  ({k + 1}/{len(HAIR_STRENGTHS)})")
        for i, ell_0 in enumerate(HAIR_CHARGES):
            for j, a in enumerate(SPINS):
                try:
                    obs[k, i, j], R_A[k, i, j] = observables(a, alpha, ell_0)
                except ValueError:
                    pass   # no horizon / photon orbits / converged QNM: leave NaN

    valid = np.isfinite(obs).all(axis=-1)
    print(f"valid cells: {valid.sum()} / {valid.size}")
    for alpha, fraction in zip(HAIR_STRENGTHS, valid.mean(axis=(1, 2))):
        print(f"  alpha = {alpha:.2f}: {100 * fraction:.1f}% valid")

    np.savez(SCAN_FILE, spins=SPINS, hair_charges=HAIR_CHARGES,
             hair_strengths=HAIR_STRENGTHS, observables=obs, R_A=R_A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="recompute existing files")
    force = parser.parse_args().force

    DATA_DIR.mkdir(exist_ok=True)
    with np.errstate(all="ignore"):   # degenerate horizons produce NaNs; handled above
        for path, scan in [(KERR_FILE, scan_kerr), (SCAN_FILE, scan_hairy)]:
            if path.exists() and not force:
                print(f"{path.name} exists, skipping (use --force to recompute)")
            else:
                scan()
                print(f"saved data/{path.name}")
