"""
Validation of the shadow calculation (paper Sec. III.A) and Fig. 1.

  1. Schwarzschild: R_s = 3 sqrt(3) M and delta_s = 0.
  2. Kerr: photon-orbit radii against r = 2M[1 + cos(2/3 arccos(-+a/M))].
  3. Convergence of R_s and delta_s in the number of boundary points.
  4. Fig. 1: reproduction of Fig. 1 of Contreras, Ovalle & Casadio (2021).

    python -m pipeline.validate_shadow
"""

import matplotlib.pyplot as plt
import numpy as np

from physics import HairyKerr, photon_orbit_range, shadow_boundary, shadow_observables
from .style import COLUMN, save, use_revtex


def schwarzschild():
    obs = shadow_observables(HairyKerr())
    exact = 3 * np.sqrt(3)
    print("Schwarzschild")
    print(f"  R_s = {obs['R_s']:.9f} M  (exact {exact:.9f}, rel. dev. "
          f"{abs(obs['R_s'] - exact) / exact:.1e}),  delta_s = {obs['delta_s']:.1e}")


def kerr_photon_orbits():
    worst = 0.0
    for a in np.linspace(0.1, 0.99, 90):
        exact = [2 * (1 + np.cos(2 / 3 * np.arccos(sign * a))) for sign in (-1, +1)]
        found = photon_orbit_range(HairyKerr(a=a))
        worst = max(worst, *(abs(f - e) / e for f, e in zip(found, exact)))
    print(f"Kerr photon orbits, a/M in [0.1, 0.99]: max rel. error {worst:.1e}")


def sampling_convergence(a=0.5, alpha=0.4, ell_0=1.0):
    print(f"Boundary sampling at (a, alpha, ell_0) = ({a}, {alpha}, {ell_0})")
    bh = HairyKerr(a=a, alpha=alpha, ell_0=ell_0)
    for n in (100, 250, 500, 1000, 2000):
        obs = shadow_observables(bh, n_points=n)
        print(f"  n = {n:4d}: R_s = {obs['R_s']:.8f}, delta_s = {obs['delta_s']:.8f}")


def contreras_figure(a=0.3, alpha=0.6):
    """Fig. 1: shadows for ell = 0.2, 0.8 and ell_K = 2M, with ell_0 = alpha * ell."""
    cases = [(0.2, "black", r"$\ell = 0.2$; $r_H < r_{\rm Kerr}$"),
             (0.8, "blue", r"$\ell = 0.8$; $r_H < r_{\rm Kerr}$"),
             (2.0, "red", r"$\ell = \ell_K$; $r_H = r_{\rm Kerr}$")]

    use_revtex(base=8)
    fig, ax = plt.subplots(figsize=(COLUMN, COLUMN))
    for ell, color, label in cases:
        sigma, beta = shadow_boundary(HairyKerr(a=a, alpha=alpha, ell_0=alpha * ell))
        ax.plot(sigma, beta, color=color, label=label)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set(xlabel=r"$\sigma/\mathcal{M}$", ylabel=r"$\beta/\mathcal{M}$",
           xlim=(-5, 6), ylim=(-6, 6), aspect="equal")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.3)
    save(fig, "TST_contreras_2021_figure_1.png")
    plt.close(fig)


if __name__ == "__main__":
    schwarzschild()
    kerr_photon_orbits()
    sampling_convergence()
    contreras_figure()
