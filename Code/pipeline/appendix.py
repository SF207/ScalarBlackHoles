"""
Appendix A: why Li's reduced exponents fail for hairy horizons.

  1. Kerr identity r+^2 + a^2 = r+ in Leaver's units (2M = 1).
  2. The effective hairy horizons break r+ + r- = 2M and r+ r- = a^2 (M = 1, a = 0.5).
  3. Relative difference between K(r+) = (r+^2 + a^2) omega - a m and the
     reduced form omega r+ - a m, in Leaver's units, at omega M = 0.5 - 0.08i, m = 2.
All at ell_0 = 1.0 M.

    python -m pipeline.appendix
"""

import numpy as np

from physics import effective_horizons

ELL_0 = 1.0   # in units of M


def kerr_identity():
    M = 0.5
    worst = max(abs(rp**2 + a**2 - rp)
                for a in np.linspace(0, 0.9, 10) * M
                for rp, _ in [effective_horizons(a, 0.0, ELL_0 * M, M)])
    print(f"Kerr, 2M = 1: max |r+^2 + a^2 - r+| over a/M in [0, 0.9] = {worst:.1e}")


def horizon_relations(a=0.5):
    print(f"\nEffective horizons at M = 1, a = {a}")
    print(f"{'alpha':>5} | {'r+ + r-':>8} | {'r+ r-':>7} | (r+ + r-)^2 vs (2M)^2")
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8):
        rp, rm = effective_horizons(a, alpha, ELL_0)
        print(f"{alpha:5.1f} | {rp + rm:8.4f} | {rp * rm:7.4f} | "
              f"{100 * abs((rp + rm)**2 - 4) / 4:.2f}%")


def reduced_exponent_error(omega_M=0.5 - 0.08j, m=2):
    M = 0.5                       # Leaver's units
    omega = omega_M / M
    alphas = (0.0, 0.2, 0.4, 0.6)
    print(f"\n|K_full - K_reduced| / |K_full| at omega M = {omega_M}, m = {m}")
    print(f"{'a/M':>5} | " + " | ".join(f"alpha={al:<4}" for al in alphas))
    for spin in (0.3, 0.6, 0.9):
        a = spin * M
        row = []
        for alpha in alphas:
            rp, _ = effective_horizons(a, alpha, ELL_0 * M, M)
            full = (rp**2 + a**2) * omega - a * m
            reduced = omega * rp - a * m
            row.append(100 * abs(full - reduced) / abs(full))
        print(f"{spin:5.1f} | " + " | ".join(f"{v:9.2f}%" for v in row))


if __name__ == "__main__":
    kerr_identity()
    horizon_relations()
    reduced_exponent_error()
