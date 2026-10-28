"""
Rotating black hole with primary scalar hair (paper Sec. II).

The metric is Kerr in Gurses-Gursey form with the constant mass M replaced by
the mass function

    m(r) = M - (alpha / 2) r exp(-r / L),        L = M - ell_0 / 2,

so that

    Delta(r) = r^2 - 2 r m(r) + a^2,
    F(r)     = 1 - 2 m(r) / r.

The Kerr limit is alpha -> 0 or ell_0 -> 2M (L -> 0).
Units: G = c = 1. All scripts set M = 1.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import brentq


@dataclass
class HairyKerr:
    a: float = 0.0       # spin  J/M
    alpha: float = 0.0   # hair strength (amplitude)
    ell_0: float = 0.0   # primary hair charge (sets the radial reach L)
    M: float = 1.0

    def __post_init__(self):
        if self.ell_0 > 2 * self.M:
            raise ValueError("asymptotic flatness requires ell_0 <= 2M")

    @property
    def L(self):
        """Decay length of the hair."""
        return self.M - self.ell_0 / 2

    @property
    def has_hair(self):
        return self.alpha != 0 and self.L > 0

    # --- mass function and metric functions ---------------------------------

    def mass(self, r):
        r = np.asarray(r, dtype=float)
        if not self.has_hair:
            return np.full_like(r, self.M)
        return self.M - 0.5 * self.alpha * r * np.exp(-r / self.L)

    def mass_prime(self, r):
        r = np.asarray(r, dtype=float)
        if not self.has_hair:
            return np.zeros_like(r)
        return -0.5 * self.alpha * np.exp(-r / self.L) * (1 - r / self.L)

    def Delta(self, r):
        return r**2 + self.a**2 - 2 * r * self.mass(r)

    def F(self, r):
        return 1 - 2 * self.mass(r) / r

    def F_prime(self, r):
        return 2 * self.mass(r) / r**2 - 2 * self.mass_prime(r) / r

    # --- event horizon ------------------------------------------------------

    @cached_property
    def horizon(self):
        """Outer horizon: the largest root of Delta(r) = 0 (Eq. horizon).

        Delta is transcendental, so we scan inward from r = 6M for the first
        sign change and refine it with Brent's method.
        """
        r = np.linspace(0.001 * self.M, 6 * self.M, 2000)
        D = self.Delta(r)
        for i in range(len(r) - 1, 0, -1):
            if D[i] * D[i - 1] < 0:
                return brentq(lambda x: float(self.Delta(x)), r[i - 1], r[i])
        raise ValueError(f"no horizon for {self}")
