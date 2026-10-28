"""
Black hole shadow and its two observables (paper Sec. II.A).

Each unstable spherical photon orbit of radius r maps to one point
(sigma, beta) on the shadow boundary through the impact parameters

    xi(r)  = a + r^2/a - 4 (a^2 + r^2 F) / (a (r F' + 2F))
    eta(r) = r^2 [r^2 + 2a(a - xi) - (a - xi)^2 F] / (a^2 + r^2 F)

    sigma  = -xi / sin(theta_0)
    beta   = sqrt(eta - xi^2 cot^2(theta_0) + a^2 cos^2(theta_0)).

Sweeping r between the prograde and retrograde equatorial orbits (the roots
of eta = 0) traces the whole boundary. The boundary is then reduced to
    R_s      Hioki-Maeda radius of the reference circle,
    delta_s  distortion relative to that circle,
    R_A      area-equivalent radius sqrt(A/pi), a cross-check on R_s.
"""

import numpy as np
from scipy.optimize import brentq


def impact_parameters(bh, r):
    """(xi, eta) of the spherical photon orbit at radius r. Needs a != 0."""
    a, F, dF = bh.a, bh.F(r), bh.F_prime(r)
    xi = a + r**2 / a - 4 * (a**2 + r**2 * F) / (a * (r * dF + 2 * F))
    a_xi = a - xi
    eta = r**2 * (r**2 + 2 * a * a_xi - a_xi**2 * F) / (a**2 + r**2 * F)
    return xi, eta


def photon_sphere(bh):
    """Non-rotating case: the photon sphere is the root of r F' - 2F = 0."""
    condition = lambda r: float(r * bh.F_prime(r) - 2 * bh.F(r))
    return _first_root(condition, bh.horizon + 0.01 * bh.M, 8 * bh.M)


def photon_orbit_range(bh):
    """Radii (r_pro, r_retro) of the prograde and retrograde equatorial orbits.

    eta > 0 between them and eta < 0 outside, so we bracket the two sign
    changes of eta on a dense grid and refine each with Brent's method.
    """
    if bh.a == 0:
        r_ph = photon_sphere(bh)
        return r_ph, r_ph

    eta = lambda r: float(impact_parameters(bh, r)[1])
    r = np.linspace(bh.horizon + 0.001 * bh.M, 8 * bh.M, 2000)
    values = impact_parameters(bh, r)[1]
    neg, pos = values < 0, values >= 0

    rising = np.flatnonzero(neg[:-1] & pos[1:])    # eta: - -> +
    falling = np.flatnonzero(pos[:-1] & neg[1:])   # eta: + -> -
    if len(rising) == 0 or len(falling) == 0:
        raise ValueError(f"no photon orbits for {bh}")
    i, j = rising[0], falling[-1]
    return brentq(eta, r[i], r[i + 1]), brentq(eta, r[j], r[j + 1])


def shadow_boundary(bh, n_points=500, theta_0=np.pi / 2):
    """Closed shadow boundary (sigma, beta) seen from inclination theta_0."""
    r_pro, r_retro = photon_orbit_range(bh)

    if bh.a == 0:   # spherical symmetry: a circle of radius b_c
        b_c = r_pro / np.sqrt(float(bh.F(r_pro)))
        phi = np.linspace(0, 2 * np.pi, 2 * n_points)
        return b_c * np.cos(phi), b_c * np.sin(phi)

    eps = 1e-6 * bh.M
    r = np.linspace(r_pro + eps, r_retro - eps, n_points)
    xi, eta = impact_parameters(bh, r)
    sigma = -xi / np.sin(theta_0)
    beta_sq = eta - xi**2 / np.tan(theta_0)**2 + bh.a**2 * np.cos(theta_0)**2
    beta = np.sqrt(np.clip(beta_sq, 0, None))

    # upper half left -> right, then lower half right -> left
    return (np.concatenate([sigma, sigma[::-1]]),
            np.concatenate([beta, -beta[::-1]]))


def shadow_observables(bh, n_points=500, theta_0=np.pi / 2):
    """R_s and delta_s (Hioki & Maeda 2009), plus the area radius R_A."""
    sigma, beta = shadow_boundary(bh, n_points, theta_0)

    sigma_r = sigma.max()             # rightmost point (sigma_r, 0)
    sigma_l = sigma.min()             # leftmost point of the true boundary
    top = np.argmax(beta)             # topmost point (sigma_t, beta_t)
    sigma_t, beta_t = sigma[top], beta[top]

    # reference circle through the rightmost, top and bottom points
    if abs(sigma_r - sigma_t) > 1e-10:
        sigma_c = (sigma_r**2 - sigma_t**2 - beta_t**2) / (2 * (sigma_r - sigma_t))
    else:
        sigma_c = (sigma_r + sigma_l) / 2
    R_s = abs(sigma_r - sigma_c)
    delta_s = abs((sigma_c - R_s) - sigma_l) / R_s

    area = 0.5 * abs(np.sum(sigma[:-1] * beta[1:] - sigma[1:] * beta[:-1]))  # shoelace
    return {"R_s": R_s, "delta_s": delta_s, "R_A": np.sqrt(area / np.pi)}


def _first_root(f, r_min, r_max, n=2000):
    r = np.linspace(r_min, r_max, n)
    values = [f(x) for x in r]
    for i in range(n - 1):
        if values[i] * values[i + 1] < 0:
            return brentq(f, r[i], r[i + 1])
    raise ValueError("no root found")
