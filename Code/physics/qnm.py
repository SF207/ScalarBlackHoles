"""
Fundamental quasinormal mode of the hairy black hole (paper Sec. II.B).

Leaver's continued-fraction method, following Li (2023) with the corrected
characteristic exponents of Appendix A. One function per step:

  1. effective_horizons          Delta ~ (r - r+)(r - r-)          Eq. (horizoniter)
  2. characteristic_exponents    p1, p2 at the two horizons        Eqs. (p1)-(p2)
  3. spheroidal_coefficients     B1, B2, eta_cf, B3 of the GSWE    Eq. (gswe)
  4. recurrence_constants        c0..c4 of the 3-term recurrence   Eq. (recurrence)
  5. continued_fraction          its root in omega is the QNM      Eq. (cf)
  6. angular_separation_constant Lambda_lm (Cook & Zalutskiy 2014)
  7. solve_qnm                   Newton's method on step 5

Convention: omega = omega_R + i omega_I with omega_I < 0 (decaying mode).
"""

import numpy as np
import qnm as kerr_qnm   # Stein's Kerr QNM package: starting guess only


# --- 1. Effective horizons ---------------------------------------------------

def effective_horizons(a, alpha, ell_0, M=1.0, tol=1e-12, max_iter=50):
    """Fixed-point iteration r+- = M +- sqrt(M^2 - a^2 - alpha r+-^2 exp(-r+-/L)),
    seeded with the Kerr horizons."""
    r_plus = M + np.sqrt(M**2 - a**2)
    r_minus = M - np.sqrt(M**2 - a**2)
    L = M - ell_0 / 2
    if alpha == 0 or L <= 0:
        return r_plus, r_minus

    for _ in range(max_iter):
        hair_plus = alpha * r_plus**2 * np.exp(-r_plus / L)
        hair_minus = alpha * r_minus**2 * np.exp(-r_minus / L)
        new_plus = M + np.sqrt(max(M**2 - a**2 - hair_plus, 0))
        new_minus = M - np.sqrt(max(M**2 - a**2 - hair_minus, 0))
        converged = abs(new_plus - r_plus) < tol and abs(new_minus - r_minus) < tol
        r_plus, r_minus = new_plus, new_minus
        if converged:
            break
    return r_plus, r_minus


# --- 2. Characteristic exponents ----------------------------------------------

def characteristic_exponents(omega, a, m, r_plus, r_minus, s):
    """Ingoing exponent p1 at r+ and Li's choice p2 at r-.

    The full K(r) = (r^2 + a^2) omega - a m is kept; the reduced form
    omega r - a m only holds for Kerr in units 2M = 1 (Appendix A).
    """
    b = r_plus - r_minus
    K_plus = (r_plus**2 + a**2) * omega - a * m
    K_minus = (r_minus**2 + a**2) * omega - a * m
    p1 = -s - 1j * (K_plus / b)
    p2 = -s + 1j * (K_minus / b)
    return p1, p2


# --- 3. Generalized spheroidal wave equation ----------------------------------

def spheroidal_coefficients(omega, a, m, Lambda, r_plus, r_minus, s):
    """Coefficients of the GSWE obeyed by y(x), with x = r - r- and b = r+ - r-:

        x(x-b) y'' + (B1 + B2 x) y' + [omega^2 x(x-b) - 2 eta omega (x-b) + B3] y = 0.

    B1, B2 and eta have closed forms. B3 is read off numerically: the
    y-coefficient obtained by substituting R = (x-b)^p1 x^p2 y into the radial
    equation is a quadratic in x, so we sample it and fit.
    """
    p1, p2 = characteristic_exponents(omega, a, m, r_plus, r_minus, s)
    b = r_plus - r_minus
    lam = Lambda + a**2 * omega**2 - 2 * a * m * omega

    B1 = (-2 * p2 - s - 1) * b
    B2 = 2 * (p1 + p2 + s + 1)
    eta = -omega * (r_plus + r_minus) - 1j * s

    def y_coefficient(x):
        r = x + r_minus
        Delta, dDelta = x * (x - b), 2 * x - b
        K = (r**2 + a**2) * omega - a * m
        potential = (K**2 - 1j * s * K * dDelta) / Delta + 4j * s * omega * r - lam
        from_R2 = (p1 * (p1 - 1) / (x - b)**2 + 2 * p1 * p2 / ((x - b) * x)
                   + p2 * (p2 - 1) / x**2)
        from_R1 = p1 / (x - b) + p2 / x
        return Delta * from_R2 + (s + 1) * dDelta * from_R1 + potential

    x = np.array([0.3, 0.5, 0.8, 1.0, 1.5])
    _, _, constant = np.polyfit(x, [y_coefficient(xi) for xi in x], 2)
    B3 = constant - 2 * eta * omega * b
    return B1, B2, B3, eta


# --- 4. Three-term recurrence --------------------------------------------------

def recurrence_constants(omega, a, m, Lambda, r_plus, r_minus, s):
    """Constants c0..c4 of Leaver's recurrence alpha_n d_{n+1} + beta_n d_n + gamma_n d_{n-1} = 0,

        alpha_n = n^2 + (c0 + 1) n + c0
        beta_n  = -2 n^2 + (c1 + 2) n + c3
        gamma_n = n^2 + (c2 - 3) n + c4 - c2 + 2.

    Writing y = exp(i omega x) x^sigma0 f(zeta), zeta = (x-b)/x, the GSWE operator
    maps f = zeta^n to  alpha_{n-1} zeta^{n-1} + beta_n zeta^n + gamma_{n+1} zeta^{n+1}.
    Applying it to f = 1 and f = zeta and fitting gives every constant.
    """
    B1, B2, B3, eta = spheroidal_coefficients(omega, a, m, Lambda, r_plus, r_minus, s)
    b = r_plus - r_minus
    sigma0 = -B2 / 2 - 1j * eta
    x = np.array([0.35, 0.55, 0.75, 1.05, 1.35, 1.65])
    zeta = (x - b) / x

    def operator_on_power(n):
        """GSWE operator (with exp(i omega x) x^sigma0 removed) applied to zeta^n."""
        dzeta, d2zeta = b / x**2, -2 * b / x**3
        f = zeta**n
        df = n * zeta**(n - 1) * dzeta if n else 0
        d2f = n * (n - 1) * zeta**(n - 2) * dzeta**2 + n * zeta**(n - 1) * d2zeta if n else 0
        g = f
        dg = sigma0 / x * f + df
        d2g = sigma0 * (sigma0 - 1) / x**2 * f + 2 * sigma0 / x * df + d2f
        return (x * (x - b) * d2g
                + (2j * omega * x * (x - b) + B1 + B2 * x) * dg
                + (1j * omega * (B1 + B2 * x) - 2 * eta * omega * (x - b) + B3) * g)

    def neighbours(n):
        """(alpha_{n-1}, beta_n, gamma_{n+1}) from a least-squares fit."""
        basis = np.column_stack([zeta**(n - 1), zeta**n, zeta**(n + 1)])
        return np.linalg.lstsq(basis, operator_on_power(n), rcond=None)[0]

    _, beta_0, gamma_1 = neighbours(0)
    alpha_0, beta_1, gamma_2 = neighbours(1)
    c0, c3, c4 = alpha_0, beta_0, gamma_1
    c1, c2 = beta_1 - c3, gamma_2 - c4
    return c0, c1, c2, c3, c4


# --- 5. Continued fraction ----------------------------------------------------

def continued_fraction(omega, a, m, Lambda, r_plus, r_minus, s, depth=300):
    """beta_0 - alpha_0 gamma_1 / (beta_1 - alpha_1 gamma_2 / (beta_2 - ...)),
    truncated after `depth` levels and evaluated bottom-up. Zero at a QNM."""
    c0, c1, c2, c3, c4 = recurrence_constants(omega, a, m, Lambda, r_plus, r_minus, s)
    n = np.arange(depth + 1)
    alpha = n**2 + (c0 + 1) * n + c0
    beta = -2 * n**2 + (c1 + 2) * n + c3
    gamma = n**2 + (c2 - 3) * n + c4 - c2 + 2

    tail = beta[depth]
    for k in range(depth - 1, 0, -1):
        if abs(tail) < 1e-30:
            tail = 1e-30 + 0j
        tail = beta[k] - alpha[k] * gamma[k + 1] / tail
    return beta[0] - alpha[0] * (gamma[1] / tail)


# --- 6. Angular equation --------------------------------------------------------
# The hair leaves the angular equation untouched, so Lambda_lm is the Kerr
# spin-weighted spheroidal eigenvalue at c = a omega. Spectral method of
# Cook & Zalutskiy (2014), Eqs. (52) and (55).

def _F(s, l, m):
    l1 = l + 1
    if l1 == 0:
        return 0.0
    return np.sqrt((l1**2 - m**2) / ((2*l + 3) * (2*l + 1))) * np.sqrt((l1**2 - s**2) / l1**2)


def _G(s, l, m):
    if l == 0:
        return 0.0
    return np.sqrt((l**2 - m**2) / (4 * l**2 - 1)) * np.sqrt(1 - s**2 / l**2)


def _H(s, l, m):
    if l == 0 or s == 0:
        return 0.0
    return -m * s / (l * (l + 1))


def _matrix_element(s, c, m, l, lp):
    if lp == l - 2:
        return -c**2 * _F(s, lp, m) * _F(s, lp + 1, m)
    if lp == l - 1:
        return _F(s, lp, m) * (-c**2 * (_H(s, lp + 1, m) + _H(s, lp, m)) + 2 * c * s)
    if lp == l:
        B = (_F(s, lp, m) * _G(s, lp + 1, m) + _G(s, lp, m) * _F(s, lp - 1, m)
             + _H(s, lp, m)**2)
        return lp * (lp + 1) - s * (s + 1) - c**2 * B + 2 * c * s * _H(s, lp, m)
    if lp == l + 1:
        return _G(s, lp, m) * (-c**2 * (_H(s, lp - 1, m) + _H(s, lp, m)) + 2 * c * s)
    if lp == l + 2:
        return -c**2 * _G(s, lp, m) * _G(s, lp - 1, m)
    return 0.0


def angular_separation_constant(omega, a, s=-2, l=2, m=2, size=20):
    """Eigenvalue of the truncated Cook-Zalutskiy matrix that continues
    the spherical value l(l+1) - s(s+1)."""
    c = a * omega
    spherical = l * (l + 1) - s * (s + 1)
    if abs(c) < 1e-12:
        return complex(spherical)
    ls = range(max(abs(m), abs(s)), max(abs(m), abs(s)) + size)
    matrix = np.array([[_matrix_element(s, c, m, li, lj) for lj in ls] for li in ls])
    eigenvalues = np.linalg.eigvals(matrix)
    return eigenvalues[np.argmin(abs(eigenvalues - spherical))]


# --- 7. Newton's method ------------------------------------------------------------

def solve_qnm(a, alpha=0.0, ell_0=1.0, M=1.0, s=-2, l=2, m=2,
              depth=300, tol=1e-10, max_iter=100):
    """Fundamental (n = 0) QNM frequency omega, starting from the Kerr value.

    Stops when the Newton step |d omega| < tol * max(|omega|, 1/M) (Eq. newtontol).
    Returns (omega, info); info["cf_residual"] = |CF(omega)| measures how well
    omega solves the characteristic equation.
    """
    r_plus, r_minus = effective_horizons(a, alpha, ell_0, M)
    omega_kerr = kerr_qnm.modes_cache(s=s, l=l, m=m, n=0)(a / M)[0] / M

    def cf(w, Lambda):
        return continued_fraction(w, a, m, Lambda, r_plus, r_minus, s, depth)

    omega = omega_kerr
    for iteration in range(1, max_iter + 1):
        Lambda = angular_separation_constant(omega, a, s, l, m)
        value = cf(omega, Lambda)
        h = 1e-8 * max(abs(omega), 1 / M)
        slope = (cf(omega + h, Lambda) - value) / h
        if abs(slope) < 1e-30:
            break
        step = value / slope
        omega -= step
        if abs(step) < tol * max(abs(omega), 1 / M):
            break

    Lambda = angular_separation_constant(omega, a, s, l, m)
    info = {
        "r_plus": r_plus, "r_minus": r_minus, "Lambda": Lambda,
        "iterations": iteration, "omega_kerr": omega_kerr,
        "cf_residual": abs(cf(omega, Lambda)),
    }
    return omega, info
