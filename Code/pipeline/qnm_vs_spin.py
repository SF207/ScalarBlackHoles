"""
Fig. 2: fundamental QNM frequency against spin at ell_0 = 1, for four alpha,
and the Sec. III.B example of a Kerr black hole copying a hairy ringdown.

    python -m pipeline.qnm_vs_spin
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from physics import solve_qnm
from .config import CF_TOLERANCE
from .style import COLUMN, save, use_revtex

SPINS = np.linspace(0.05, 0.90, 18)
CURVES = [(0.0, "black", "-"), (0.1, "blue", "--"), (0.3, "orange", "-."), (0.5, "red", ":")]


def qnm_curve(alpha, ell_0=1.0):
    """Spins and frequencies of the converged modes along one curve."""
    points = []
    for a in SPINS:
        omega, info = solve_qnm(a=a, alpha=alpha, ell_0=ell_0)
        if info["cf_residual"] < CF_TOLERANCE:
            points.append((a, omega))
    a, omega = map(np.array, zip(*points))
    return a, omega


def kerr_mimic(a=0.9, alpha=0.5, ell_0=1.0):
    """Kerr spin with the same omega_R as the hairy black hole, and its damping."""
    hairy, _ = solve_qnm(a=a, alpha=alpha, ell_0=ell_0)
    a_kerr = brentq(lambda s: solve_qnm(a=s)[0].real - hairy.real, a, 0.975)
    kerr, _ = solve_qnm(a=a_kerr)
    print(f"Hairy (a, alpha, ell_0) = ({a}, {alpha}, {ell_0}): omega_R = {hairy.real:.6f}, "
          f"-omega_I = {-hairy.imag:.4f}")
    print(f"Kerr with the same omega_R: a_Kerr = {a_kerr:.3f}, -omega_I = {-kerr.imag:.4f}")


if __name__ == "__main__":
    use_revtex(base=8)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN, COLUMN * 1.05), sharex=True)

    for alpha, color, ls in CURVES:
        a, omega = qnm_curve(alpha)
        style = dict(color=color, ls=ls, lw=1.0, marker="o", ms=2,
                     label=f"alpha={alpha}" if alpha else "Kerr")
        top.plot(a, omega.real, **style)
        bottom.plot(a, -omega.imag, **style)

    top.set(ylabel=r"Re($\omega$) $M$")
    top.set_title("QNM oscillation frequency", pad=3)
    bottom.set(xlabel="Spin  $a/M$", ylabel=r"$-$Im($\omega$) $M$")
    bottom.set_title("QNM damping rate", pad=3)
    for ax in (top, bottom):
        ax.grid(True, alpha=0.3)
    top.legend()
    fig.tight_layout(pad=0.3)
    save(fig, "qnm_vs_spin.png")
    plt.close(fig)

    kerr_mimic()
