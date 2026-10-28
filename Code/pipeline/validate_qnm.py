"""
Validation of the QNM solver (paper Sec. III.A, Tables II and III).

  Table II   Kerr limit (alpha = 0) against Stein's `qnm` package.
  Table III  Hairy modes against Table I of Li (2023), ell_0 = 1.

    python -m pipeline.validate_qnm
"""

import numpy as np
import qnm as kerr_qnm

from physics import angular_separation_constant, solve_qnm

# Li (2023) Table I: fundamental s = -2, l = m = 2 mode at ell_0 = 1.
# (a/M, alpha): (Re omega, -Im omega)
LI_2023_TABLE_I = {
    (0.1, 0.0): (0.387018, 0.088706), (0.1, 0.1): (0.387514, 0.089016),
    (0.1, 0.5): (0.389570, 0.090307), (0.1, 1.0): (0.392310, 0.092044),
    (0.2, 0.0): (0.402145, 0.088311), (0.2, 0.1): (0.402704, 0.088634),
    (0.2, 0.5): (0.405023, 0.089980), (0.2, 1.0): (0.408122, 0.091795),
    (0.3, 0.0): (0.419527, 0.087729), (0.3, 0.1): (0.420173, 0.088067),
    (0.3, 0.5): (0.422858, 0.089476), (0.3, 1.0): (0.426465, 0.091385),
    (0.4, 0.0): (0.439842, 0.086882), (0.4, 0.1): (0.440609, 0.087229),
    (0.4, 0.5): (0.443813, 0.088683), (0.4, 1.0): (0.448146, 0.090663),
    (0.5, 0.0): (0.464123, 0.085639), (0.5, 0.1): (0.465067, 0.085980),
    (0.5, 0.5): (0.469026, 0.087410), (0.5, 1.0): (0.474434, 0.089363),
    (0.6, 0.0): (0.494045, 0.083765), (0.6, 0.1): (0.495261, 0.084067),
    (0.6, 0.5): (0.500398, 0.085320), (0.6, 1.0): (0.507520, 0.087007),
    (0.7, 0.0): (0.532600, 0.080793), (0.7, 0.1): (0.534283, 0.080988),
    (0.7, 0.5): (0.541479, 0.081742), (0.7, 1.0): (0.551710, 0.082628),
    (0.8, 0.0): (0.586017, 0.075630), (0.8, 0.1): (0.588656, 0.075563),
    (0.8, 0.5): (0.600233, 0.075057), (0.8, 1.0): (0.617659, 0.073752),
    (0.9, 0.0): (0.671614, 0.064869), (0.9, 0.1): (0.677072, 0.064026),
    (0.9, 0.5): (0.703157, 0.059362), (0.9, 1.0): (0.753330, 0.048227),
}


def fmt(w):
    return f"{w.real:.6f} {w.imag:+.6f}i"


def kerr_limit():
    print("Table II: Kerr limit vs the qnm package (s = -2, l = m = 2, n = 0)")
    print(f"{'a/M':>5} | {'omega M':>22} | {'|CF|':>7} | {'|d omega| M':>11} | {'|d Lambda|':>10}")
    mode = kerr_qnm.modes_cache(s=-2, l=2, m=2, n=0)
    for a in np.arange(0.1, 1.0, 0.1).round(1):
        omega_ref, Lambda_ref, _ = mode(a)
        omega, info = solve_qnm(a=a)
        Lambda = angular_separation_constant(omega_ref, a)
        print(f"{a:5.1f} | {fmt(omega):>22} | {info['cf_residual']:7.0e} | "
              f"{abs(omega - omega_ref):11.1e} | {abs(Lambda - Lambda_ref):10.1e}")


def li_comparison():
    print("\nTable III: hairy modes vs Li (2023) Table I (ell_0 = 1)")
    print(f"{'a/M':>5} {'alpha':>5} | {'Li (2023)':>22} | {'this work':>22} | "
          f"{'|CF|':>7} | rel. dev.")
    for (a, alpha), (re, minus_im) in LI_2023_TABLE_I.items():
        omega_li = complex(re, -minus_im)
        omega, info = solve_qnm(a=a, alpha=alpha, ell_0=1.0)
        deviation = abs(omega - omega_li) / abs(omega_li)
        print(f"{a:5.1f} {alpha:5.1f} | {fmt(omega_li):>22} | {fmt(omega):>22} | "
              f"{info['cf_residual']:7.0e} | {100 * deviation:.2f}%")


if __name__ == "__main__":
    kerr_limit()
    li_comparison()
