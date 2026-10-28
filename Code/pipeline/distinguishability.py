"""
Distinguishability of hairy black holes from Kerr (paper Secs. II.C, III.B-E).

For each hairy black hole with observables x, and each precision scenario,

    d_min = min over a_Kerr of  sqrt( sum_i [(x_i - y_i(a_Kerr)) / sigma_i]^2 ),

where y(a_Kerr) are the Kerr observables (cubic interpolation over the Kerr
reference family). d_min >= N means no Kerr black hole of any spin fits
within N sigma.

    python -m pipeline.distinguishability      (needs data/ from pipeline.scan)

Prints the numbers quoted in Sec. III and saves Fig. 3 (kerr_dist_merged.png).
All percentages are of the cells with valid observables (3745 of 3750).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, spearmanr

from physics import effective_horizons
from .config import (ABSOLUTE_UNCERTAINTY, CHANNELS, EHT_SHADOW_DIAMETER,
                     KERR_SPIN_MAX, KERR_SPINS, OBSERVABLES, SCENARIOS,
                     SGR_A_DISTANCE, SGR_A_MASS)
from .scan import KERR_FILE, SCAN_FILE
from .style import PAGE, save, use_revtex


# --- distance to the Kerr family -------------------------------------------------

def uncertainties(x, precision):
    """sigma_i for observables x: absolute for delta_s, fractional otherwise."""
    return np.array([precision[name] if name in ABSOLUTE_UNCERTAINTY
                     else precision[name] * abs(value)
                     for name, value in zip(OBSERVABLES, x)])


def distance_to_kerr(x, precision, channel, kerr, a_max=KERR_SPIN_MAX):
    """(d_min, best-fit a_Kerr in [0, a_max]) using only the observables in `channel`."""
    if np.isnan(x).any():
        return np.nan, np.nan
    use = [OBSERVABLES.index(name) for name in channel]
    sigma = uncertainties(x, precision)[use]

    def misfit(a_kerr):
        return np.sum(((x[use] - kerr(a_kerr)[use]) / sigma) ** 2)

    best = minimize_scalar(misfit, bounds=(0.0, a_max), method="bounded")
    return np.sqrt(best.fun), best.x


def distance_maps(observables, kerr, a_max=KERR_SPIN_MAX):
    """d_min and best a_Kerr for every (scenario, channel, cell)."""
    d_min, a_kerr = {}, {}
    cells = observables.reshape(-1, len(OBSERVABLES))
    for scenario, precision in SCENARIOS.items():
        for channel, names in CHANNELS.items():
            result = np.array([distance_to_kerr(x, precision, names, kerr, a_max)
                               for x in cells])
            d_min[scenario, channel] = result[:, 0].reshape(observables.shape[:-1])
            a_kerr[scenario, channel] = result[:, 1].reshape(observables.shape[:-1])
    return d_min, a_kerr


def excluded_percent(d, threshold):
    """Percentage of valid cells (finite d) with d_min >= threshold.
    The denominator is the number of valid cells, not the full grid."""
    valid = np.isfinite(d)
    return 100 * np.sum(d[valid] >= threshold) / valid.sum()


def chi2_threshold(N, n_observables):
    """d_min equivalent to an N-sigma (two-sided Gaussian) significance.

    With one fitted parameter (a_Kerr), d_min^2 follows chi^2 with
    n_observables - 1 degrees of freedom. For two observables this is N itself.
    """
    p = 2 * norm.sf(N)   # 0.0455 for 2 sigma, 0.0027 for 3 sigma
    return np.sqrt(chi2.isf(p, n_observables - 1))


def clipped_horizon_mask(spins, hair_charges, hair_strengths):
    """True where the effective-horizon iteration clipped a negative radicand."""
    return np.array([[[effective_horizons(a, alpha, ell_0)[2] for a in spins]
                      for ell_0 in hair_charges] for alpha in hair_strengths])


# --- reports -------------------------------------------------------------------------

def report_observables(kerr, obs, R_A):
    """Sec. III.B: ranges of the observables, and R_A vs R_s (Sec. III.A).
    Kerr ranges are over the fitted spins [0, KERR_SPIN_MAX]."""
    kerr_fit = kerr(np.linspace(0, KERR_SPIN_MAX, 2000))
    print("\n== Behaviour of the observables ==")
    for k, name in enumerate(OBSERVABLES):
        print(f"  {name:8s} Kerr (a <= {KERR_SPIN_MAX}) "
              f"{kerr_fit[:, k].min():.4f} .. {kerr_fit[:, k].max():.4f}"
              f"   hairy {np.nanmin(obs[..., k]):.4f} .. {np.nanmax(obs[..., k]):.4f}")
    R_s = obs[..., 0]
    diff = 100 * np.abs(R_A - R_s) / R_s
    print(f"  |R_A - R_s| / R_s: median {np.nanmedian(diff):.2f}%, max {np.nanmax(diff):.1f}%")


def print_table(d_min, threshold=lambda N, channel: N, levels=(2, 3), mask=None):
    """Rows: scenarios. Columns: channels. Cells: % excluded at each level."""
    print(f"  {'':12s}" + "".join(f"{c:>14s}" for c in CHANNELS))
    for N in levels:
        print(f"  {N} sigma")
        for scenario in SCENARIOS:
            row = []
            for channel in CHANNELS:
                d = d_min[scenario, channel]
                if mask is not None:
                    d = np.where(mask, np.nan, d)
                row.append(excluded_percent(d, threshold(N, channel)))
            print(f"  {scenario:12s}" + "".join(f"{v:14.1f}" for v in row))


def report_exclusion_table(d_min):
    """Table IV: % of the valid cells excluded, pooled over all alpha slices."""
    n_valid = np.isfinite(d_min["Current", "Combined"]).sum()
    print(f"\n== Table IV: % of the {n_valid} valid black holes with no Kerr fit ==")
    print("  threshold d_min >= N")
    print_table(d_min)

    nu = len(CHANNELS["Combined"]) - 1
    print(f"  chi^2 threshold, nu = n_obs - 1 (Combined: nu = {nu}, d_min >= "
          f"{chi2_threshold(2, nu + 1):.2f} / {chi2_threshold(3, nu + 1):.2f})")
    print_table(d_min, threshold=lambda N, channel: chi2_threshold(N, len(CHANNELS[channel])))


def report_joint_only(d_min, a_kerr, scenarios=("Near-term", "Next-gen")):
    """Sec. III.C: cells excluded only by the combination, and why.

    In those cells the shadow and the ringdown are each matched by a Kerr
    black hole, but by Kerr black holes of different spins.
    """
    print("\n== Joint-only exclusions (Combined >= 3, Shadow < 3, QNM < 3) ==")
    for scenario in scenarios:
        shadow, ringdown, combined = (d_min[scenario, c] for c in CHANNELS)
        valid = np.isfinite(combined)
        joint_only = valid & (combined >= 3) & (shadow < 3) & (ringdown < 3)
        gap = np.abs(a_kerr[scenario, "Shadow only"] - a_kerr[scenario, "QNM only"])
        print(f"  {scenario:10s}: {100 * joint_only.sum() / valid.sum():.1f}% of cells; "
              f"median |a_Kerr(shadow) - a_Kerr(QNM)| = {np.median(gap[joint_only]):.3f} "
              f"(other cells: {np.median(gap[valid & ~joint_only]):.3f})")


def report_clipped_horizons(d_min, obs, clipped, spins, hair_charges, hair_strengths):
    """Sec. III.A: cells whose effective-horizon iteration clipped a negative
    radicand. Their QNMs are unreliable; Table IV is repeated without them."""
    valid = np.isfinite(obs).all(axis=-1)
    flagged = clipped & valid
    alpha, ell_0, a = np.meshgrid(hair_strengths, hair_charges, spins, indexing="ij")
    omega_R = obs[..., OBSERVABLES.index("omega_R")]
    wrong_branch = valid & ((omega_R < 0.3) | (omega_R > 0.9))

    print("\n== Clipped effective horizons ==")
    print(f"  {clipped.sum()} cells clipped, {flagged.sum()} of them with valid observables")
    print(f"  region: a >= {a[flagged].min():.2f}, alpha >= {alpha[flagged].min():.2f}, "
          f"ell_0 <= {ell_0[flagged].max():.2f}")
    print(f"  omega_R outside [0.3, 0.9] (wrong branch): {wrong_branch.sum()} cells, "
          f"{(wrong_branch & clipped).sum()} of them clipped")
    print("  Table IV without clipped cells:")
    print_table(d_min, levels=(3,), mask=clipped)


def report_kerr_spin_limit(obs, kerr, a_kerr, a_max=0.974):
    """Limitations: best fits pinned at the search bound, and the effect of raising it."""
    edge = KERR_SPIN_MAX - 0.005
    print(f"\n== Kerr spin search limit (a_Kerr <= {KERR_SPIN_MAX}) ==")
    for scenario in SCENARIOS:
        counts = [np.sum(a_kerr[scenario, c] > edge) for c in ("QNM only", "Combined")]
        print(f"  {scenario:10s}: best fit a_Kerr > {edge:.3f} in {counts[0]} (QNM) "
              f"/ {counts[1]} (Combined) cells")

    assert a_max <= KERR_SPINS[-1], "a_max must lie inside the Kerr reference table"
    print(f"  Table IV with a_Kerr <= {a_max}:")
    print_table(distance_maps(obs, kerr, a_max)[0], levels=(3,))


def report_parameter_dependence(d_min, a_kerr, spins, hair_charges, hair_strengths,
                                scenario="Next-gen"):
    """Sec. III.D: which parameter controls detectability."""
    d = d_min[scenario, "Combined"]
    print(f"\n== Detectability vs parameters ({scenario}, combined) ==")

    alpha, ell_0, a = np.meshgrid(hair_strengths, hair_charges, spins, indexing="ij")
    valid = np.isfinite(d)
    for name, values in [("ell_0", ell_0), ("alpha", alpha), ("a/M", a)]:
        print(f"  Spearman(d_min, {name:5s}) = {spearmanr(d[valid], values[valid])[0]:+.3f}")

    print("  excluded at 3 sigma, per alpha slice:")
    for k, al in enumerate(hair_strengths):
        print(f"    alpha = {al:.2f}: {excluded_percent(d[k], 3):5.1f}%")

    k, i, j = np.unravel_index(np.nanargmin(d), d.shape)
    print(f"  most Kerr-like: (a, alpha, ell_0) = ({spins[j]:.2f}, {hair_strengths[k]:.2f}, "
          f"{hair_charges[i]:.2f}), d_min = {d[k, i, j]:.1e}, "
          f"a_Kerr = {a_kerr[scenario, 'Combined'][k, i, j]:.3f}")


def report_hair_reach(d_min, hair_charges, hair_strengths):
    """Sec. III.D: how far out the hair must reach to be seen."""
    print("\n== Reach of detectable hair ==")
    d_next = d_min["Next-gen", "Combined"]
    d_best = d_min["Optimistic", "Combined"]
    for k, alpha in enumerate(hair_strengths):
        excluded_rows = np.flatnonzero((d_best[k] >= 3).any(axis=1))
        largest = f"{hair_charges[excluded_rows].max():.2f}" if len(excluded_rows) else "none"
        print(f"  alpha = {alpha:.2f}: max d_min at ell_0 = {hair_charges[-1]:.2f} (Next-gen) "
              f"= {np.nanmax(d_next[k, -1]):.3f};  largest ell_0 excluded at 3 sigma "
              f"(Optimistic) = {largest}")


def report_eht(obs):
    """Sec. III.E: hairy shadows against the EHT diameter of Sgr A*."""
    G_MSUN_OVER_C2 = 1476.625          # m
    KPC = 3.0856775814913673e19        # m
    MICROARCSEC = np.pi / 180 / 3600 / 1e6
    theta_g = SGR_A_MASS * G_MSUN_OVER_C2 / (SGR_A_DISTANCE * KPC) / MICROARCSEC

    mean, err = EHT_SHADOW_DIAMETER
    R_s = obs[..., 0]
    print("\n== Sgr A* shadow ==")
    print(f"  one gravitational radius = {theta_g:.2f} uas")
    print(f"  Schwarzschild diameter   = {2 * np.sqrt(27) * theta_g:.1f} uas")
    print(f"  hairy R_s in [{np.nanmin(R_s):.3f}, {np.nanmax(R_s):.3f}] M  ->  "
          f"[{2 * np.nanmin(R_s) * theta_g:.1f}, {2 * np.nanmax(R_s) * theta_g:.1f}] uas")
    print(f"  EHT 1-sigma interval     = [{mean - err:.1f}, {mean + err:.1f}] uas")


# --- Fig. 3 ----------------------------------------------------------------------

def precision_label(p):
    return (f"$\\sigma_{{R_s}}$={100 * p['R_s']:g}%\n"
            f"$\\sigma_{{\\delta_s}}$={p['delta_s']:g}\n"
            f"$\\sigma_{{Re}}$={100 * p['omega_R']:g}%\n"
            f"$\\sigma_{{Im}}$={100 * p['omega_I']:g}%")


def plot_maps(d_min, spins, hair_charges, hair_strengths):
    """Rows: precision scenarios. Columns: alpha slices. Combined channel."""
    use_revtex(base=7)
    n_rows, n_cols = len(SCENARIOS), len(hair_strengths)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(PAGE, PAGE * 0.70),
                             sharex=True, sharey=True)

    for r, (scenario, precision) in enumerate(SCENARIOS.items()):
        for c, alpha in enumerate(hair_strengths):
            ax, d = axes[r, c], d_min[scenario, "Combined"][c]
            filled = ax.contourf(spins, hair_charges, d, levels=np.linspace(0, 6, 25),
                                 cmap="viridis", extend="max")
            lines = ax.contour(spins, hair_charges, d, levels=[1, 3, 5],
                               colors="white", linewidths=1.0)
            ax.clabel(lines, inline=True, fontsize=5, fmt=lambda v: f"{int(v)}$\\sigma$")
            if r == 0:
                ax.set_title(fr"$\alpha$ = {alpha:.2f}", pad=3)
            if r == n_rows - 1:
                ax.set_xlabel("Spin  $a/M$")
            if c == 0:
                ax.set_ylabel(r"$\ell_0/M$")
                ax.text(-1.15, 0.5, scenario, transform=ax.transAxes, fontsize=8,
                        fontweight="bold", rotation=90, va="center", ha="center")
                ax.text(-0.70, 0.5, precision_label(precision), transform=ax.transAxes,
                        fontsize=5.5, va="center", ha="center")

    fig.subplots_adjust(left=0.145, right=0.90, top=0.92, bottom=0.09,
                        hspace=0.12, wspace=0.08)
    colorbar = fig.colorbar(filled, cax=fig.add_axes([0.915, 0.09, 0.013, 0.83]))
    colorbar.set_label(r"$\sigma$ to nearest Kerr (Mahalanobis distance)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    fig.suptitle("Distinguishability of GD hairy Kerr black holes from Kerr",
                 fontsize=9, y=0.995)
    save(fig, "kerr_dist_merged.png")
    plt.close(fig)


if __name__ == "__main__":
    kerr_data, scan = np.load(KERR_FILE), np.load(SCAN_FILE)
    kerr = interp1d(kerr_data["spins"], kerr_data["observables"], kind="cubic", axis=0)
    obs = scan["observables"]
    grid = scan["spins"], scan["hair_charges"], scan["hair_strengths"]

    valid = np.isfinite(obs).all(axis=-1)
    print(f"{valid.sum()} of {valid.size} parameter combinations have valid observables")

    print("Profiling over a_Kerr for every scenario and channel...")
    d_min, a_kerr = distance_maps(obs, kerr)

    report_observables(kerr, obs, scan["R_A"])
    report_exclusion_table(d_min)
    report_joint_only(d_min, a_kerr)
    report_clipped_horizons(d_min, obs, clipped_horizon_mask(*grid), *grid)
    report_kerr_spin_limit(obs, kerr, a_kerr)
    report_parameter_dependence(d_min, a_kerr, *grid)
    report_hair_reach(d_min, *grid[1:])
    report_eht(obs)
    plot_maps(d_min, *grid)
