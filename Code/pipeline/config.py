"""Every parameter of the paper's analysis, in one place."""

from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_DIR / "data"
FIGURE_DIR = CODE_DIR / "figures"

# --- Parameter scan (Sec. III.A): 25 x 25 x 6 = 3750 hairy black holes ----------
SPINS = np.linspace(0.10, 0.85, 25)            # a / M
HAIR_CHARGES = np.linspace(0.05, 1.70, 25)     # ell_0 / M
HAIR_STRENGTHS = np.linspace(0.10, 1.00, 6)    # alpha

# --- Kerr reference family, profiled over a_Kerr (Sec. II.C) --------------------
KERR_SPINS = np.linspace(0.0, 1.0, 40, endpoint=False)   # 0 ... 0.975
KERR_SPIN_MAX = 0.94                                     # upper bound of the fit

# A QNM is accepted only if it solves the continued fraction to this level.
CF_TOLERANCE = 1e-6

# --- Observables -----------------------------------------------------------------
# omega_I is stored as the damping rate -Im(omega) > 0.
OBSERVABLES = ("R_s", "delta_s", "omega_R", "omega_I")
CHANNELS = {
    "Shadow only": ("R_s", "delta_s"),
    "QNM only": ("omega_R", "omega_I"),
    "Combined": OBSERVABLES,
}

# --- Measurement precision scenarios (Table I) ---------------------------------
# delta_s is an absolute uncertainty; the other three are fractional.
SCENARIOS = {
    "Current":    {"R_s": 0.10,  "delta_s": 0.10,  "omega_R": 0.10,  "omega_I": 0.30},
    "Near-term":  {"R_s": 0.03,  "delta_s": 0.03,  "omega_R": 0.03,  "omega_I": 0.15},
    "Next-gen":   {"R_s": 0.01,  "delta_s": 0.01,  "omega_R": 0.01,  "omega_I": 0.05},
    "Optimistic": {"R_s": 0.005, "delta_s": 0.005, "omega_R": 0.001, "omega_I": 0.01},
}
ABSOLUTE_UNCERTAINTY = {"delta_s"}

# --- Sgr A* (Sec. III.E) -----------------------------------------------------------
EHT_SHADOW_DIAMETER = (48.7, 7.0)   # micro-arcsec, EHT 2022
SGR_A_MASS = 4.152e6                # solar masses, GRAVITY 2019
SGR_A_DISTANCE = 8.178              # kpc, GRAVITY 2019
