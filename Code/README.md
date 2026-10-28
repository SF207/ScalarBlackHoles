# Scalar-hair black holes: shadow + ringdown

Code for *Distinguishing Scalar-Haired Rotating Black Holes from Kerr Using Shadow and Ringdown Observations*.

For a rotating black hole with primary scalar hair (mass function
`m(r) = M - (alpha/2) r exp(-r/(M - ell_0/2))`), we compute the shadow
radius `R_s`, its distortion `delta_s`, and the fundamental `l = m = 2`
quasinormal frequency, then ask whether any Kerr black hole can mimic them.

## Setup

```
pip install -r requirements.txt
```

## Reproduce the paper

Run from this `Code/` folder:

```
python -m pipeline
```

or one step at a time:

| Command                                 | Paper output                                                    |
|-----------------------------------------|-----------------------------------------------------------------|
| `python -m pipeline.validate_qnm`       | Tables II, III (Kerr limit, Li 2023 comparison)                 |
| `python -m pipeline.validate_shadow`    | Shadow validation numbers, Fig. 1                               |
| `python -m pipeline.qnm_vs_spin`        | Fig. 2; Kerr spin that copies a hairy ringdown (Sec. III.B)     |
| `python -m pipeline.scan`               | Observables of 3750 black holes → `data/` (~5 min)              |
| `python -m pipeline.distinguishability` | Fig. 3 and the numbers of Sec. III, listed below                |
| `python -m pipeline.appendix`           | Appendix A tables                                               |

`distinguishability` prints, in order:

- Kerr ranges over the fitted spins `a_Kerr <= 0.94`, hairy ranges, `R_A` vs `R_s`
- Table IV with `d_min >= N`, and with chi-squared thresholds (ν = n_obs − 1;
  Combined: 2.83 / 3.76). Percentages are of the 3745 valid black holes.
- Joint-only exclusions and the Kerr-spin gap between the two channels
- Cells with clipped effective horizons, wrong-branch QNMs, and Table IV without them
- Best fits pinned at the `a_Kerr` search limit, and Table IV with the limit raised to 0.974
- Spearman correlations, per-alpha exclusion, the most Kerr-like case
- Largest `d_min` at `ell_0 = 1.70`, and the largest `ell_0` excluded per alpha
- Sgr A* shadow comparison

`scan` skips files that already exist in `data/`; pass `--force` to recompute.
Figures are written to `figures/`.

## Layout

```
physics/                    the physics, no plotting
  spacetime.py              metric functions m(r), Delta, F and the horizon
  shadow.py                 photon orbits, shadow boundary, R_s, delta_s
  qnm.py                    Leaver continued fraction + Newton -> QNM frequency
pipeline/                   one script per piece of the paper
  config.py                 every parameter: grid, precision scenarios, tolerances
  scan.py                   compute observables over the parameter grid
  distinguishability.py     profiled distance d_min to the Kerr family
  validate_qnm.py, validate_shadow.py, qnm_vs_spin.py, appendix.py
  style.py                  revtex figure sizing
data/                       cached scan results (.npz)
figures/                    paper figures
```

Units: `G = c = M = 1`.
