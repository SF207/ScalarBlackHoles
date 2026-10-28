"""Reproduce every number and figure in the paper:  python -m pipeline"""

import runpy

STEPS = [
    "pipeline.validate_qnm",        # Tables II, III
    "pipeline.validate_shadow",     # shadow checks, Fig. 1
    "pipeline.qnm_vs_spin",         # Fig. 2
    "pipeline.scan",                # observables for 3750 black holes -> data/
    "pipeline.distinguishability",  # Table IV, Sec. III.B-E numbers, Fig. 3
    "pipeline.appendix",            # Appendix A tables
]

for step in STEPS:
    print(f"\n{'=' * 70}\n{step}\n{'=' * 70}")
    runpy.run_module(step, run_name="__main__")
