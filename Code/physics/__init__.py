"""Physics of rotating black holes with primary scalar hair: metric, shadow, QNMs."""

from .spacetime import HairyKerr
from .shadow import shadow_boundary, shadow_observables, photon_orbit_range
from .qnm import solve_qnm, effective_horizons, angular_separation_constant, continued_fraction
