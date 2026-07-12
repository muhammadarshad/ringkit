"""ringkit — ring-topology ecosystem (Z256 / QH4). See docs/project-governance/ECOSYSTEM.md."""
from .core import native, calculus
from .linalg import solve, fit
from .stats import stats
from .physics import measure, qcm
from .rnp import tensor
from . import rnp                          # numpy replacement (rk.rnp)
from . import rmath                        # stdlib-math replacement (rk.rmath)
from . import collections                   # ring-native data structures
from .ml import autograd, optim
from . import nn                            # engineer-facing model framework (torch-shaped)
from . import data                          # engineer-facing data plumbing (encode/split/batch)
