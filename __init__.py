"""ringkit — ring-topology ecosystem (Z256 / QH4). See docs/project-governance/ECOSYSTEM.md."""
from .core import native, calculus
from .linalg import solve, fit
from .stats import stats
from .physics import measure, qcm
from .rnp import tensor
from . import rnp                          # numpy replacement (rk.rnp)
from . import rmath                        # stdlib-math replacement (rk.rmath)
from . import rcollections                  # ring-native data structures
from .ml import autograd, optim
from . import nn                            # engineer-facing model framework (torch-shaped)
from . import data                          # engineer-facing data plumbing (encode/split/batch)
from . import rlearn                        # classical ML, ring-native (sklearn-shaped)
from . import emulation                      # EMULATION ENGINE (traditional models: checkpoint/onix/infer/ract) — SEPARATE from the pure ring nn
from .device import device, devices, default_device, Device   # .device() backend selection (cpu/cpu+simd/cuda/metal)
