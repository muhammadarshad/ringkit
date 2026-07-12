"""ringkit — ring-topology ecosystem (Z256 / QH4). See docs/ECOSYSTEM.md."""
from .core import native, calculus
from .linalg import solve, fit
from .stats import stats
from .physics import measure, qcm
from .array import tensor
from .array import numpy as rnp
from .ml import autograd, optim
from . import nn                            # engineer-facing model framework (torch-shaped)
from . import data                          # engineer-facing data plumbing (encode/split/batch)
