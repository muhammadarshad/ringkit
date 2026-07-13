"""
ringkit.ml.kvpolar — REMOVED / DEPRECATED SHIM.

"Polar" was the Euclidean import: Cartesian→polar via `atan2` plus the L2 magnitude
`sqrt(x^2 + y^2)`. That is an MPRC ANTI-PATTERN (Prime Directive): `a^2 + b^2 = c^2` is foreign
standard math, it is LOSSY (distinct pairs collapse to one magnitude), and it forced a SIN/COS
reconstruction on decode. It has been removed.

The ring-native KV element is ADI (accumulation, differential) — exact, reversible, differentiable,
multiplier-free, no Euclidean. Use `ringkit.ml.kvadi`. This shim re-exports it for back-compat only.
"""
from ringkit.ml.kvadi import (   # noqa: F401
    encode_pair, decode_pair, encode, decode, accumulation, differential,
)
