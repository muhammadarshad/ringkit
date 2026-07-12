"""
ring_optim.py — ring-native optimizer (SRD Phase 5), on ring_native + ring_autograd.

Finding from the T5.2 spike: the update rule is NOT the bottleneck — once gradients are
ENERGY (non-wrapping, from ring_autograd), a plain sign step converges from 100% of starts.
The parameter is an ARC (ring position, wraps mod 256); the gradient is ENERGY (signed).
Step against the gradient's sign. Multiplier-free.
"""
from ringkit.core import native as rn


def sign(g):
    """sign of an ENERGY gradient: -1 / 0 / +1."""
    if g > 0:
        return 1
    if g < 0:
        return -1
    return 0


def sgd_step(param_val, grad, lr=1):
    """One sign-SGD step on an ARC parameter: move lr steps against the gradient, wrap mod 256."""
    s = sign(grad)
    return (param_val - rn.mul(lr, s)) & 0xFF


def coordinate_step(params, grads, loss_fn):
    """Coarse-to-fine, loss-gated coordinate step (SRD T5.5).

    Root cause of the SIN-training stall (derived by math): a +-1 step on a weight moves the
    pre-activation arg by its input coefficient x_i, so simultaneous steps overshoot the target
    angle (limit cycle). Fix: apply each parameter's gradient-sign step ONE AT A TIME and KEEP it
    only if it strictly reduces the loss. Overshoots are reverted; the fine (unit-coefficient /
    coprime) channel — the bias — lands the exact angle. This is the (state, r) codec as descent.

    params  : list of autograd Vars (ARC parameters)
    grads   : their ENERGY gradients
    loss_fn : zero-arg closure returning the current (non-wrapping) loss
    Returns number of accepted moves. Multiplier-free.
    """
    strict = 0
    for p, g in zip(params, grads):
        base = loss_fn()
        old = p.val
        s = sign(g)
        moved = False
        if s != 0:                                  # descent (against gradient): accept if NOT worse
            p.val = (old - s) & 0xFF                #   -> crosses SIN level-set plateaus
            l = loss_fn()
            if l <= base:
                moved = True
                if l < base:
                    strict += 1
            else:
                p.val = old
        if not moved:                               # opposite direction: strict improvement only
            d = s if s != 0 else 1
            p.val = (old + d) & 0xFF
            if loss_fn() < base:
                strict += 1
            else:
                p.val = old
    return strict
