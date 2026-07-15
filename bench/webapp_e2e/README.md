# webapp e2e anchoring recipe (C6: torch allowed here, as the labeled ground-truth engine)

The three-step anchor chain that gated `tests/test_quanta.py`'s APP-E2E section (2026-07-15).
Never gate a ring port against a hand-written oracle alone — the oracle inherits the port's
misreadings (this chain caught a wrong 4D-rope rotate-half that both sides shared, a missing
vacuum branch, and an arc wrap at phase 1.0).

1. `app_e2e_anchor.py` — runs the REAL torch app models (vlm-transformers repo modules +
   deployed checkpoints) on a deterministic real-regime frame + a real photo; saves cubes and
   logits to `app_e2e_ref.npz`. Also proves the Mamba2 HF export == the gated .pth weights.
   Run: `~/.venvs/ringkit-bench/bin/python app_e2e_anchor.py`
2. `app_e2e_oracle.py` — the numpy oracle of the app forward; must match the anchor logits to
   fp32 precision (~2e-5) BEFORE the ring is compared to it. This oracle's math is what
   `tests/test_quanta.py` inlines.
3. `app_e2e_ring.py` — the ring `quanta` forwards at full app scale (N=1808) vs the anchor;
   cos 1.000000 + argmax on all four (model x image) combinations, incl. the real shelf photo.
   Run with plain `python3` (ringkit importable from the repo parent).

Inputs it expects locally: the rotor heads checkpoint in the HF cache
(`models--marshadbits--qcm-rp2k-cloud-matched-results/.../heads_rdt_L2_k2384_best.pth`), the
Soliton .pth under `~/Projects/vlm-transformers/huggingface/exported/`, and (for the real-photo
leg) `~/Pictures/ShelfImages/Picture 1.jpg` — swap in any real photo; the regimes that matter
are padding (vacuum tokens) and saturated 255s.
