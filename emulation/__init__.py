"""
ringkit.emulation — the EMULATION ENGINE for traditional models we do not control (Gemma, external
checkpoints). Ingest their weights and run their forward on the ring — like TurboQuant/PolarQuant,
but float-free (FPU replaced by ringkit QCM kernels). See docs/project-governance/STANCE_emulation_vs_os.md.

This is SEPARATE from the pure ring-native nn (ringkit.nn / core / physics / ml), which is the
operating system for OUR MPRC architectures. Emulation code lives here so it never disturbs the
native stack.

  checkpoint — torch .pth -> ring (integer mantissa-shift)
  onix       — Gemma .onix -> ring (already integer; QCM qsm dot + shift/divide dequant)
  infer      — ring fixed-point linear / attention / softmax (shift-add, no float)
  ract       — ring fixed-point activations (exp/sigmoid/gelu/rmsnorm/layernorm/tanh/softcap), float-free
  gemma      — full Gemma2-2B autoregressive forward on ring primitives (float-free)
  gemma_weights — file-backed Gemma2-2B provider (onix mmap + embed.bin/norms.bin, streaming)
  tokenizer  — Gemma BPE encode/decode (string processing only)
"""
from . import checkpoint
from . import onix
from . import infer
from . import ract
from . import gemma
from . import gemma_weights
from . import tokenizer
