"""QCM physics constants — SINGLE SOURCE OF TRUTH, native port of qcm-med-vision/qcm_constants.py.

Every value derives from N=3 through the QCM chain; they are NOT hyperparameters and MUST NOT be
changed (docs/QCM_CRITICAL_WARNINGS.md). `_verify_chain()` runs at import and asserts the whole chain.

    N = sqrt(D) - 1 = 3          master quantum number
    D = 16 = (N+1)^2             Euclidean time slices
    D_MODEL = 128 = D*(N^2-1)    D * dim(su(3)) = 16*8
    H = 113 = 7*D + 1            prime spatial extent; gcd(H,7)=1
    L = 1808 = D*H               spacetime volume = sequence length
    Q = 64 = 256/(N+1)           vacuum spacing
    vacuums = {0,64,128,192}     phase singularities (4th roots of unity)

All values are integers (the ring is integer). MC_BETA is the only rational — it parameterises the
lattice Boltzmann LUT OFFLINE (the I/O bridge), never the compute path; carried as (num, den).
"""

# -- Master quantum number --
N = 3

# -- Spacetime structure --
D = 16                       # (N+1)^2 -- Euclidean time slices
H = 113                      # 7*D+1 -- prime spatial extent; gcd(H,7)=1
L = D * H                    # 1808 -- spacetime volume = sequence length

# -- Representation theory --
D_MODEL = D * (N**2 - 1)     # 128 -- D * dim(su(3)) = 16*8
ACTIVE_SLICES = (N + 1)**2 - 1   # 15 -- dim(su(4)) = SU(4) generators

# -- Z256 ring structure --
TAU = 256                    # ring order = vocab size
Q = TAU // (N + 1)           # 64 -- vacuum spacing
VACUUMS = (0, Q, 2 * Q, 3 * Q)   # (0,64,128,192) -- phase singularities
ACTIVE_BINS = TAU - len(VACUUMS)  # 252 -- non-vacuum elements

# -- Ring strides (co-prime to 256, vacuum-avoiding) --
STRIDE_X = 3
STRIDE_Y = 5
STRIDE_U = 7                 # master stride -- U Observer; triple-constraint solution
STRIDE_W = 9
STRIDES = (STRIDE_X, STRIDE_Y, STRIDE_U, STRIDE_W)

# Anti-strides (modular inverses mod 256)
ANTI_X = 171                 # 3*171 == 1 (mod 256)
ANTI_Y = 205                 # 5*205 == 1 (mod 256)
ANTI_U = 183                 # 7*183 == 1 (mod 256)
ANTI_W = 57                  # 9*57  == 1 (mod 256)

# -- MPRC geometry --
MPRC_BINS = ACTIVE_BINS // STRIDE_U      # 36 = 252/7 = (2N)^2
MPRC_QUADRANTS = N + 1                    # 4
MPRC_BINS_PER_QUADRANT = MPRC_BINS // MPRC_QUADRANTS  # 9

# -- Lattice Monte Carlo (MC_BETA is I/O-only: builds the Boltzmann LUT offline, never on the path) --
MC_STEPS = 2
MC_BETA_NUM, MC_BETA_DEN = 5, 2          # 2.5 as a ring rational (num/den), not a float

# -- Training constants (fixed, not tunable) --
# GATE_BIAS = -2.0 as a ring rational: CRITICAL FIX 1 (RDT keep-gate; sigmoid(-2)=0.119 -> 60.1%).
GATE_BIAS_NUM, GATE_BIAS_DEN = -2, 1
# TEMPERATURE = 0.07 (InfoNCE) as a ring rational (7/100), I/O-only.
TEMPERATURE_NUM, TEMPERATURE_DEN = 7, 100

# -- Text tokenizer --
VOCAB_SIZE = TAU             # 256
PAD_TOKEN = 0                # vacuum ground state
BOS_TOKEN = 1
EOS_TOKEN = 2
MAX_TEXT_LEN = Q            # 64 -- spectral period of the quantum walk
TEXT_ENCODER_DEPTH = N + 1   # 4 -- fundamental rep of SU(4)

# Trig amplitude of the ring SIN/COS tables (core.native): peak |SIN(64)| = SCALE.
SCALE = 21                   # = 3*7 (matches transformer/ring_trig and core.native)


def _verify_chain():
    """Verify the full derivation chain is internally consistent (runs at import)."""
    assert N == 3
    assert D == (N + 1)**2
    assert D_MODEL == D * (N**2 - 1) == 128
    assert H == 7 * D + 1
    assert H % 7 != 0, "gcd(H,7) must be 1"
    assert L == D * H == 1808
    assert Q == TAU // (N + 1) == 64
    assert VACUUMS == tuple(k * Q for k in range(N + 1))
    assert ACTIVE_BINS == 252
    assert MPRC_BINS == ACTIVE_BINS // STRIDE_U == 36 == (2 * N)**2
    assert MPRC_QUADRANTS == 4
    assert MPRC_BINS_PER_QUADRANT == 9
    assert ACTIVE_SLICES == (N + 1)**2 - 1 == 15
    assert TEXT_ENCODER_DEPTH == 4
    for stride, anti in ((STRIDE_X, ANTI_X), (STRIDE_Y, ANTI_Y),
                         (STRIDE_U, ANTI_U), (STRIDE_W, ANTI_W)):
        assert (stride * anti) % TAU == 1, f"{stride}*{anti} != 1 (mod 256)"


_verify_chain()
