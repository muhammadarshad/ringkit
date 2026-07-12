"""
ringkit.ml.attention — genuine ring-native attention (NOT the retrieval-lookup).

This is the defining transformer mechanism, done on Z256: each query position scores every key by
CONTENT similarity and reads the matching value. Unlike a lexical/kNN index, the (key,value)
bindings are per-example and may be entirely novel at test time — solving the task REQUIRES
content-based routing, which is what makes it attention and not a lookup table.

    score(i, j) = - sum_d ring_distance(Q[i][d], K[j][d])      (ENERGY: signed, NOT folded mod 256)
    hard:  out[i] = V[ argmax_j score(i, j) ]                  (route to best-matching key)
    soft:  out[i] = ring circular-mean of V weighted by a nonneg similarity kernel of the scores

Q,K,V are lists of ring-int vectors (rows). Multiplier-free; no numpy, no float, no softmax-in-float.
"""
from ringkit.core import native as rn
from ringkit.stats import stats as rs


def ring_distance(a, b):
    d = (a - b) & 0xFF
    e = 256 - d
    return d if d < e else e


def scores(Q, K):
    """Content-similarity score matrix. score(i,j) = -sum_d ring_distance(Q_i[d], K_j[d]).
    Higher (closer to 0) = better match. Signed ENERGY values (no mod fold)."""
    S = []
    for qi in Q:
        row = []
        for kj in K:
            s = 0
            for d in range(len(qi)):
                s -= ring_distance(qi[d], kj[d])
            row.append(s)
        S.append(row)
    return S


def attend(Q, K, V, hard=True):
    """Ring attention. Returns (out, idx) where out[i] is the attended value vector and idx[i] the
    key position attended to (hard). Soft mode blends V by a triangular similarity kernel using the
    exact ring circular mean, so blending respects the ring wrap."""
    if not (len(K) == len(V)):
        raise ValueError(f"attend: keys/values length mismatch {len(K)} vs {len(V)}")
    S = scores(Q, K)
    out, idx = [], []
    for i, srow in enumerate(S):
        best_j = 0
        best = srow[0]
        for j in range(1, len(srow)):
            if srow[j] > best:
                best, best_j = srow[j], j
        idx.append(best_j)
        if hard:
            out.append(list(V[best_j]))
        else:
            # triangular similarity kernel: w_j >= 0, peaks at the best-matching key. Blend V by the
            # EXACT ring circular_mean, giving each value multiplicity w_j (integer weighting).
            w = [max(0, srow[j] - best + 128) for j in range(len(srow))]   # 128 = ring half-window
            dv = len(V[0])
            vec = []
            for d in range(dv):
                pool = []
                for j in range(len(V)):
                    for _ in range(w[j]):
                        pool.append(V[j][d])
                vec.append(rs.circular_mean(pool) if pool else V[best_j][d])
            out.append(vec)
    return out, idx
