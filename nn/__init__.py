"""
ringkit.nn — engineer-facing model framework. Torch-shaped on the outside, ring-native inside.

Write ordinary models. The Z256 ring machinery — fold-late (energy/phase), unit-safety (no
zero-divisor collapse), exact linear SOLVE instead of blind descent, and content-routing
attention — is handled for you. You never touch mod-256, vacuums, or strides.

    import ringkit as rk
    layer = rk.nn.Linear(in_features=4, out_features=2)
    layer.fit(X, Y)                 # learns the exact ring map (solve, not descent) when it can
    pred = layer.predict(X_test)    # generalizes to unseen inputs

    out, who = rk.nn.attention(queries, keys, values)   # content-based routing, not lookup

Escape hatch: every layer exposes `.raw` (the underlying ring weights) for power users who DO
want the internals. Regular engineers can ignore it entirely.

    cache = rk.nn.KVCache(dim=4)                       # decode-time memory of the past
    cache.append(key, value)                           # 1 byte/coord, no scales, no calibration
    out = cache.attend(query, beta=16)                 # beta = temperature: 0 hot .. 255 argmax

Modules: layers (Layer/Linear/Dense/Sequential), transformer (RoPE/Attention/Transformer/
HopBlock/Stacked — stacked multi-block trained recall), kvcache (KVCache: ring-native decode cache,
Boltzmann-soft attention, cached == uncached bit-for-bit).
"""
from ringkit.ml import attention as _attn
from ringkit.ml import kvcache as _kv
from ringkit.nn.layers import Layer, Linear, Dense, Sequential
from ringkit.nn.transformer import (positional_encode, Attention, TransformerBlock,
                                    Transformer, HopBlock, Stacked)

# re-export content-based attention at the framework level (the real transformer primitive)
attention = _attn.attend
attention_scores = _attn.scores

# the decode-time KV cache (ours: Boltzmann-soft, circular-blended, data-free by construction)
KVCache = _kv.RingKVCache
attend_full = _kv.attend_full
