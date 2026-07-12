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

Modules: layers (Layer/Linear/Dense/Sequential), transformer (RoPE/Attention/Transformer).
"""
from ringkit.ml import attention as _attn
from ringkit.nn.layers import Layer, Linear, Dense, Sequential
from ringkit.nn.transformer import (positional_encode, Attention, TransformerBlock, Transformer)

# re-export content-based attention at the framework level (the real transformer primitive)
attention = _attn.attend
attention_scores = _attn.scores
