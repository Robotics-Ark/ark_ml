#!/usr/bin/env python3
"""
Debug text embedding dimension issue
"""

import torch
from arkml.algos.vla.pi05.models import Pi05Policy

# Create a dummy Pi05Policy
policy = Pi05Policy(
    policy_type="pi0.5",
    model_path="lerobot/pi0.5",
    obs_dim=256,
    action_dim=8,
    image_dim=(3, 224, 224),
    pred_horizon=1,
    hidden_dim=512,  # This should be the embedding dim too
    vocab_size=32000,
    fast_vocab_size=1000
)

print("Text embedding shape:", policy.text_embedding.weight.shape)
print("Text projection weight shape:", policy.text_projection.weight.shape)

# Test with dummy tokens
dummy_tokens = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)  # [batch=1, seq_len=5]
embedded = policy.text_embedding(dummy_tokens)
print("Embedded shape:", embedded.shape)

# Mean pooling
mask = (dummy_tokens != 0).float().unsqueeze(-1)  # [batch, seq_len, 1]
masked_embs = embedded * mask
pooled_emb = masked_embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
print("Pooled shape:", pooled_emb.shape)

print(f"Expected pooled shape: [1, 512], Actual: {pooled_emb.shape}")