#!/usr/bin/env python3
"""
Check the tokenizer vocab size
"""

from arkml.algos.vla.pi05.models import DummyTokenizer

tokenizer = DummyTokenizer()
print("Vocab size:", len(tokenizer.vocab))
print("Max vocab value:", max(tokenizer.vocab.values()))
print("Vocab:", tokenizer.vocab)

# Test encoding a string
text = "pickup the block"
encoded = tokenizer.encode(text)
print("Encoded text:", encoded)
print("Max token value:", max(encoded) if encoded else 0)