This code demonstrates:
Core Components:

Tokenizer: Converts text to numbers
Multi-head attention: The key mechanism in transformers
Feed-forward networks: Process attention outputs
Positional embeddings: Track word positions
Training loop: Optimize the model

What this teaches you:

How transformers actually work under the hood
Self-attention mechanisms
Causal masking (prevents "cheating" by looking ahead)
Text generation with temperature sampling

Limitations:

Uses tiny architecture (would need 100x-1000x more parameters for real use)
Character-level tokenization (real models use subword tokenization like BPE)
Minimal training data
No advanced features (dropout scheduling, gradient clipping, etc.)
