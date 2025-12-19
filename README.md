Transformer-Based English → French Machine Translation (From Scratch)

📌Project Overview

This project implements a Transformer model from scratch for English → French machine translation using PyTorch.

The goal is to understand and demonstrate the core architecture of Transformers (as proposed in “Attention Is All You Need”) by building every component manually, rather than relying on pre-built high-level libraries.

This project is developed as part of a Master’s-level academic project, focusing on:

Deep learning fundamentals
Sequence-to-sequence learning
Attention mechanisms
Practical NLP model training and evaluation

🚀 Key Features

Encoder–Decoder Transformer architecture
Multi-Head Self-Attention
Positional Encoding
Tokenization and vocabulary building from raw text
Teacher forcing during training
Early stopping to prevent overfitting
Translation inference using greedy decoding
Trained and evaluated on an English–French parallel corpus

🧠 Model Architecture

The model follows the standard Transformer architecture:

Encoder
Token embedding + positional encoding
Multi-head self-attention
Feed-forward neural network
Residual connections + layer normalization

Decoder
Masked self-attention (prevents future token access)
Encoder–decoder cross-attention
Feed-forward network
Output projection to vocabulary size
