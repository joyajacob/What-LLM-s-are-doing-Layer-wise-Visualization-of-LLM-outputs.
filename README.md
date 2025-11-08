# What-LLM-s-are-doing-Layer-wise-Visualization-of-LLM-outputs.
What LLM’s are doing:  Layer-wise Visualization of LLM outputs.
This project visualizes how token representations evolve across the 32 decoder layers of LLaMA-2-7B. Given a prompt (e.g., “How are you?”), the code extracts hidden states from every layer, normalizes them, and projects the 4096-d vectors to 2-D using PCA, t-SNE, and Laplacian Eigenmaps (LE). It then plots per-token trajectories layer by layer, revealing patterns like punctuation separation, function-vs-content word grouping, and semantic convergence.

Features

Hidden state extraction for all 33 steps (embeddings + 32 layers)

2-D projections via PCA (global, deterministic), t-SNE (local clusters), LE (smooth, manifold-aware)

Trajectories for input tokens and optional generated continuation

Matplotlib/Plotly visuals suitable for teaching & lightweight analysis

How it works

Tokenize input → add BOS/EOS

Run forward pass with output_hidden_states=True

Stack to (layers, tokens, 4096) → standardize

Reduce to 2-D (PCA / t-SNE / LE)

Plot token paths across layers

Outputs

Layer-wise trajectory plots for each method

(Optional) trajectories for cumulative sequences including generated response

Notes: PCA is stable/reproducible; t-SNE highlights fine-grained neighborhoods (seed-sensitive); LE provides smooth, readable paths.
