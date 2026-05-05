# SAE vs. Cobweb

Two experiments benchmarking Cobweb-based sparse representations against standard Sparse Autoencoders (SAEs).

---

## Experiment 1 — MNIST (`mnist_example.py`)

### Setup
- **Data**: MNIST, 10 000 train / 2 000 test, flattened to 784-d float32.
- **Representation dim**: 128 (all methods).

### Methods

| Method | Description |
|---|---|
| PCA (128d) | Standard PCA baseline. |
| AE (128d) | 3-layer MLP autoencoder, MSE loss, 20 epochs. |
| L1-SAE (128d) | ReLU encoder + unit-norm decoder, L1 penalty λ=3e-4, 20 epochs. |
| TopK-SAE (128d, k=16) | Hard TopK sparsification + AuxK dead-neuron revival (auxk=16, w=1/32), 20 epochs. |
| Cobweb-BFS (128d) | 128 BFS-order Cobweb nodes; log P(node\|x) features, StandardScaler. |
| Cobweb-Depth | Nodes at deepest depth < 128; same encoding. |
| Cobweb-TopK | First depth ≥ 128 nodes; per-instance top-16 sparsification. |

### Evaluation
- **Linear probe** (LinearSVC) overall and per-class.
- **KNN curves** k ∈ {1,3,5,10,20,50}.
- **UMAP** and **t-SNE** scatter plots (7 panels each).
- **Reconstruction gallery** — Original / PCA / AE / L1-SAE / TopK-SAE, first 10 samples.
- **Summary CSV** (`mnist_output/summary.csv`): `method, lin_probe_pct, knn5_pct, avg_l0, dead_pct`.

### Outputs
```
tests/moc/mnist_output/
  summary.csv
  scatter_umap.png
  scatter_tsne.png
  linear_probe_per_class.png
  knn_vs_k.png
  reconstructions.png
  cobweb_tree_labels.png
  arrays/          ← gitignored: all .npy and .pt files
```

---

## Experiment 2 — GPT-2 Last-Layer Activations (3 scripts)

Evaluates SAEs on GPT-2 (small, 117 M) residual-stream activations from WikiText-103, probed with POS tags.

---

### `collect_gpt_acts.py`

Runs GPT-2 over WikiText-103, extracts the **last Transformer block** hidden state for every token position, and aligns each token with its spaCy Universal POS tag.

- **Model**: `gpt2` (d_model = 768).
- **Corpus**: WikiText-103-raw-v1 (Hugging Face `datasets`).
- **Collected**: up to 200 k train / 40 k test token-position vectors.
- **Label task**: Universal POS tags (17 classes) via spaCy `en_core_web_sm`.

**Outputs** (`gpt_acts_output/acts/`):
```
acts_train.npy   (N_train, 768)
acts_test.npy    (N_test,  768)
pos_train.npy    (N_train,) int16 POS indices
pos_test.npy     (N_test,)  int16
token_train/test.npy        GPT-2 token ids
pos_vocab.json              {tag: index}
```

---

### `train_llm_saes.py`

Trains four SAE variants on the whitened activations (StandardScaler per-feature):

| Method | Description |
|---|---|
| **L1-SAE** | ReLU encoder → unit-norm decoder, MSE + λ·‖h‖₁, λ=3e-4. |
| **TopK-SAE** | Hard TopK (k=32) + AuxK dead-neuron revival, unit-norm decoder. |
| **JumpReLU-SAE** | Learnable per-neuron threshold θᵢ (stored as log θᵢ); STE gradient via bandwidth-limited rectangle; L0 penalty drives avg active features toward k=32. Faithful to Rajamanoharan et al. 2024. |
| **Cobweb-TopK** | `CobwebContinuousTree` built on first 50 k training vectors; 3072 BFS nodes extracted; log P(node\|x) encoded, StandardScaler, per-instance top-32 sparsification. |

- **Latent dim**: 3072 (4× expansion, standard in the SAE literature).
- **Epochs**: 10.  **LR**: 2e-4.  **Batch**: 512.

**Outputs** (`gpt_acts_output/models/`):
```
l1sae.pt / topksae.pt / jumprelu.pt
Z_{method}_train/test.npy
input_scaler_mean/std.npy
meta.json
```

---

### `compare_llm_saes.py`

Loads saved latents and produces:

| Output | Description |
|---|---|
| `summary.csv` | `method, lin_probe_pct, knn5_pct, avg_l0, dead_pct` |
| `knn_vs_k.png` | KNN accuracy curves k ∈ {1,3,5,10,20,50} |
| `linear_probe_per_pos.png` | Per-POS-tag bar chart (Logistic Regression) |
| `dead_neurons.png` | % dead neurons per method |
| `act_histograms.png` | Non-zero activation magnitude distributions (log scale) |
| `scatter_umap.png` | UMAP 2D, coloured by POS tag (N=5 000 subsample) |
| `scatter_tsne.png` | t-SNE 2D, same (N=3 000) |
| `l0_vs_linear.png` | Scatter: avg L0 vs. linear-probe accuracy |

All plots saved to `gpt_acts_output/plots/`.

---

## Running order

```bash
# 1. Collect activations (~10–20 min depending on hardware)
python tests/moc/collect_gpt_acts.py

# 2. Train SAEs (~30–60 min on CPU, ~5 min on GPU)
python tests/moc/train_llm_saes.py

# 3. Compare and visualise
python tests/moc/compare_llm_saes.py
```

## Dependencies (beyond base env)
```
transformers   # GPT-2
datasets       # WikiText-103
spacy          # POS tagging
# python -m spacy download en_core_web_sm
umap-learn
```
 