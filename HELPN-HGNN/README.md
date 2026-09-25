# Multimodal Brain Disease Detection via Hypergraph Neural Network with Hyper-events and Learnable Prompt Nodes

**HELPN-HGNN** 

A multimodal brain disorder classification framework using hyper-events, learnable prompt nodes, hypergraph neural networks, structural MRI features, and expert system collaboration.

## Branch Naming Convention

| Branch | Description |
|---|---|
| `main` | Stable release version |
| `dev` | Active development, may be unstable |
| `feat/<name>` | New feature under development (e.g., `feat/hyper-event-hga`) |
| `fix/<name>` | Bug fix branch (e.g., `fix/data-loader-bug`) |
| `exp/<name>` | Experimental ideas / ablation studies (e.g., `exp/expert-system-ablation`) |

The current code is pushed to the `main` branch.

## Project Structure

```
├── run.py                              # Entry point
├── models/                             # Neural network modules
│   ├── graph_classifier.py             # Main GraphClassifier model (HELPN-HGNN)
│   ├── hypergraph_capsule.py           # Hypergraph feature extraction module guided by hyper-events
│   ├── edge_attr_mlp.py               # Edge attribute MLP
│   ├── smri_mlp.py                    # sMRI structural phenotype feature MLP
│   ├── moe.py                         # Expert System Collaboration / gated expert selection
│   └── focal_loss.py                  # Focal loss function
├── data/                               # Data loading utilities
│   └── data_loader.py                 # Graph cache loader, subject features, ID extraction
├── utils/                              # Utility functions
│   ├── augmentation.py                # Graph augmentation (drop edge, feature noise)
│   ├── seed.py                        # Random seed setup
│   ├── laplacian_pe.py                # Laplacian positional encoding (optional, with fallback)
│   └── logging_config.py              # Logging configuration
├── train/                              # Training pipeline
│   └── trainer.py                     # Argument parsing, training/validation/test loop
└── data_preprocessing/                 # Data preprocessing scripts
    ├── build_spatiotemporal_graph.py   # Spatiotemporal graph construction from fMRI
    └── build_hypergraph.py             # Hyper-event hypergraph construction from fMRI
```

## Model Architecture

The `GraphClassifier` integrates multiple components:

1. **Spatiotemporal Brain Graph Construction** — builds brain-region nodes and functional edges from 4D fMRI data
2. **Hyper-event Construction** — detects synchronized BOLD peak activations across multiple brain regions and represents them as hyperedges
3. **Learnable Prompt Nodes (LPN) Injection** — injects learnable prompt nodes connected to all original nodes, compressing long-range dependencies to at most two hops
4. **Hypergraph Feature Extraction Network Guided by Hyper-events** — extracts hyperedge-level and sample-level hypergraph features from hyper-events
5. **Hyper-event Guided Attention (HGA) Layers (×3)** — multi-head attention guided by hyper-event features; performs local aggregation, prompt-mediated global aggregation, and local-global fusion
6. **Structural Phenotype Feature Extraction** — processes sMRI tissue proportion and mean intensity, together with gender and handedness information
7. **Feature Fusion** — concatenates graph, demographic, and sMRI embeddings
8. **Expert System Collaboration** — gated expert selection with dynamic weighting; implemented as MoE-style gating in code
9. **Focal Loss** — optional training loss for class imbalance with adjustable gamma and label smoothing

## Data Preprocessing

### Spatiotemporal Graph Construction

Builds correlation graphs from 4D fMRI data using SLIC superpixel segmentation:

```bash
python data_preprocessing/build_spatiotemporal_graph.py \
    --data_dir /path/to/data \
    --graph_cache_dir /path/to/graph_cache \
    --use_cached
```

### Hyper-event Hypergraph Construction

Builds hypergraphs from fMRI by detecting co-activation events across ROIs:

```bash
python data_preprocessing/build_hypergraph.py \
    -i /path/to/rest.nii.gz \
    -o /path/to/output_dir \
    --n_segments 500 \
    --pctl 90
```

## Training

```bash
python run.py \
    --data_dir /path/to/data \
    --excel /path/to/subjects.xlsx \
    --hidden_dim 64 \
    --gat_heads 1 \
    --dropout 0.176 \
    --lr 0.000252 \
    --batch_size 16 \
    --scheduler plateau \
    --weight_decay 0.00033 \
    --patience 5 \
    --label_smoothing 0.145 \
    --num_prompts 3 \
    --moe_thresh 0.626 \
    --warmup_epochs 7 \
    --moe_experts 8 \
    --class_weights 0.817,1.172 \
    --moe_k 2,7
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Root directory with class subfolders |
| `--excel` | required | Excel file with subject gender/handedness |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--hidden_dim` | 64 | GNN hidden dimension |
| `--gat_heads` | 4 | Number of attention heads in HGA layers |
| `--dropout` | 0.15 | Dropout probability |
| `--num_prompts` | — | Number of learnable prompt nodes (LPN) |
| `--moe_experts` | 4 | Number of experts in Expert System Collaboration |
| `--moe_k` | "2" | Selectable expert counts (comma-separated) |
| `--moe_thresh` | — | Gating threshold for expert selection |
| `--class_weights` | "1.0,1.0" | Class weights for imbalanced data |
| `--label_smoothing` | 0.1 | Label smoothing factor |
| `--warmup_epochs` | 10 | Learning rate warm-up epochs |
| `--scheduler` | plateau | LR scheduler (`plateau` or `cosine`) |
