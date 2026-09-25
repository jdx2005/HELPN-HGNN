# HELPN-HGNN

**HELPN-HGNN**: **H**ypergraph-**E**nhanced **L**ocal-**P**rompt **N**etwork with **H**eterogeneous **G**raph **N**eural **N**etworks

A framework for brain disorder classification using spatiotemporal graph neural networks, hypergraph capsule networks, and structural MRI features.

## Branch Naming Convention

| Branch | Description |
|---|---|
| `main` | Stable release version |
| `dev` | Active development, may be unstable |
| `feat/<name>` | New feature under development (e.g., `feat/hypergraph-capsule`) |
| `fix/<name>` | Bug fix branch (e.g., `fix/data-loader-bug`) |
| `exp/<name>` | Experimental ideas / ablation studies (e.g., `exp/moe-ablation`) |

The current code is pushed to the `main` branch.

## Project Structure

```
├── run.py                              # Entry point
├── models/                             # Neural network modules
│   ├── graph_classifier.py             # Main GraphClassifier model
│   ├── hypergraph_capsule.py           # HypergraphCapsule module
│   ├── edge_attr_mlp.py               # Edge attribute MLP
│   ├── smri_mlp.py                    # sMRI feature MLP
│   ├── moe.py                         # Mixture of Experts
│   └── focal_loss.py                  # Focal loss function
├── data/                               # Data loading utilities
│   └── data_loader.py                 # Graph cache loader, subject features, ID extraction
├── utils/                              # Utility functions
│   ├── augmentation.py                # Graph augmentation (drop edge, feature noise)
│   ├── seed.py                        # Random seed setup
│   ├── laplacian_pe.py                # Laplacian positional encoding (with fallback)
│   └── logging_config.py              # Logging configuration
├── train/                              # Training pipeline
│   └── trainer.py                     # Argument parsing, training/validation/test loop
└── data_preprocessing/                 # Data preprocessing scripts
    ├── build_spatiotemporal_graph.py   # Spatiotemporal graph construction from fMRI
    └── build_hypergraph.py             # Hypergraph construction from fMRI
```

## Model Architecture

The `GraphClassifier` integrates multiple components:

1. **Laplacian Positional Encoding** — encodes graph structural information
2. **Prompt Node Injection** — injects learnable prompt nodes connected to all original nodes
3. **Hypergraph Capsule** — extracts hypergraph features from co-activation events
4. **GATConv Layers (×3)** — graph attention convolution with Jumping Knowledge aggregation
5. **Transformer Encoder** — global context modeling via a learnable global token
6. **Subject & sMRI Feature MLPs** — processes demographic and structural MRI features
7. **Feature Fusion** — concatenates graph, subject, and sMRI embeddings
8. **Mixture of Experts (MoE)** — adaptive expert selection with dynamic gating
9. **Focal Loss** — handles class imbalance with adjustable gamma and label smoothing

## Data Preprocessing

### Spatiotemporal Graph Construction

Builds correlation graphs from 4D fMRI data using SLIC superpixel segmentation:

```bash
python data_preprocessing/build_spatiotemporal_graph.py \
    --data_dir /path/to/data \
    --graph_cache_dir /path/to/graph_cache \
    --use_cached
```

### Hypergraph Construction

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
| `--gat_heads` | 4 | Number of GAT attention heads |
| `--dropout` | 0.15 | Dropout probability |
| `--moe_experts` | 4 | Number of MoE experts |
| `--moe_k` | "2" | Selectable expert counts (comma-separated) |
| `--class_weights` | "1.0,1.0" | Class weights for imbalanced data |
| `--label_smoothing` | 0.1 | Label smoothing factor |
| `--warmup_epochs` | 10 | Learning rate warm-up epochs |
| `--scheduler` | plateau | LR scheduler (`plateau` or `cosine`) |

Licence
The dataset and code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 international License.
