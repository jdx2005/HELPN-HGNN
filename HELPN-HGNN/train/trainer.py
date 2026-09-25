import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
import logging

from models import GraphClassifier, FocalLoss
from data import load_cached_graphs, load_subject_features, extract_subject_id
from utils import augment_graph, set_seed, logger


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="fMRI Graph + Hypergraph + sMRI GNN Binary Classification")
    parser.add_argument('--data_dir', required=True,
                        help="Root directory containing 0/1 subfolders, each with subject directories")
    parser.add_argument('--excel', type=str, required=True,
                        help='Path to Excel file with subject gender and handedness features')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout probability')
    parser.add_argument('--graph_cache_dir', type=str, default='graph_cache',
                        help='Directory to save/load graph structure cache (.pt)')
    parser.add_argument('--use_cached', action='store_true',
                        help='If specified, train directly from cached graph structures without regenerating')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='GNN hidden layer dimension')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='Number of GAT heads')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Optimizer L2 regularization coefficient')
    parser.add_argument('--scheduler', choices=['plateau', 'cosine'],
                        default='plateau', help='Type of learning rate scheduler')
    parser.add_argument('--patience', type=int, default=5,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Cross-entropy label smoothing (0 to disable)')
    parser.add_argument('--num_prompts', type=int, default=3,
                        help='Number of prompt nodes injected per sample')
    parser.add_argument('--moe_experts', type=int, default=4,
                        help='Number of MoE experts')
    parser.add_argument('--moe_k', type=str, default="2",
                        help="List of selectable expert counts, comma-separated, e.g., '1,2,3'")
    parser.add_argument('--moe_thresh', type=float, default=0.8,
                        help="MoE expert activation threshold")
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='LR warm-up epoch count')
    parser.add_argument('--class_weights', type=str, default='1.0,1.0',
                        help='Comma-separated class weights for imbalanced data handling')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load cached graph data
    args.graph_cache_dir = os.path.join(args.data_dir, 'graphs_cache')
    data_list = load_cached_graphs(args.graph_cache_dir)
    logger.info(f"Loaded graph samples from cache: {len(data_list)}")
    if len(data_list) == 0:
        logger.info("Cache is empty, exiting.")
        return

    # Attach subject features to graph data
    subj_features = load_subject_features(args.excel)
    for data in data_list:
        subj_id = extract_subject_id(getattr(data, "name", ""))
        if subj_id and subj_id in subj_features:
            data.subject_feat = torch.tensor(
                subj_features[subj_id], dtype=torch.float
            ).unsqueeze(0)
        else:
            logger.warning(f"Cannot find features for sample {getattr(data, 'name', 'unknown')}")
    logger.info("Attached subject features to graph data")

    # Split dataset into training, validation, and test sets
    total = len(data_list)
    test_n = max(1, int(total * args.test_size))
    val_n = max(1, int(total * args.val_size))
    train_n = total - val_n - test_n
    lengths = [train_n, val_n, test_n]
    if sum(lengths) < total:
        lengths[0] += total - sum(lengths)
    train_ds, val_ds, test_ds = random_split(
        data_list, lengths, generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    def count_labels(dataset):
        pos_count = sum(1 for data in dataset if data.y.item() == 1)
        neg_count = len(dataset) - pos_count
        return pos_count, neg_count

    train_pos, train_neg = count_labels(train_ds)
    val_pos, val_neg = count_labels(val_ds)
    test_pos, test_neg = count_labels(test_ds)

    logger.info(f"Dataset sizes: Train={len(train_ds)}, Validation={len(val_ds)}, Test={len(test_ds)}")
    logger.info(f"Train set: Positive={train_pos}, Negative={train_neg}")
    logger.info(f"Validation set: Positive={val_pos}, Negative={val_neg}")
    logger.info(f"Test set: Positive={test_pos}, Negative={test_neg}")

    # Initialize model, optimizer, and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    args_moe_k_list = list(map(int, args.moe_k.split(',')))
    model = GraphClassifier(
        in_channels=1,
        hidden_dim=args.hidden_dim,
        heads=args.gat_heads,
        dropout=args.dropout,
        num_prompts=args.num_prompts,
        moe_experts=args.moe_experts,
        moe_k=args_moe_k_list,
        moe_thresh=args.moe_thresh
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.warmup_epochs, T_mult=1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.patience)

    # Define loss function
    class_weights = torch.tensor(list(map(float, args.class_weights.split(','))),
                                 dtype=torch.float,
                                 device=device)
    inv_weights = 1.0 / class_weights
    criterion = FocalLoss(gamma=1.0,
                          weight=inv_weights,
                          smoothing=args.label_smoothing,
                          reduction='mean')

    # Training loop
    initial_lr = args.lr
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            warmup_lr = initial_lr * (epoch / args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        model.train()
        loss_sum, preds, trues, total_graphs = 0, [], [], 0
        for batch in train_loader:
            batch = augment_graph(batch, drop_prob=-1, noise_std=-1, seed=42)
            batch = batch.to(device)
            out = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, batch.subject_feat.to(device), batch.smri_feat.to(device),
                hypergraph=batch.hypergraph
            )
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            preds += out.argmax(1).detach().cpu().tolist()
            trues += batch.y.cpu().tolist()
        avg_loss = loss_sum / total_graphs
        train_acc = accuracy_score(trues, preds)

        # Validation loop
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.batch, batch.subject_feat.to(device), batch.smri_feat.to(device),
                    hypergraph=batch.hypergraph
                )
                val_preds += out.argmax(1).cpu().tolist()
                val_trues += batch.y.cpu().tolist()
        val_acc = accuracy_score(val_trues, val_preds)

        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'origin_ADHD_622_best_graph_model_bf_cw_lf.pth')
            logger.info(f"Saved best model at epoch {epoch} with Val Acc: {val_acc:.4f}")

    # Test the best model
    model.load_state_dict(torch.load('origin_ADHD_622_best_graph_model_bf_cw_lf.pth'))
    test_preds, test_trues, test_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, batch.subject_feat.to(device), batch.smri_feat.to(device),
                hypergraph=batch.hypergraph
            )
            probs = F.softmax(out, dim=1)[:, 1]
            test_probs += probs.cpu().tolist()
            test_preds += out.argmax(1).cpu().tolist()
            test_trues += batch.y.cpu().tolist()

    acc = accuracy_score(test_trues, test_preds)
    cm = confusion_matrix(test_trues, test_preds)
    cr = classification_report(
        test_trues,
        test_preds,
        target_names=['Normal', 'Disorder'],
        digits=4
    )
    logger.info("=== Test Set Classification Performance ===")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("Confusion Matrix:\n" +
                np.array2string(
                    cm.astype(float),
                    formatter={'float_kind': lambda x: f"{x:.4f}"}
                ))
    logger.info("Classification Report:\n" + cr)
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    f1 = f1_score(test_trues, test_preds)
    auc = roc_auc_score(test_trues, test_probs)
    logger.info(f"ACC (%): {acc * 100:.2f}% | SEN (%): {sen * 100:.2f}% | SPE (%): {spe * 100:.2f}%")
    logger.info(f"F1-score: {f1:.4f} | AUC: {auc:.4f}")