"""
Hyperparameter search for the MLP baseline.

Input: 200-dim one-hot encoding (8 positions × 25 tokens).
Output: binary classification (ok / bad).
Saves best hyperparams and training curves to results/mlp_best.json.
"""

import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from grammar import ALL_TOKENS, TOKEN_TO_IDX, VOCAB_SIZE, SEQ_LEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data ──────────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    seqs = [[TOKEN_TO_IDX[row[f"t{i}"]] for i in range(SEQ_LEN)] for row in rows]
    labels = [1 if row["label"] == "ok" else 0 for row in rows]
    return seqs, labels


def to_onehot_tensor(seqs, labels):
    n = len(seqs)
    x = torch.zeros(n, SEQ_LEN * VOCAB_SIZE)
    for i, seq in enumerate(seqs):
        for pos, tok in enumerate(seq):
            x[i, pos * VOCAB_SIZE + tok] = 1.0
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def load_data(base):
    print("Loading data...", flush=True)
    tr_seqs, tr_lab = load_csv(os.path.join(base, "train.csv"))
    va_seqs, va_lab = load_csv(os.path.join(base, "validate.csv"))
    te_seqs, te_lab = load_csv(os.path.join(base, "test.csv"))
    tr_x, tr_y = to_onehot_tensor(tr_seqs, tr_lab)
    va_x, va_y = to_onehot_tensor(va_seqs, va_lab)
    te_x, te_y = to_onehot_tensor(te_seqs, te_lab)
    print("Done loading.", flush=True)
    return (tr_x.to(DEVICE), tr_y.to(DEVICE),
            va_x.to(DEVICE), va_y.to(DEVICE),
            te_x.to(DEVICE), te_y.to(DEVICE))

# ── model ─────────────────────────────────────────────────────────────────────

def build_mlp(hidden_sizes, dropout):
    layers = []
    in_dim = SEQ_LEN * VOCAB_SIZE
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── training ──────────────────────────────────────────────────────────────────

def train_model(model, tr_x, tr_y, va_x, va_y,
                lr, batch_size, max_epochs=30, patience=4):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = TensorDataset(tr_x, tr_y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses, val_accs = [], [], []
    best_val = float("inf")
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        train_losses.append(ep_loss / len(tr_x))

        model.eval()
        with torch.no_grad():
            val_out = model(va_x)
            vl = loss_fn(val_out, va_y).item()
            acc = (val_out.argmax(1) == va_y).float().mean().item()
        val_losses.append(vl)
        val_accs.append(acc)

        if vl < best_val - 1e-4:
            best_val = vl
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return train_losses, val_losses, val_accs


def test_accuracy(model, te_x, te_y):
    model.eval()
    with torch.no_grad():
        out = model(te_x)
        return (out.argmax(1) == te_y).float().mean().item()

# ── search ────────────────────────────────────────────────────────────────────

SEARCH_SPACE = [
    # (n_layers, width, dropout, lr, batch_size)
    (1, 128,  0.0, 1e-3, 256),
    (1, 256,  0.0, 1e-3, 256),
    (1, 512,  0.0, 1e-3, 256),
    (2, 128,  0.0, 1e-3, 256),
    (2, 256,  0.0, 1e-3, 256),
    (2, 512,  0.0, 1e-3, 256),
    (3, 128,  0.0, 1e-3, 256),
    (3, 256,  0.0, 1e-3, 256),
    (1, 256,  0.1, 1e-3, 256),
    (2, 256,  0.1, 1e-3, 256),
    (2, 256,  0.0, 3e-4, 256),
    (2, 256,  0.0, 3e-4, 512),
    (2, 512,  0.1, 3e-4, 256),
    (3, 256,  0.1, 3e-4, 256),
    (3, 512,  0.1, 3e-4, 256),
]


def run_search(tr_x, tr_y, va_x, va_y, te_x, te_y):
    results = []
    for i, (n_layers, width, dropout, lr, bs) in enumerate(SEARCH_SPACE):
        hidden = [width] * n_layers
        model = build_mlp(hidden, dropout)
        n_params = count_params(model)
        t0 = time.time()
        tr_l, va_l, va_a = train_model(model, tr_x, tr_y, va_x, va_y,
                                        lr=lr, batch_size=bs)
        test_acc = test_accuracy(model, te_x, te_y)
        elapsed = time.time() - t0
        rec = {
            "n_layers": n_layers, "width": width, "dropout": dropout,
            "lr": lr, "batch_size": bs, "n_params": n_params,
            "epochs_run": len(tr_l),
            "final_val_acc": va_a[-1],
            "best_val_acc":  max(va_a),
            "test_acc": test_acc,
            "train_losses": tr_l,
            "val_losses": va_l,
            "val_accs": va_a,
            "elapsed_s": round(elapsed, 1),
        }
        results.append(rec)
        print(f"[{i+1}/{len(SEARCH_SPACE)}] layers={n_layers} w={width} "
              f"drop={dropout} lr={lr:.0e} bs={bs} | "
              f"val_acc={rec['best_val_acc']:.4f} test_acc={test_acc:.4f} "
              f"params={n_params} t={elapsed:.0f}s", flush=True)
    return results


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    data_dir    = os.path.join(base, "data")
    results_dir = os.path.join(base, "results")
    os.makedirs(results_dir, exist_ok=True)

    tr_x, tr_y, va_x, va_y, te_x, te_y = load_data(data_dir)
    results = run_search(tr_x, tr_y, va_x, va_y, te_x, te_y)

    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"\nBest: layers={best['n_layers']} w={best['width']} "
          f"drop={best['dropout']} lr={best['lr']:.0e} bs={best['batch_size']} "
          f"val_acc={best['best_val_acc']:.4f} test_acc={best['test_acc']:.4f} "
          f"params={best['n_params']}")

    out = {"best": best, "all_results": results}
    path = os.path.join(results_dir, "mlp_best.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {path}")
