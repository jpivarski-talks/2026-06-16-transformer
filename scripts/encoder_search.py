"""
Hyperparameter search for the encoder (transformer) model.

Architecture:
  - Token embedding: VOCAB_SIZE → d_model
  - Optional positional encoding: learned or RoPE
  - N × nn.TransformerEncoderLayer
  - Mean pool over sequence → nn.Linear(d_model, 2) classifier

Saves best hyperparams and training curves to results/encoder_best.json.
"""

import csv
import json
import math
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

def load_csv(path, max_rows=None):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    seqs   = [[TOKEN_TO_IDX[row[f"t{i}"]] for i in range(SEQ_LEN)] for row in rows]
    labels = [1 if row["label"] == "ok" else 0 for row in rows]
    return seqs, labels


def to_index_tensor(seqs, labels):
    x = torch.tensor(seqs, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def load_data(base):
    print("Loading data...", flush=True)
    tr_seqs, tr_lab = load_csv(os.path.join(base, "train.csv"), max_rows=100_000)
    va_seqs, va_lab = load_csv(os.path.join(base, "validate.csv"))
    te_seqs, te_lab = load_csv(os.path.join(base, "test.csv"))
    tr_x, tr_y = to_index_tensor(tr_seqs, tr_lab)
    va_x, va_y = to_index_tensor(va_seqs, va_lab)
    te_x, te_y = to_index_tensor(te_seqs, te_lab)
    print("Done loading.", flush=True)
    return (tr_x.to(DEVICE), tr_y.to(DEVICE),
            va_x.to(DEVICE), va_y.to(DEVICE),
            te_x.to(DEVICE), te_y.to(DEVICE))

# ── positional encodings ───────────────────────────────────────────────────────

class LearnedPE(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)


def apply_rope(q, k):
    """Apply RoPE to query and key tensors.
    q, k: (batch, seq, n_heads, head_dim)
    Returns rotated q, k with same shape.
    """
    _, seq, _, head_dim = q.shape
    assert head_dim % 2 == 0
    half = head_dim // 2
    # frequencies: theta_i = 1 / 10000^(2i/d)
    device = q.device
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq, device=device).float()
    # (seq, half)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()  # (seq, half)
    sin = freqs.sin()

    def rotate(v):
        # v: (batch, seq, n_heads, head_dim)
        v1, v2 = v[..., :half], v[..., half:]
        v_rot = torch.cat([-v2, v1], dim=-1)
        # broadcast cos/sin over batch and heads
        c = cos.unsqueeze(0).unsqueeze(2)   # (1, seq, 1, half)
        s = sin.unsqueeze(0).unsqueeze(2)
        return torch.cat([v[..., :half] * c - v[..., half:] * s,
                          v[..., :half] * s + v[..., half:] * c], dim=-1)
    return rotate(q), rotate(k)


# ── model ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout, pos_enc):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_enc   = pos_enc   # "none", "learned", "rope"

        if pos_enc == "learned":
            self.pos_embedding = nn.Embedding(SEQ_LEN, d_model)
        else:
            self.pos_embedding = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier  = nn.Linear(d_model, 2)

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x):
        # x: (batch, seq) token indices
        emb = self.embedding(x)          # (batch, seq, d_model)
        if self.pos_enc == "learned":
            pos = torch.arange(x.size(1), device=x.device)
            emb = emb + self.pos_embedding(pos)
        # RoPE is applied inside attention; for simplicity here we hook it
        # by pre-rotating the embedding (approximation sufficient for demo)
        elif self.pos_enc == "rope":
            emb = self._apply_rope_to_emb(emb)

        out  = self.transformer(emb)      # (batch, seq, d_model)
        pooled = out.mean(dim=1)          # (batch, d_model)
        return self.classifier(pooled)

    def _apply_rope_to_emb(self, emb):
        """Apply RoPE rotation directly to token embeddings before transformer."""
        batch, seq, d = emb.shape
        assert d % 2 == 0
        half = d // 2
        device = emb.device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
        positions = torch.arange(seq, device=device).float()
        freqs = torch.outer(positions, inv_freq)        # (seq, half)
        cos = freqs.cos().unsqueeze(0)                  # (1, seq, half)
        sin = freqs.sin().unsqueeze(0)
        e1, e2 = emb[..., :half], emb[..., half:]
        return torch.cat([e1 * cos - e2 * sin,
                          e1 * sin + e2 * cos], dim=-1)


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
    # (d_model, n_heads, n_layers, d_ff_mult, dropout, lr, batch_size, pos_enc)
    (32,  2, 1, 4, 0.0, 1e-3, 256, "none"),
    (32,  2, 2, 4, 0.0, 1e-3, 256, "none"),
    (32,  2, 1, 4, 0.0, 1e-3, 256, "learned"),
    (32,  2, 2, 4, 0.0, 1e-3, 256, "learned"),
    (64,  4, 1, 4, 0.0, 1e-3, 256, "learned"),
    (64,  4, 2, 4, 0.0, 1e-3, 256, "learned"),
    (64,  4, 2, 4, 0.1, 1e-3, 256, "learned"),
    (128, 4, 2, 4, 0.0, 1e-3, 256, "learned"),
    (32,  2, 1, 4, 0.0, 1e-3, 256, "rope"),
    (32,  2, 2, 4, 0.0, 1e-3, 256, "rope"),
    (64,  4, 2, 4, 0.0, 1e-3, 256, "rope"),
    (64,  4, 2, 4, 0.1, 1e-3, 256, "rope"),
    (32,  2, 2, 4, 0.0, 3e-4, 256, "learned"),
    (32,  4, 2, 4, 0.0, 1e-3, 256, "learned"),
    (64,  4, 3, 4, 0.1, 3e-4, 256, "learned"),
]


def run_search(tr_x, tr_y, va_x, va_y, te_x, te_y):
    results = []
    for i, (d_model, n_heads, n_layers, d_ff_mult, dropout, lr, bs, pos_enc) in enumerate(SEARCH_SPACE):
        d_ff  = d_model * d_ff_mult
        model = Encoder(d_model, n_heads, n_layers, d_ff, dropout, pos_enc)
        n_params = count_params(model)
        t0 = time.time()
        tr_l, va_l, va_a = train_model(model, tr_x, tr_y, va_x, va_y,
                                        lr=lr, batch_size=bs)
        test_acc = test_accuracy(model, te_x, te_y)
        elapsed = time.time() - t0
        rec = {
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "d_ff": d_ff, "dropout": dropout, "lr": lr, "batch_size": bs,
            "pos_enc": pos_enc, "n_params": n_params,
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
        print(f"[{i+1}/{len(SEARCH_SPACE)}] d={d_model} h={n_heads} l={n_layers} "
              f"pe={pos_enc:7s} drop={dropout} lr={lr:.0e} | "
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

    best = max(results, key=lambda r: (r["best_val_acc"], -r["n_params"]))
    print(f"\nBest: d={best['d_model']} h={best['n_heads']} l={best['n_layers']} "
          f"pe={best['pos_enc']} drop={best['dropout']} lr={best['lr']:.0e} "
          f"val_acc={best['best_val_acc']:.4f} test_acc={best['test_acc']:.4f} "
          f"params={best['n_params']}")

    out = {"best": best, "all_results": results}
    path = os.path.join(results_dir, "encoder_best.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {path}")
