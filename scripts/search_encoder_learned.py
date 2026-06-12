"""
Random hyperparameter search for Encoder("learned"), nhead=1, num_layers=1.
Runs each configuration over multiple seeds and reports mean ± std test accuracy.
Results are printed as they complete and saved to search_results.csv.
"""

import csv
import itertools
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Data ──────────────────────────────────────────────────────────────────────

BLANK = "<BLANK>"
SEQUENCE_LENGTH = 8

TOKEN_INDEX = {
    x: i for i, x in enumerate([
        "THE", "A", "AN",
        "CAT", "MOUSE", "ROBOT", "ALLIGATOR", "ANT",
        "HOUSE", "YARD", "STREET", "AIRPORT", "OCEAN",
        "IN", "UNDER", "ABOVE",
        "SMILES", "BURPS", "SLEEPS", "EXPLODES",
        "CHASES", "EATS", "LOVES", "HATES",
        BLANK,
    ])
}
VOCABULARY_SIZE = len(TOKEN_INDEX)


def df_to_onehot(df):
    x = torch.zeros((len(df), SEQUENCE_LENGTH, VOCABULARY_SIZE))
    token_columns = [f"t{i}" for i in range(SEQUENCE_LENGTH)]
    indexes = df[token_columns].apply(lambda col: col.map(TOKEN_INDEX).values)
    for i, colname in enumerate(token_columns):
        x[torch.arange(len(df)), i, indexes[colname]] = 1
    y = torch.tensor((df["label"] == "ok").astype(int).values, dtype=torch.long)
    return x, y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

print("Loading data...", flush=True)
train_x, train_y = df_to_onehot(pd.read_csv("data/train.csv",    nrows=20000))
valid_x, valid_y = df_to_onehot(pd.read_csv("data/validate.csv", nrows=20000))
testy_x, testy_y = df_to_onehot(pd.read_csv("data/test.csv",     nrows=20000))
train_x, train_y = train_x.to(device), train_y.to(device)
valid_x, valid_y = valid_x.to(device), valid_y.to(device)
testy_x, testy_y = testy_x.to(device), testy_y.to(device)
print("Done.\n", flush=True)

# ── Model ─────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, d_model=32, dim_feedforward=128):
        super().__init__()
        self.input_embedding  = nn.Linear(VOCABULARY_SIZE, d_model, bias=False)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=1, dim_feedforward=dim_feedforward,
            dropout=0, batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(layer, num_layers=1)
        self.decision = nn.Linear(d_model, 2)

    def forward(self, x):
        emb = self.input_embedding(x)
        emb = emb + self.position_embedding(torch.arange(x.size(1), device=x.device))
        return self.decision(self.encoder(emb).mean(dim=1))


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, num_epochs=50, batch_size=256, lr=1e-3, weight_decay=5e-4):
    loader    = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.CrossEntropyLoss()
    best_val, wait = float("inf"), 0

    for _ in range(num_epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(valid_x), valid_y).item()

        if val_loss < best_val - 1e-4:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= 5:   # early stopping: patience=5
                break


def test_accuracy(model):
    model.eval()
    with torch.no_grad():
        preds = model(testy_x).argmax(1)
    return (preds == testy_y).float().mean().item()


# ── Search space ──────────────────────────────────────────────────────────────

GRID = {
    "d_model":         [16, 32, 64, 128],
    "dim_feedforward": [64, 128, 256, 512],
    "num_epochs":      [50, 100, 200],
    "batch_size":      [64, 128, 256, 512],
    "lr":              [1e-4, 5e-4, 1e-3, 3e-3],
    "weight_decay":    [0.0, 1e-4, 5e-4, 1e-3],
}

SEEDS      = [0, 1, 2, 3, 4]   # 5 seeds per config for stability estimate
N_CONFIGS  = 60                 # random configs to try (full grid is 4^6 = 4096)

all_combos = list(itertools.product(*GRID.values()))
random.seed(42)
selected   = random.sample(all_combos, N_CONFIGS)

# ── Run ───────────────────────────────────────────────────────────────────────

outfile = open("search_results.csv", "w", newline="")
writer  = csv.writer(outfile)
writer.writerow(list(GRID.keys()) + ["mean_acc", "std_acc"] + [f"acc_seed{s}" for s in SEEDS])

print(f"{'d_model':>7} {'d_ff':>5} {'epochs':>6} {'batch':>5} {'lr':>7} {'wd':>7} "
      f"{'mean_acc':>9} {'std_acc':>8}", flush=True)
print("-" * 70, flush=True)

best_mean = 0.0

for cfg in selected:
    params = dict(zip(GRID.keys(), cfg))
    accs = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        model = Encoder(d_model=params["d_model"], dim_feedforward=params["dim_feedforward"]).to(device)
        train(model,
              num_epochs=params["num_epochs"],
              batch_size=params["batch_size"],
              lr=params["lr"],
              weight_decay=params["weight_decay"])
        accs.append(test_accuracy(model))

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    writer.writerow(list(cfg) + [f"{mean_acc:.4f}", f"{std_acc:.4f}"] + [f"{a:.4f}" for a in accs])
    outfile.flush()

    marker = " <<<" if mean_acc > best_mean else ""
    if mean_acc > best_mean:
        best_mean = mean_acc

    print(f"{params['d_model']:>7} {params['dim_feedforward']:>5} {params['num_epochs']:>6} "
          f"{params['batch_size']:>5} {params['lr']:>7.0e} {params['weight_decay']:>7.0e} "
          f"{mean_acc:>9.4f} {std_acc:>8.4f}{marker}", flush=True)

outfile.close()
print("\nResults saved to search_results.csv", flush=True)
