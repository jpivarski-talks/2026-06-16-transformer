"""
Hyperparameter search for FullTransformer (encoder-decoder).

Training formulation (no train/inference mismatch):
  - src: the first token of each valid sentence, shape (batch, 1, vocab)
  - tgt_input: [BOS, tok0, tok1, ..., tok6], shape (batch, 8, vocab)
  - tgt_target: [tok0, tok1, ..., tok7] as integer indices, shape (batch, 8)
  - Causal mask on tgt prevents decoder from peeking ahead.
  - Loss: CrossEntropy over all 8 positions.

Inference (exact same src as training):
  - src: one-hot("THE"), shape (n, 1, vocab)
  - Start tgt = [BOS, THE]; generate tokens one by one.
  - Evaluate with temperature sampling for diverse outputs.

Results saved to full_transformer_search_results.csv.
"""

import csv
import itertools
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from lark import Lark

# ── Grammar / validity ─────────────────────────────────────────────────────────

ENTITY_CONS  = ["CAT", "MOUSE", "ROBOT"]
ENTITY_VOWEL = ["ALLIGATOR", "ANT"]
PLACE_CONS   = ["HOUSE", "YARD", "STREET"]
PLACE_VOWEL  = ["AIRPORT", "OCEAN"]
INTRANS      = ["SMILES", "BURPS", "SLEEPS", "EXPLODES"]
TRANS        = ["CHASES", "EATS", "LOVES", "HATES"]
PREPS        = ["IN", "UNDER", "ABOVE"]

def _alt(*words):
    return " | ".join(f'"{w}"' for w in words)

_GRAMMAR = f"""
    sentence: subj intrans
            | subj trans obj
            | pp subj intrans
            | subj intrans pp
            | subj pp intrans
            | pp subj trans obj
            | subj trans obj pp
            | subj pp trans obj

    subj: "THE" entity | "A" entity_cons | "AN" entity_vowel
    obj:  "THE" entity | "A" entity_cons | "AN" entity_vowel
    pp:   prep "THE" place | prep "A" place_cons | prep "AN" place_vowel

    entity:       entity_cons | entity_vowel
    entity_cons:  {_alt(*ENTITY_CONS)}
    entity_vowel: {_alt(*ENTITY_VOWEL)}
    place:        place_cons | place_vowel
    place_cons:   {_alt(*PLACE_CONS)}
    place_vowel:  {_alt(*PLACE_VOWEL)}

    prep:   {_alt(*PREPS)}
    intrans:{_alt(*INTRANS)}
    trans:  {_alt(*TRANS)}

    %ignore " "
"""

_parser = Lark(_GRAMMAR, start="sentence", parser="lalr")
BLANK = "<BLANK>"

def is_valid(tokens_8):
    hit_blank = False
    for t in tokens_8:
        if t == BLANK:
            hit_blank = True
        elif hit_blank:
            return False
    core = [t for t in tokens_8 if t != BLANK]
    if not core:
        return False
    try:
        _parser.parse(" ".join(core))
        return True
    except Exception:
        return False


# ── Vocabulary ─────────────────────────────────────────────────────────────────

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
INDEX_TOKEN = {v: k for k, v in TOKEN_INDEX.items()}


# ── Data ──────────────────────────────────────────────────────────────────────

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

# Valid sentences only; src = first token, tgt = full sentence
train_valid = train_x[train_y == 1].to(device)   # (N, 8, vocab)
valid_valid  = valid_x[valid_y == 1].to(device)

# src: first token only → shape (N, 1, vocab)
train_src = train_valid[:, :1, :]
valid_src  = valid_valid[:, :1, :]

# tgt_input: [BOS, tok0, ..., tok6] where BOS = zero vector
def make_decoder_input(x):
    bos = torch.zeros(x.size(0), 1, VOCABULARY_SIZE, device=x.device)
    return torch.cat([bos, x[:, :-1, :]], dim=1)   # (N, 8, vocab)

train_tgt_input  = make_decoder_input(train_valid)
valid_tgt_input  = make_decoder_input(valid_valid)
train_tgt_target = train_valid.argmax(dim=-1)      # (N, 8) integer indices
valid_tgt_target = valid_valid.argmax(dim=-1)

print(f"  train valid: {len(train_valid)}, val valid: {len(valid_valid)}", flush=True)
print("Done.\n", flush=True)


# ── Model ──────────────────────────────────────────────────────────────────────

class FullTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=1, num_layers=1, dim_feedforward=64):
        super().__init__()
        self.input_embedding    = nn.Linear(VOCABULARY_SIZE, d_model, bias=False)
        # position embedding covers max(src_len=1, tgt_len=8) = 8 positions
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, VOCABULARY_SIZE)

    def forward(self, src, tgt, tgt_mask=None):
        src_pos = torch.arange(src.size(1), device=src.device)
        tgt_pos = torch.arange(tgt.size(1), device=tgt.device)
        src_emb = self.input_embedding(src) + self.position_embedding(src_pos)
        tgt_emb = self.input_embedding(tgt) + self.position_embedding(tgt_pos)
        memory  = self.encoder(src_emb)
        out     = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_projection(out)   # (batch, tgt_len, VOCABULARY_SIZE)


# ── Training ──────────────────────────────────────────────────────────────────

CAUSAL_MASK = torch.triu(
    torch.full((SEQUENCE_LENGTH, SEQUENCE_LENGTH), float("-inf"), device=device),
    diagonal=1,
)

def train_generative(model, num_epochs=100, batch_size=64, lr=1e-3, weight_decay=5e-4):
    loader    = DataLoader(
        TensorDataset(train_src, train_tgt_input, train_tgt_target),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.CrossEntropyLoss()
    best_val, wait = float("inf"), 0

    for _ in range(num_epochs):
        model.train()
        for src_b, tgt_in_b, tgt_out_b in loader:
            out  = model(src_b, tgt_in_b, tgt_mask=CAUSAL_MASK)   # (batch, 8, vocab)
            loss = loss_fn(out.view(-1, VOCABULARY_SIZE), tgt_out_b.view(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(valid_src, valid_tgt_input, tgt_mask=CAUSAL_MASK)
            val_loss = loss_fn(val_out.view(-1, VOCABULARY_SIZE), valid_tgt_target.view(-1)).item()

        if val_loss < best_val - 1e-4:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= 5:
                break


# ── Evaluation ─────────────────────────────────────────────────────────────────

N_EVAL   = 200
TEMP     = 1.5    # temperature for sampling (higher → more diversity)

def eval_validity(model, n=N_EVAL, temperature=TEMP):
    """Sample n sentences from THE and return the fraction passing is_valid."""
    model.eval()

    # src: one-hot THE, shape (n, 1, vocab) — same as training format
    src = torch.zeros(n, 1, VOCABULARY_SIZE, device=device)
    src[:, 0, TOKEN_INDEX["THE"]] = 1.0

    # tgt starts as [BOS, THE], shape (n, 2, vocab)
    tgt = torch.zeros(n, 2, VOCABULARY_SIZE, device=device)
    tgt[:, 1, TOKEN_INDEX["THE"]] = 1.0

    generated = [["THE"] for _ in range(n)]

    with torch.no_grad():
        for _ in range(SEQUENCE_LENGTH - 1):   # generate 7 more tokens
            tgt_len = tgt.size(1)
            mask = torch.triu(
                torch.full((tgt_len, tgt_len), float("-inf"), device=device),
                diagonal=1,
            )
            out      = model(src, tgt, tgt_mask=mask)        # (n, tgt_len, vocab)
            logits   = out[:, -1, :] / temperature           # (n, vocab)
            probs    = F.softmax(logits, dim=-1)
            next_idxs = torch.multinomial(probs, num_samples=1).squeeze(1)  # (n,)

            new_oh = torch.zeros(n, 1, VOCABULARY_SIZE, device=device)
            new_oh[torch.arange(n), 0, next_idxs] = 1.0
            tgt = torch.cat([tgt, new_oh], dim=1)

            for i, idx in enumerate(next_idxs.tolist()):
                generated[i].append(INDEX_TOKEN[idx])

    return sum(is_valid(s) for s in generated) / n


# ── Search ─────────────────────────────────────────────────────────────────────

GRID = {
    "d_model":         [16, 32, 64],
    "num_layers":      [1, 2],
    "dim_feedforward": [64, 128, 256],
    "num_epochs":      [100, 200, 400],
    "batch_size":      [64, 128],
    "lr":              [1e-3, 3e-3],
    "weight_decay":    [0.0, 5e-4],
}

SEEDS     = [0, 1, 2]
N_CONFIGS = 40

all_combos = list(itertools.product(*GRID.values()))
random.seed(42)
selected   = random.sample(all_combos, N_CONFIGS)

outfile = open("full_transformer_search_results.csv", "w", newline="")
writer  = csv.writer(outfile)
writer.writerow(list(GRID.keys()) + ["mean_val", "std_val"] + [f"val_seed{s}" for s in SEEDS])

print(f"{'d_model':>7} {'layers':>6} {'d_ff':>5} {'epochs':>6} {'batch':>5} "
      f"{'lr':>7} {'wd':>7} {'mean_val':>9} {'std_val':>8}", flush=True)
print("-" * 70, flush=True)

best_mean = 0.0

for cfg in selected:
    params = dict(zip(GRID.keys(), cfg))
    vals   = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        model = FullTransformer(
            d_model         = params["d_model"],
            nhead           = 1,
            num_layers      = params["num_layers"],
            dim_feedforward = params["dim_feedforward"],
        ).to(device)
        train_generative(
            model,
            num_epochs   = params["num_epochs"],
            batch_size   = params["batch_size"],
            lr           = params["lr"],
            weight_decay = params["weight_decay"],
        )
        vals.append(eval_validity(model))

    mean_val = float(np.mean(vals))
    std_val  = float(np.std(vals))

    writer.writerow(list(cfg) + [f"{mean_val:.4f}", f"{std_val:.4f}"] + [f"{v:.4f}" for v in vals])
    outfile.flush()

    marker = " <<<" if mean_val > best_mean else ""
    if mean_val > best_mean:
        best_mean = mean_val

    print(f"{params['d_model']:>7} {params['num_layers']:>6} {params['dim_feedforward']:>5} "
          f"{params['num_epochs']:>6} {params['batch_size']:>5} "
          f"{params['lr']:>7.0e} {params['weight_decay']:>7.0e} "
          f"{mean_val:>9.4f} {std_val:>8.4f}{marker}", flush=True)

outfile.close()
print("\nResults saved to full_transformer_search_results.csv", flush=True)
