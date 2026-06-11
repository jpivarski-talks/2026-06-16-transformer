"""
Generate train/validate/test datasets for the transformer notebook.

Each row is 8 tokens (padded with <BLANK>) plus a label: "ok" or "bad".
The datasets are balanced: exactly 50% ok, 50% bad.

Strategy:
  Phase 1 — generate sentences with 1-4 random modifications each.
            collect valid ones (some survive modification) and invalid ones.
  Phase 2 — fill any remaining valid quota with unmodified valid sentences.
  Final   — shuffle and write.
"""

import csv
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from grammar import (
    ALL_TOKENS, BLANK, SEQ_LEN, is_valid, all_valid_sentences,
)


def random_valid_sentence(valid_pool, rng):
    return list(rng.choice(valid_pool))


def apply_modification(seq, rng):
    """Apply one random replacement or swap to an 8-token sequence."""
    op = rng.choice(["replace", "swap"])
    seq = list(seq)
    if op == "replace":
        pos = rng.randrange(SEQ_LEN)
        seq[pos] = rng.choice(ALL_TOKENS)
    else:  # swap
        i, j = rng.sample(range(SEQ_LEN), 2)
        seq[i], seq[j] = seq[j], seq[i]
    return seq


def generate_dataset(n, seed):
    rng = random.Random(seed)
    target = n // 2          # want this many valid and this many invalid
    valid_pool_src = all_valid_sentences()

    valid_rows   = []
    invalid_rows = []

    # Phase 1: modified sentences
    while len(invalid_rows) < target or len(valid_rows) < target:
        sentence = random_valid_sentence(valid_pool_src, rng)
        k = rng.randint(1, 4)
        for _ in range(k):
            sentence = apply_modification(sentence, rng)
        if is_valid(sentence):
            if len(valid_rows) < target:
                valid_rows.append(sentence)
        else:
            if len(invalid_rows) < target:
                invalid_rows.append(sentence)

    # Phase 2: fill valid shortfall with unmodified sentences
    while len(valid_rows) < target:
        valid_rows.append(random_valid_sentence(valid_pool_src, rng))

    rows = [(s, "ok") for s in valid_rows] + [(s, "bad") for s in invalid_rows]
    rng.shuffle(rows)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"t{i}" for i in range(SEQ_LEN)] + ["label"])
        for seq, label in rows:
            writer.writerow(seq + [label])
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    specs = [
        ("train.csv",    1_000_000, 42),
        ("validate.csv",   100_000, 43),
        ("test.csv",       100_000, 44),
    ]
    for filename, n, seed in specs:
        path = os.path.join(base, filename)
        rows = generate_dataset(n, seed)
        write_csv(path, rows)
