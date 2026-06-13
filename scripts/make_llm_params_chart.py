#!/usr/bin/env python3
"""
Regenerate the LLM-parameter-count chart, extended to 2026.

Usage:
    python scripts/make_llm_params_chart.py
Output:
    img/llm-number-of-parameters-2026.svg
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Company colours ───────────────────────────────────────────────────────────
COLORS = {
    "OpenAI":           "#0ea982",  # teal-green   (original)
    "Google":           "#126eff",  # blue          (original)
    "Microsoft/Nvidia": "#f7b408",  # amber         (original)
    "Huawei":           "#cf0a2c",  # red           (original)
    "Anthropic":        "#cd9d7b",  # tan           (original)
    "Meta":             "#e85d04",  # vivid orange  (new)
    "DeepSeek":         "#7b2fbe",  # purple        (new)
    "xAI":              "#888888",  # grey          (new)
    "Alibaba":          "#e91e8c",  # deep pink     (new)
    "Mistral":          "#2ec4b6",  # teal-turquoise (new)
}

# ── Model data ────────────────────────────────────────────────────────────────
# (company, label, point_year, point_params_B, vertical_label)
# vertical_label=True → 90° rotated text; label_x/label_y unused.
# Parameter counts for undisclosed models are industry estimates.
#
# label_x, label_y: text anchor in DATA COORDINATES (year, B-params).
# These are placed manually to avoid overlap.  rotation is always 30°.
MODELS = [
    # company              label                               point yr    params  vert    lx       ly
    ("OpenAI",           "OpenAI GPT-1",                    2018+5/12,    0.117,  True,   None, None),
    ("Google",           "Google BERT",                     2018+10/12,   0.34,   True,   None, None),
    ("OpenAI",           "OpenAI GPT-2",                    2019+1/12,    1.5,    True,   None, None),
    ("Google",           "Google XLNet",                    2019+5/12,    0.34,   True,   None, None),
    ("Google",           "Google T5",                       2019+9/12,    11,     True,   None, None),

    ("OpenAI",           "OpenAI\nGPT-3",                   2020+4/12,    175,    False,  2020.6,  280),
    ("Microsoft/Nvidia", "Microsoft-Nvidia\nMegatron-Turing-NLG", 2021+9/12, 530, False, 2021.1,  640),
    ("Google",           "Google\nGLaM",                    2021+11/12,   1200,   False,  2022.15, 1310),
    # Minerva pushed left so Gemini 1.0 can occupy the space above it
    ("Google",           "Google\nMinerva",                 2022+5/12,    540,    False,  2022.0,  610),
    # PanGu-Σ and GPT-4 share the same month (Mar 2023)
    # PanGu-Σ label lowered (below GLaM) to reduce stacking
    ("Huawei",           "Huawei\nPanGu-Σ",            2023+2/12,    1085,   False,  2022.3,  950),
    ("OpenAI",           "OpenAI\nGPT-4",                   2023+2/12,    1650,   False,  2023.25, 1780),
    # Gemini 1.0 – between Minerva (610B) and PanGu-Σ (950B) in y space
    ("Google",           "Google DeepMind\nGemini 1.0",     2023+11/12,   1460,   False,  2022.9,  770),
    # Gemini 1.5 and Claude 3 – stagger: 1.5 upper-right, Claude 3 further right
    ("Google",           "Google DeepMind\nGemini 1.5",     2024+1/12,    2250,   False,  2024.3,  2430),
    ("Anthropic",        "Anthropic\nClaude 3",             2024+2/12,    1875,   False,  2024.6,  1700),

    # ── new 2024-2026 data ───────────────────────────────────────────────────
    # GPT-4o: estimated ~200 B
    ("OpenAI",           "OpenAI\nGPT-4o",                  2024+4/12,    200,    False,  2024.5,  310),
    # Llama 3.1 405B: confirmed by Meta
    ("Meta",             "Meta\nLlama 3.1 405B",            2024+6/12,    405,    False,  2024.75, 540),
    # DeepSeek V3: 671 B total MoE – label left to avoid right-side crowd
    ("DeepSeek",         "DeepSeek\nV3",                    2024+11/12,   671,    False,  2024.25, 870),
    # DeepSeek R1: same 671 B MoE – label right and down
    ("DeepSeek",         "DeepSeek\nR1",                    2025+0/12,    671,    False,  2025.4,  400),
    # Grok 3: xAI, estimated ~3 T total MoE – near top of y-axis
    ("xAI",              "xAI\nGrok 3",                     2025+1/12,    3000,   False,  2025.4,  2920),
    # Gemini 2.5 Pro: estimated ~1.5 T – below Grok 3 label
    ("Google",           "Google DeepMind\nGemini 2.5 Pro", 2025+2/12,    1500,   False,  2025.75, 1270),
    # Llama 4 Maverick: 400 B total MoE – lower right
    ("Meta",             "Meta\nLlama 4",                   2025+3/12,    400,    False,  2026.1,  560),
    # Claude Opus 4: estimated ~1.9 T – upper right
    ("Anthropic",        "Anthropic\nClaude Opus 4",        2025+4/12,    1875,   False,  2026.2,  2200),

    # ── additional 2025-2026 models ──────────────────────────────────────────
    # Qwen3-235B: 235 B total (22 B active MoE), confirmed by Alibaba, Apr 2025
    ("Alibaba",          "Alibaba\nQwen3 235B",              2025+3/12,    235,    False,  2025.0,  130),
    # Mistral Large 3: 675 B total (41 B active MoE), confirmed, Dec 2025
    ("Mistral",          "Mistral\nLarge 3",                 2025+11/12,   675,    False,  2026.25, 850),
    # GPT-5: OpenAI, Aug 2025; params not disclosed; industry estimate ~3 T
    ("OpenAI",           "OpenAI\nGPT-5",                   2025+7/12,    3000,   False,  2025.85, 3600),
    # GPT-5.5: OpenAI, Apr 2026; params not disclosed; estimate ~4.5 T
    ("OpenAI",           "OpenAI\nGPT-5.5",                 2026+3/12,    4500,   False,  2026.45, 4800),
    # Claude Fable 5: Anthropic, Jun 2026; params not disclosed; estimate ~5 T
    ("Anthropic",        "Anthropic\nClaude Fable 5",       2026+5/12,    5000,   False,  2026.6,  5300),
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12.5, 5.20))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ROT = 30

for company, label, year, params, vert, lx, ly in MODELS:
    color = COLORS[company]
    ax.scatter(year, params, color=color, s=18, zorder=5, linewidths=0)

    if vert:
        # Vertical text, offset a few points above the dot
        ax.annotate(
            label, xy=(year, params),
            xytext=(0, 6), textcoords="offset points",
            rotation=90, va="bottom", ha="center",
            fontsize=8, fontfamily="DejaVu Sans", color="black",
        )
    else:
        # Angled text anchored at explicit data-coordinate position.
        # Use a thin grey connector line only when the label is far from
        # the dot (distance in display pixels > threshold).
        ax.annotate(
            label,
            xy=(year, params),
            xytext=(lx, ly),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color="#aaaaaa",
                lw=0.6,
                shrinkA=4,
                shrinkB=4,
            ),
            rotation=ROT, va="bottom", ha="left",
            fontsize=8, fontfamily="DejaVu Sans", color="black",
            multialignment="left", linespacing=1.2,
            annotation_clip=False,
        )

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(2017.8, 2028.0)
ax.set_xticks(range(2018, 2027))
ax.set_xticklabels([str(y) for y in range(2018, 2027)],
                   fontsize=9, fontfamily="DejaVu Sans")

ax.set_ylim(0, 6000)
ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
ax.set_yticklabels(["0", "1000", "2000", "3000", "4000", "5000", "6000"],
                   fontsize=9, fontfamily="DejaVu Sans")
ax.set_ylabel("billions of parameters", fontsize=9, fontfamily="DejaVu Sans")

# Full box, same as original
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("black")
ax.tick_params(axis="both", which="major", length=3.5, width=0.8, color="black",
               direction="out")

# ── Save ──────────────────────────────────────────────────────────────────────
out = "img/llm-number-of-parameters-2026.svg"
plt.savefig(out, format="svg", bbox_inches="tight", dpi=72)
print(f"Saved {out}")
