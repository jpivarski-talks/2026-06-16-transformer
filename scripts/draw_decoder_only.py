"""
Draw a decoder-only transformer diagram in the style of attention_research_1.png.
Saves to img/decoder-only.png.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle

# ── Colour palette (matches the original) ─────────────────────────────────────
C_ORANGE = "#F9C98A"   # Masked Multi-Head Attention
C_YELLOW = "#F9F2A0"   # Add & Norm
C_BLUE   = "#A8D8EA"   # Feed Forward
C_GREEN  = "#B5EAD7"   # Linear / Softmax / Output Probabilities
C_PINK   = "#FFD6E0"   # Input Embedding

# ── Layout ────────────────────────────────────────────────────────────────────
FIG_W = 4.2
CX    = FIG_W / 2          # horizontal centre

BOX_W  = 3.0
BOX_H  = 0.54
AN_H   = 0.38              # Add & Norm is shorter

FS_BOX   = 8.5
FS_SMALL = 6.5

# Build figure; ylim is set at the end once we know the top
fig, ax = plt.subplots(figsize=(FIG_W, 9.2))
ax.set_xlim(0, FIG_W)
ax.axis("off")


# ── Drawing helpers ────────────────────────────────────────────────────────────

def box(y_bot, h, color, text, w=BOX_W):
    """Draw a rounded-corner box; y_bot is the bottom edge. Returns top y."""
    rect = FancyBboxPatch(
        (CX - w / 2, y_bot), w, h,
        boxstyle="round,pad=0.07",
        facecolor=color, edgecolor="#888888", linewidth=0.9, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(CX, y_bot + h / 2, text,
            ha="center", va="center", fontsize=FS_BOX, zorder=3)
    return y_bot + h


def arrow(y_from, y_to, x=None, color="black"):
    if x is None:
        x = CX
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=0.9, mutation_scale=10),
                zorder=4)


def plus_circle(y):
    """Positional-encoding add circle; returns top y."""
    r = 0.19
    c = Circle((CX, y + r), r,
               facecolor="white", edgecolor="#888888", linewidth=0.9, zorder=3)
    ax.add_patch(c)
    ax.text(CX, y + r, "+", ha="center", va="center", fontsize=11,
            color="#555555", zorder=4)
    return y + 2 * r


def residual_bypass(y_enter, y_exit):
    """Right-side L-shaped skip connection from y_enter to y_exit."""
    x_side = CX + BOX_W / 2 + 0.28
    x_join = CX + BOX_W / 2
    # vertical run
    ax.plot([x_side, x_side], [y_enter, y_exit],
            color="#999999", lw=0.9, zorder=1)
    # arrow into the Add & Norm top-right corner
    ax.annotate("", xy=(x_join + 0.02, y_exit),
                xytext=(x_side, y_exit),
                arrowprops=dict(arrowstyle="-|>", color="#999999",
                                lw=0.8, mutation_scale=8),
                zorder=2)
    # short horizontal stub at entry
    ax.plot([x_join, x_side], [y_enter, y_enter],
            color="#999999", lw=0.9, zorder=1)


# ── Build diagram bottom-up ────────────────────────────────────────────────────

Y = 0.35   # starting y (bottom of first element)

# "Input tokens" label
ax.text(CX, Y, "Input tokens",
        ha="center", va="bottom", fontsize=FS_BOX, style="italic", color="#444444")
Y += 0.30

arrow(Y, Y + 0.18)
Y += 0.18

# Input Embedding
Y = box(Y, BOX_H, C_PINK, "Input Embedding")
arrow(Y, Y + 0.14)
Y += 0.14

# Positional Encoding add-circle
ax.text(CX - BOX_W / 2 - 0.08, Y + 0.19,
        "Positional\nEncoding",
        ha="right", va="center", fontsize=FS_SMALL, color="#666666", linespacing=1.3)
ax.annotate("", xy=(CX - BOX_W / 2 + 0.01, Y + 0.19),
            xytext=(CX - BOX_W / 2 - 0.08, Y + 0.19),
            arrowprops=dict(arrowstyle="-|>", color="#999999",
                            lw=0.8, mutation_scale=8), zorder=4)
Y = plus_circle(Y)
arrow(Y, Y + 0.22)
Y += 0.22

# ── Dashed repeated block ──────────────────────────────────────────────────────
BLOCK_BOT = Y
GAP_IN = 0.20   # padding inside the dashed box at top and bottom
Y += GAP_IN

# ① Masked Multi-Head Attention
Y_mha_bot = Y
Y = box(Y, BOX_H, C_ORANGE, "Masked Multi-Head Attention")
Y_mha_top = Y
arrow(Y, Y + 0.12)
Y += 0.12

# ② Add & Norm  (residual from below MHA)
Y_an1_bot = Y
Y = box(Y, AN_H, C_YELLOW, "Add & Norm")
Y_an1_top = Y
residual_bypass(Y_mha_bot, Y_an1_top)
arrow(Y, Y + 0.22)
Y += 0.22

# ③ Feed Forward
Y_ff_bot = Y
Y = box(Y, BOX_H, C_BLUE, "Feed Forward")
Y_ff_top = Y
arrow(Y, Y + 0.12)
Y += 0.12

# ④ Add & Norm  (residual from below FF)
Y_an2_bot = Y
Y = box(Y, AN_H, C_YELLOW, "Add & Norm")
Y_an2_top = Y
residual_bypass(Y_ff_bot, Y_an2_top)

Y += GAP_IN
BLOCK_TOP = Y

# Draw the dashed border
dash = FancyBboxPatch(
    (CX - BOX_W / 2 - 0.30, BLOCK_BOT),
    BOX_W + 0.60, BLOCK_TOP - BLOCK_BOT,
    boxstyle="square,pad=0.0",
    facecolor="none", edgecolor="#999999", linewidth=0.9,
    linestyle=(0, (5, 3)), zorder=1,
)
ax.add_patch(dash)
ax.text(CX + BOX_W / 2 + 0.33, BLOCK_BOT + 0.20,
        "N×", ha="left", va="center", fontsize=10, color="#555555")

# ── Linear → Softmax → Output ─────────────────────────────────────────────────
arrow(BLOCK_TOP, BLOCK_TOP + 0.22)
Y = BLOCK_TOP + 0.22

Y = box(Y, BOX_H, C_GREEN, "Linear")
arrow(Y, Y + 0.18)
Y += 0.18

Y = box(Y, BOX_H, C_GREEN, "Softmax")
arrow(Y, Y + 0.18)
Y += 0.18

ax.text(CX, Y + 0.04, "Output Probabilities",
        ha="center", va="bottom", fontsize=FS_BOX, fontweight="bold", color="#333333")

TOP = Y + 0.35

# ── Footnote ──────────────────────────────────────────────────────────────────
ax.text(CX, 0.06,
        "Prompt and generated tokens form one unified sequence",
        ha="center", va="bottom", fontsize=5.8, color="#888888", style="italic")

# ── Fix axes limits now that we know the content extent ───────────────────────
ax.set_ylim(0, TOP + 0.05)

plt.tight_layout(pad=0.15)
plt.savefig("img/decoder-only.png", dpi=180, bbox_inches="tight",
            facecolor="white")
print(f"Saved img/decoder-only.png  (content height ≈ {TOP:.2f})")
