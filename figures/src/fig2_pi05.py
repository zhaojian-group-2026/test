"""Fig. 2: pi0.5-style VLA with the 10-step flow-matching denoising loop made explicit.

Camera frames: pi0.5 (Physical Intelligence et al., 2025, arXiv:2504.16054), Fig. 5 ("4x images"),
cropped by src/fetch_assets.py.
"""
import numpy as np
from figlib import *

F = Fig(1000, 372)
LHAT = "$\\hat{l}$"   # subtask symbol (renders cleaner than a combining hat on ℓ)


def pills(x0, y, n, color, w=17, h=8, gap=7, fills=None):
    for i in range(n):
        F.rect(x0 + i * (w + gap), y - h / 2, w, h, rx=h / 2,
               fill=(fills[i] if fills else color[1]), stroke=color[0], sw=1.0)


# ------------------------------------------------------------ legend / reading order (top-left)
F.rect(14, 16, 16, 12, rx=3, fill=BLUE[1], stroke=BLUE[0], sw=1.1)
F.text(36, 26, "prefix (4 images, prompt, state): one forward pass, KV cached", size=FS_XS, anchor="start")
F.rect(14, 38, 16, 12, rx=3, fill=ORANGE[1], stroke=ORANGE[0], sw=1.1)
F.text(36, 48, "action expert: 10 forward passes per chunk (NFE = 10) on the cached prefix", size=FS_XS,
       anchor="start")
F.text(14, 72, "Inference: ① predict subtask " + LHAT + " as text → append it to the prefix → ② denoise the action chunk",
       size=FS_XS, anchor="start", color=SUB)

# ------------------------------------------------------------ VLM (prefix)
VX0, VX1, VY = 14, 598, 214
F.rect(VX0, VY - 34, VX1 - VX0, 68, rx=8, fill=BLUE[1], stroke=BLUE[0], sw=1.3)
pills(VX0 + 16, VY - 24, 23, BLUE, fills=[WHITE] * 23)
pills(VX0 + 16, VY + 24, 23, BLUE, fills=[WHITE] * 23)
F.rect(VX0 - 6, VY - 13, VX1 - VX0 + 12, 26, rx=6, fill=WHITE, stroke=BLUE[0], sw=1.1)
F.text((VX0 + VX1) / 2, VY + 5, "VLM backbone   PaliGemma: SigLIP (400M) + Gemma (2B)", size=FS_S,
       weight="bold", color=BLUE_D)

# inputs: 4 real camera views + text chips
IY = 286
PW, PH, PG = 66, 43, 5
for k in range(4):
    x = 14 + k * (PW + PG)
    F.poly([(x + 8, IY - 6), (x + PW - 8, IY - 6), (x + PW - 15, IY - 30), (x + 15, IY - 30)], fill=BLUE[1],
           stroke=BLUE[0], sw=1.1)
    F.text(x + PW / 2, IY - 14, "ViT", size=FS_XS - 1.5, color=BLUE_D)
    F.line(x + PW / 2, IY - 30, x + PW / 2, VY + 34, stroke=LINE, sw=1.1)
    F.image(x, IY, PW, PH, f"assets/fig2_cam{k}.png", rx=3)
F.text(14 + (4 * PW + 3 * PG) / 2, IY + PH + 15, "camera views $o_{t}$: front, rear, 2× wrist", size=FS_XS, color=INK)

chips = [("clean the bedroom", "high-level prompt", False),
         ("$q_{t}$", "state (text)", False),
         ("pick up the pillow", "subtask " + LHAT + " from ①", True)]
cx = 14 + 4 * (PW + PG) + 6
chip_x = []
for txt, lab, dashed in chips:
    w = est_width(txt, FS_XS) + 16
    x = cx + w / 2
    F.rect(cx, IY + 8, w, 24, rx=5, fill=WHITE, stroke=BLUE[0] if dashed else GREY[0], sw=1.1,
           dash="4 3" if dashed else None)
    F.text(x, IY + 24.5, txt, size=FS_XS, color=INK, italic=not txt.startswith("$"))
    F.text(x, IY + 48, lab, size=FS_XS - 1, color=SUB)
    F.line(x, IY + 8, x, VY + 34, stroke=LINE, sw=1.1)
    chip_x.append((x, w))
    cx += w + 8
assert cx < VX1 + 10, cx

# step 1: subtask prediction, fed back as the low-level command
SX, SY = 280, 132
F.arrow([(SX, VY - 34), (SX, SY + 14)], color=BLUE[0], sw=1.3)
st = "pick up the pillow"
sw_ = est_width(st, FS_XS) + 18
F.rect(SX - sw_ / 2, SY - 12, sw_, 26, rx=5, fill=WHITE, stroke=BLUE[0], sw=1.2)
F.text(SX, SY + 5, st, size=FS_XS, color=INK, italic=True)
F.badge(SX - 104, SY - 30, "1", BLUE, r=8, size=11)
F.text(SX - 92, SY - 26, "subtask prediction " + LHAT + " (autoregressive text)", size=FS_XS,
       anchor="start", color=BLUE_D, weight="bold")
cmd_x, cmd_w = chip_x[2]
F.arrow([(SX + sw_ / 2, SY), (VX1 + 13, SY), (VX1 + 13, IY - 10), (cmd_x, IY - 10), (cmd_x, IY + 6)],
        color=BLUE[0], sw=1.1, dash="4 3", radius=7)

# ------------------------------------------------------------ action expert + explicit loop
AX0, AX1 = 636, 806
AC = (AX0 + AX1) / 2
F.rect(AX0, VY - 34, AX1 - AX0, 68, rx=8, fill=ORANGE[1], stroke=ORANGE[0], sw=1.3)
pills(AX0 + 16, VY - 24, 6, ORANGE)
pills(AX0 + 16, VY + 24, 6, GREY, fills=["#E5E7EB"] * 6)
F.rect(AX0 - 6, VY - 13, AX1 - AX0 + 12, 26, rx=6, fill=WHITE, stroke=ORANGE[0], sw=1.1)
F.text(AC, VY + 5, "action expert (300M)", size=FS_S, weight="bold", color=ORANGE_D)
for dy in (-6, 6):   # shared attention to the cached prefix KV
    F.line(VX1 + 6, VY + dy, AX0 - 6, VY + dy, stroke=BLUE[0], sw=1.0, dash="2 3")

# noise in (from below)
NX = AC - 34
F.rect(NX - 62, IY + 8, 124, 24, rx=5, fill=WHITE, stroke=GREY[0], sw=1.1)
F.text(NX, IY + 24.5, "noise $A^{0} ∼ N(0, I)$", size=FS_XS, color=INK)
F.arrow([(NX, IY + 8), (NX, VY + 36)], color=LINE, sw=1.2)
F.text(NX, IY + 48, "+ flow time τ (adaRMSNorm)", size=FS_XS - 1, color=SUB)

# Euler step above the expert
EW, EH = 196, 66
EX, EY = AC - EW / 2, 88
F.rect(EX, EY, EW, EH, rx=8, fill=WHITE, stroke=ORANGE[0], sw=1.3)
F.text(AC, EY + 19, "Euler step, δ = 1/10", size=FS_XS, weight="bold", color=ORANGE_D)
F.text(AC, EY + 39, "$A^{τ+δ} = A^{τ} + δ\\,v_{θ}(A^{τ}, τ \\mid o_{t}, \\hat{l})$".replace("\\,", " "),
       size=FS_XS + 0.5, color=INK)
F.text(AC, EY + 57, "τ: 0 → 0.1 → … → 1", size=FS_XS - 1, color=SUB)
F.arrow([(AC, VY - 36), (AC, EY + EH)], color=ORANGE[0], sw=1.3)
F.text(AC + 6, (VY - 36 + EY + EH) / 2 + 4, "$v_{θ}$", size=FS_XS + 1, color=ORANGE_D, anchor="start")
LX = AX1 + 24
F.arrow([(EX + EW, EY + EH / 2), (LX, EY + EH / 2), (LX, IY - 10), (AC + 34, IY - 10), (AC + 34, VY + 36)],
        color=ORANGE[0], sw=1.6, radius=9)
F.text(LX + 5, VY + 6, "×10", size=FS_S + 1, weight="bold", color=ORANGE_D, anchor="start")
F.text(AC + 40, IY - 16, "$A^{τ+δ}$", size=FS_XS, color=ORANGE_D, anchor="start")

# output chunk (after 10 steps)
OY = 38
F.arrow([(EX + 22, EY), (EX + 22, OY + 14)], color=ORANGE[0], sw=1.3)
F.text(EX + 28, EY - 10, "after 10 steps", size=FS_XS - 1, color=SUB, anchor="start")
labs = ["$a_{t}$", "$a_{t+1}$", "…", "$a_{t+H-1}$"]
for i, lab in enumerate(labs):
    x = EX - 10 + i * 46
    F.rect(x, OY - 12, 42, 24, rx=5, fill=ORANGE[1] if lab != "…" else WHITE, stroke=ORANGE[0], sw=1.1)
    F.text(x + 21, OY + 5, lab, size=FS_XS, color=INK)
F.badge(EX - 28, OY, "2", BLUE, r=8, size=11)
F.text(EX - 10, OY - 19, "action chunk $A^{1}$, H = 50", size=FS_XS, color=ORANGE_D, weight="bold", anchor="start")

# ------------------------------------------------------------ A^tau along the loop (bridge to Fig. 4)
rng = np.random.default_rng(3)
H = 50
tg = np.linspace(0, 1, H)
clean = 0.55 * np.sin(2 * np.pi * (0.8 * tg + 0.1)) + 0.25 * tg
eps = rng.standard_normal(H) * 0.7
RX0, RW, RH = 884, 106, 36
F.text(RX0 + RW / 2, 24, "one action dim.", size=FS_XS, color=SUB)
F.text(RX0 + RW / 2, 38, "of $A^{τ}$ along the loop", size=FS_XS, color=SUB)
taus = [0.0, 0.3, 0.6, 0.9, 1.0]
for k, tau in enumerate(taus):
    by = 50 + k * 60
    F.rect(RX0, by, RW, RH, rx=4, fill=WHITE, stroke=ORANGE[0] if tau == 1 else GRID, sw=1)
    sig = np.clip((1 - tau) * eps + tau * clean, -1.25, 1.25)
    pts = [(RX0 + 4 + (RW - 8) * i / (H - 1), by + RH / 2 - sig[i] * RH * 0.34) for i in range(H)]
    F.poly(pts, closed=False, stroke=ORANGE[0] if tau == 1 else GREY[0], sw=1.4 if tau == 1 else 1.0)
    F.text(RX0 + RW / 2, by + RH + 13, f"τ = {tau:g}" + ("  (action)" if tau == 1 else "  (noise)" if tau == 0 else ""),
           size=FS_XS - 1, color=INK)

F.save("out/fig2_pi05.svg")
print("ok")
