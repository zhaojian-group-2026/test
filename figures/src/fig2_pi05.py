"""Fig. 2: pi0.5-style VLA with the 10-step flow-matching denoising loop made explicit."""
import numpy as np
from figlib import *

F = Fig(1000, 470)
LHAT = "$\\hat{l}$"   # subtask symbol (renders cleaner than a combining hat on ℓ)


def pills(x0, y, n, color, w=17, h=8, gap=7, fills=None):
    for i in range(n):
        F.rect(x0 + i * (w + gap), y - h / 2, w, h, rx=h / 2,
               fill=(fills[i] if fills else color[1]), stroke=color[0], sw=1.0)


def cam_thumb(x, y, w, h, kind):
    F.rect(x, y, w, h, rx=4, fill="#E6E9ED", stroke=GRID, sw=1)
    if kind == "base":
        F.poly([(x + 4, y + h), (x + 12, y + h * 0.48), (x + w - 12, y + h * 0.48), (x + w - 4, y + h)], fill="#CDD2D8")
        F.rect(x + w * 0.20, y + h * 0.56, 13, 9, rx=2, fill=BLUE[1], stroke=BLUE[0], sw=0.9)
        F.circle(x + w * 0.62, y + h * 0.68, 5, fill=ORANGE[1], stroke=ORANGE[0], sw=0.9)
        F.line(x + w * 0.80, y + 2, x + w * 0.70, y + h * 0.42, stroke="#7B8494", sw=4)
    else:
        F.poly([(x, y + h), (x, y + h * 0.35), (x + w, y + h * 0.25), (x + w, y + h)], fill="#CDD2D8")
        F.rect(x + w * 0.38, y + h * 0.52, 15, 10, rx=2, fill=BLUE[1], stroke=BLUE[0], sw=0.9)
        sgn = 1 if kind == "left" else -1
        for dx in (-0.30, 0.30):
            cx = x + w * (0.5 + dx)
            F.poly([(cx - 4, y), (cx + 4, y), (cx + 2 + sgn, y + h * 0.42), (cx - 2 + sgn, y + h * 0.42)], fill="#5E6776")


# ------------------------------------------------------------ VLM (prefix)
VX0, VX1, VY = 14, 598, 222
F.rect(VX0, VY - 34, VX1 - VX0, 68, rx=8, fill=BLUE[1], stroke=BLUE[0], sw=1.3)
pills(VX0 + 16, VY - 24, 23, BLUE, fills=[WHITE] * 23)
pills(VX0 + 16, VY + 24, 23, BLUE, fills=[WHITE] * 23)
F.rect(VX0 - 6, VY - 13, VX1 - VX0 + 12, 26, rx=6, fill=WHITE, stroke=BLUE[0], sw=1.1)
F.text((VX0 + VX1) / 2, VY + 5, "VLM backbone   PaliGemma: SigLIP (400M) + Gemma (2B)", size=FS_S,
       weight="bold", color=BLUE_D)

# inputs under the VLM
IY = 300
for k, (kind, name) in enumerate([("base", "base"), ("left", "left wrist"), ("right", "right wrist")]):
    x = 20 + k * 68
    F.poly([(x + 6, IY - 8), (x + 56, IY - 8), (x + 48, IY - 34), (x + 14, IY - 34)], fill=BLUE[1],
           stroke=BLUE[0], sw=1.1)
    F.text(x + 31, IY - 17, "ViT", size=FS_XS - 1, color=BLUE_D)
    cam_thumb(x, IY, 62, 46, kind)
    F.text(x + 31, IY + 60, name, size=FS_XS - 1.5, color=SUB)
    F.line(x + 31, IY - 34, x + 31, VY + 34, stroke=LINE, sw=1.1)
F.text(118, IY + 78, "multi-view images $o_{t}$", size=FS_XS, color=INK)

chips = [("clean the bedroom", "high-level prompt", False),
         ("state $q_{t}$", "discretized", False),
         ("pick up the pillow", "subtask " + LHAT + " from ①", True)]
cx = 228
chip_x = []
for txt, lab, dashed in chips:
    w = max(est_width(txt, FS_XS), est_width(lab, FS_XS - 1)) + 14
    x = cx + w / 2
    F.rect(cx, IY + 4, w, 24, rx=5, fill=WHITE, stroke=BLUE[0] if dashed else GREY[0], sw=1.1,
           dash="4 3" if dashed else None)
    F.text(x, IY + 20.5, txt, size=FS_XS, color=INK, italic=not txt.startswith("state"))
    F.text(x, IY + 44, lab, size=FS_XS - 1, color=SUB)
    F.line(x, IY + 4, x, VY + 34, stroke=LINE, sw=1.1)
    chip_x.append((x, w))
    cx += w + 9
assert cx < VX1 + 8, cx

# step 1: subtask prediction, fed back as the low-level command
SX, SY = 300, 132
F.arrow([(SX, VY - 34), (SX, SY + 14)], color=BLUE[0], sw=1.3)
st = "pick up the pillow"
sw_ = est_width(st, FS_XS) + 18
F.rect(SX - sw_ / 2, SY - 12, sw_, 26, rx=5, fill=WHITE, stroke=BLUE[0], sw=1.2)
F.text(SX, SY + 5, st, size=FS_XS, color=INK, italic=True)
F.badge(SX - 104, SY - 30, "1", BLUE, r=8, size=11)
F.text(SX - 92, SY - 26, "subtask prediction " + LHAT + " (autoregressive text)", size=FS_XS,
       anchor="start", color=BLUE_D, weight="bold")
cmd_x, cmd_w = chip_x[2]
F.arrow([(SX + sw_ / 2, SY), (VX1 + 13, SY), (VX1 + 13, IY - 12), (cmd_x, IY - 12), (cmd_x, IY + 2)],
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
F.rect(NX - 62, IY + 4, 124, 24, rx=5, fill=WHITE, stroke=GREY[0], sw=1.1)
F.text(NX, IY + 20.5, "noise $A^{0} ∼ N(0, I)$", size=FS_XS, color=INK)
F.arrow([(NX, IY + 4), (NX, VY + 36)], color=LINE, sw=1.2)
F.text(NX, IY + 44, "+ flow time τ (adaRMSNorm)", size=FS_XS - 1, color=SUB)

# Euler step above the expert
EW, EH = 196, 66
EX, EY = AC - EW / 2, 92
F.rect(EX, EY, EW, EH, rx=8, fill=WHITE, stroke=ORANGE[0], sw=1.3)
F.text(AC, EY + 19, "Euler step, δ = 1/10", size=FS_XS, weight="bold", color=ORANGE_D)
F.text(AC, EY + 39, "$A^{τ+δ} = A^{τ} + δ\\,v_{θ}(A^{τ}, τ \\mid o_{t}, \\hat{l})$".replace("\\,", " "),
       size=FS_XS + 0.5, color=INK)
F.text(AC, EY + 57, "τ: 0 → 0.1 → … → 1", size=FS_XS - 1, color=SUB)
F.arrow([(AC, VY - 36), (AC, EY + EH)], color=ORANGE[0], sw=1.3)
F.text(AC + 6, (VY - 36 + EY + EH) / 2 + 4, "$v_{θ}$", size=FS_XS + 1, color=ORANGE_D, anchor="start")
# loop back on the right
LX = AX1 + 24
F.arrow([(EX + EW, EY + EH / 2), (LX, EY + EH / 2), (LX, IY - 12), (AC + 34, IY - 12), (AC + 34, VY + 36)],
        color=ORANGE[0], sw=1.6, radius=9)
F.text(LX + 5, VY + 6, "×10", size=FS_S + 1, weight="bold", color=ORANGE_D, anchor="start")
F.text(AC + 40, IY - 18, "$A^{τ+δ}$", size=FS_XS, color=ORANGE_D, anchor="start")

# output chunk (after 10 steps)
OY = 40
F.arrow([(EX + 22, EY), (EX + 22, OY + 14)], color=ORANGE[0], sw=1.3)
F.text(EX + 28, EY - 10, "after 10 steps", size=FS_XS - 1, color=SUB, anchor="start")
labs = ["$a_{t}$", "$a_{t+1}$", "…", "$a_{t+H-1}$"]
for i, lab in enumerate(labs):
    x = EX - 10 + i * 46
    F.rect(x, OY - 12, 42, 24, rx=5, fill=ORANGE[1] if lab != "…" else WHITE, stroke=ORANGE[0], sw=1.1)
    F.text(x + 21, OY + 5, lab, size=FS_XS, color=INK)
F.badge(EX - 28, OY, "2", BLUE, r=8, size=11)
F.text(EX - 10, OY - 20, "action chunk $A^{1}$, H = 50", size=FS_XS, color=ORANGE_D, weight="bold", anchor="start")

# ------------------------------------------------------------ A^tau along the loop (bridge to Fig. 4)
rng = np.random.default_rng(3)
H = 50
tg = np.linspace(0, 1, H)
clean = 0.55 * np.sin(2 * np.pi * (0.8 * tg + 0.1)) + 0.25 * tg
eps = rng.standard_normal(H) * 0.7
RX0, RW, RH = 884, 106, 40
F.text(RX0 + RW / 2, 40, "one action dim.", size=FS_XS, color=SUB)
F.text(RX0 + RW / 2, 54, "of $A^{τ}$ along the loop", size=FS_XS, color=SUB)
taus = [0.0, 0.3, 0.6, 0.9, 1.0]
for k, tau in enumerate(taus):
    by = 68 + k * 64
    F.rect(RX0, by, RW, RH, rx=4, fill=WHITE, stroke=ORANGE[0] if tau == 1 else GRID, sw=1)
    sig = np.clip((1 - tau) * eps + tau * clean, -1.25, 1.25)
    pts = [(RX0 + 4 + (RW - 8) * i / (H - 1), by + RH / 2 - sig[i] * RH * 0.34) for i in range(H)]
    F.poly(pts, closed=False, stroke=ORANGE[0] if tau == 1 else GREY[0], sw=1.4 if tau == 1 else 1.0)
    F.text(RX0 + RW / 2, by + RH + 13, f"τ = {tau:g}" + ("  (action)" if tau == 1 else "  (noise)" if tau == 0 else ""),
           size=FS_XS - 1, color=INK)

# legend + reading order
LY = 410
F.rect(14, LY - 9, 16, 12, rx=3, fill=BLUE[1], stroke=BLUE[0], sw=1.1)
F.text(36, LY + 1, "prefix (images, prompt, state): one forward pass, KV cached; the action expert attends to it",
       size=FS_XS, anchor="start", color=INK)
F.rect(14, LY + 11, 16, 12, rx=3, fill=ORANGE[1], stroke=ORANGE[0], sw=1.1)
F.text(36, LY + 21, "action expert: 10 forward passes per chunk (NFE = 10) — the loop that one-step generation (Fig. 4) removes",
       size=FS_XS, anchor="start", color=INK)
F.text(14, LY + 46, "Inference: ① predict the subtask " + LHAT + " as text, append it to the prefix, then ② denoise the action chunk.",
       size=FS_XS, anchor="start", color=SUB)

F.save("out/fig2_pi05.svg")
print("ok")
