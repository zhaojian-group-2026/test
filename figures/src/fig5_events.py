"""Fig. 5: frame camera + RGB encoder vs. DVS (accumulated) + RGB encoder vs. DVS + SNN.

Visual data come from src/sim_events.py (simulated DVS on a real background photo).
"""
import io
import math
import os
import numpy as np
from PIL import Image
from figlib import *

D = np.load("assets/fig5_sim.npz")
EV = D["events"]
os.makedirs("assets/tmp", exist_ok=True)


def save_rgb(arr, name):
    p = f"assets/tmp/{name}.png"
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(p)
    return p


def event_frame(t0, t1, name):
    e = EV[(EV[:, 2] >= t0) & (EV[:, 2] < t1)]
    e = e[np.argsort(e[:, 2])]
    img = np.ones((240, 320, 3))
    on, off = np.array([0.867, 0.541, 0.243]), np.array([0.239, 0.435, 0.647])
    for x, y, t, p in e:
        img[int(y), int(x)] = on if p > 0 else off
    return save_rgb(img, name)


RGB = [save_rgb(D[f"rgb{i}"], f"rgb{i}") for i in range(3)]
EVF = event_frame(33, 66, "evframe")

F = Fig(1000, 600)
ROW_H, GAP = 190, 9
rows_y = [6 + i * (ROW_H + GAP) for i in range(3)]
titles = [("a", "Frame camera → RGB encoder"),
          ("b", "DVS → accumulate events → fuse with RGB → RGB encoder"),
          ("c", "DVS → spiking neural network: purely event-driven")]
for (lab, t), y in zip(titles, rows_y):
    F.panel(6, y, 988, ROW_H, label=lab, title=t, title_size=FS_S + 1, pad=11)


def note(x, y, lines, anchor="middle", color=SUB, size=FS_XS - 0.5):
    for i, l in enumerate(lines):
        F.text(x, y + i * 14, l, size=size, color=color, anchor=anchor)


def frame_stack(x, y, w, h, paths):
    for k, p in enumerate(paths):
        xx, yy = x + k * 18, y - k * 12
        F.rect(xx - 1.5, yy - 1.5, w + 3, h + 3, rx=4, fill=WHITE, stroke=GRID, sw=1)
        F.image(xx, yy, w, h, p, rx=3)
    xx, yy = x + 2 * 18, y - 2 * 12
    F.rect(xx + 4, yy + 4, 52, 15, rx=3, fill=INK, opacity=0.7)
    F.text(xx + 30, yy + 15, "t + 66 ms", size=FS_XS - 2.5, color=WHITE)


def dense_encoder(x, y, label):
    """CNN/ViT drawn as shrinking layers; every cell active (dense)."""
    sizes = [(64, 64), (52, 52), (40, 40), (28, 28)]
    for k, (w, h) in enumerate(sizes):
        xx, yy = x + k * 22, y + (64 - h) / 2
        F.rect(xx, yy, w * 0.42, h, rx=3, fill=BLUE[1], stroke=BLUE[0], sw=1.1)
        n = int(h / 8)
        for j in range(n):
            F.rect(xx + 3, yy + 3 + j * (h - 6) / n, w * 0.42 - 6, (h - 6) / n - 1.5, rx=1, fill=BLUE[0], opacity=0.55)
    F.text(x + 50, y + 82, label, size=FS_XS, weight="bold", color=BLUE_D)


def tokens_out(x, y, n=5, color=BLUE):
    F.tokens(x, y, n, color, s=10, gap=3, vertical=True)


def event_cloud(x, y, w=140, h=105, dx=74, dy=28, n=2600, show_frame=True):
    """x-y-t cloud (event_utils style): frame plane at t=0, time runs to the lower right."""
    sx, sy = w / 320, h / 240
    if show_frame:
        F.g(opacity=0.55)
        F.image(x, y, w, h, RGB[0], rx=0)
        F.end()
    F.rect(x, y, w, h, rx=0, fill="none", stroke=GRID, sw=1)
    F.rect(x + dx, y + dy, w, h, rx=0, fill="none", stroke=GRID, sw=1, dash="3 3")
    for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)]:
        F.line(x + cx, y + cy, x + cx + dx, y + cy + dy, stroke=GRID, sw=1, dash="3 3")
    rng = np.random.default_rng(1)
    idx = rng.choice(len(EV), size=min(n, len(EV)), replace=False)
    pts_on, pts_off = [], []
    for ex, ey, et, ep in EV[idx]:
        u = et / 100.0
        px, py = x + ex * sx + u * dx, y + ey * sy + u * dy
        (pts_on if ep > 0 else pts_off).append((px, py))
    for pts, col in ((pts_off, EV_OFF), (pts_on, EV_ON)):
        d = "".join(f"M{f(px)},{f(py)}h0.01" for px, py in pts)
        F.path(d, stroke=col, sw=1.6, cap="round", opacity=0.8)
    F.arrow([(x + w + 4, y + h + 6), (x + w + dx - 2, y + h + dy + 4)], color=LINE, sw=1, hsize=5)
    F.text(x + w + dx / 2 + 10, y + h + dy / 2 + 18, "t", size=FS_XS, color=SUB, italic=True)
    F.text(x + w / 2, y - 5, "x", size=FS_XS - 1, color=SUB, italic=True)
    F.text(x - 6, y + h / 2, "y", size=FS_XS - 1, color=SUB, italic=True, anchor="end")


ENC_X = 600


def arrow_label(x0, x1, y, text):
    F.arrow([(x0, y), (x1, y)], sw=1.4)
    F.text((x0 + x1) / 2, y - 7, text, size=FS_XS - 1, color=SUB)


def takeaway(y, l1, l2):
    F.text(800, y, l1, size=FS_XS + 0.5, anchor="start", color=INK, weight="bold")
    F.text(800, y + 17, l2, size=FS_XS, anchor="start", color=SUB)


# ------------------------------------------------------------------ (a)
y0 = rows_y[0]
frame_stack(26, y0 + 70, 136, 102, RGB)
note(120, y0 + 186 - 2, ["RGB frames at 30 fps (motion blur)"], color=INK)
arrow_label(244, ENC_X - 12, y0 + 112, "every frame, all H×W×3 values")
dense_encoder(ENC_X, y0 + 80, "ResNet / DINOv2")
note(ENC_X + 52, y0 + 176, ["dense MACs"])
F.arrow([(ENC_X + 112, y0 + 112), (ENC_X + 140, y0 + 112)], sw=1.4)
tokens_out(ENC_X + 146, y0 + 82)
takeaway(y0 + 100, "33 ms resolution, blur", "compute ∝ pixels × frame rate")

# ------------------------------------------------------------------ (b)
y0 = rows_y[1]
event_cloud(26, y0 + 46, w=128, h=96, dx=66, dy=24)
note(118, y0 + 184, ["DVS events $(x, y, t, p)$, μs timestamps"], color=INK)
arrow_label(244, 296, y0 + 104, "sum over ΔT")
F.rect(302, y0 + 66, 104, 78, rx=3, fill=WHITE, stroke=GRID, sw=1)
F.image(302, y0 + 66, 104, 78, EVF, rx=3)
note(354, y0 + 160, ["event frame"], size=FS_XS - 1)
F.text(424, y0 + 111, "⊕", size=FS_T + 4, color=INK)
F.image(442, y0 + 66, 104, 78, RGB[2], rx=3)
note(494, y0 + 160, ["RGB frame"], size=FS_XS - 1)
arrow_label(552, ENC_X - 12, y0 + 104, "concat")
dense_encoder(ENC_X, y0 + 72, "ResNet / DINOv2")
note(ENC_X + 52, y0 + 168, ["dense MACs, zeros included"])
F.arrow([(ENC_X + 112, y0 + 104), (ENC_X + 140, y0 + 104)], sw=1.4)
tokens_out(ENC_X + 146, y0 + 74)
takeaway(y0 + 96, "events re-binned into frames", "sparsity and μs timing lost")

# ------------------------------------------------------------------ (c)
y0 = rows_y[2]
event_cloud(26, y0 + 46, w=128, h=96, dx=66, dy=24)
note(118, y0 + 184, ["same event stream, no frames"], color=INK)
arrow_label(244, 296, y0 + 104, "event = spike")
rng = np.random.default_rng(5)
RX, RY = 304, y0 + 58
for r in range(7):
    yy = RY + r * 12
    F.line(RX, yy, RX + 96, yy, stroke=GRID, sw=0.8)
    if r in (2, 3, 4):
        ts = np.sort(rng.random(rng.integers(3, 7))) * 90
        for t in ts:
            F.line(RX + 3 + t, yy - 5, RX + 3 + t, yy + 5, stroke=EV_ON if rng.random() < 0.5 else EV_OFF, sw=1.6)
note(352, y0 + 156, ["spike trains", "(active pixels only)"], size=FS_XS - 1)
F.arrow([(408, y0 + 104), (436, y0 + 104)], sw=1.4)
NX0, NY0 = 450, y0 + 50
active = {(0, 2), (0, 3), (0, 4), (1, 3), (1, 1), (2, 2)}
pos = {(l, j): (NX0 + l * 60, NY0 + j * 16) for l in range(3) for j in range(6)}
for l in range(2):
    for j in range(6):
        for k in range(6):
            a, b = pos[(l, j)], pos[(l + 1, k)]
            on = (l, j) in active and (l + 1, k) in active
            F.line(*a, *b, stroke=ORANGE[0] if on else GRID, sw=1.2 if on else 0.5, opacity=1 if on else 0.6)
for (l, j), (px, py) in pos.items():
    on = (l, j) in active
    F.circle(px, py, 5.5, fill=ORANGE[0] if on else WHITE, stroke=ORANGE[0] if on else GREY[0], sw=1.1)
F.text(NX0 + 60, NY0 + 108, "SNN (LIF neurons)", size=FS_XS, weight="bold", color=ORANGE_D)
note(NX0 + 60, NY0 + 122, ["only spiking neurons update"])
LX, LY, LW, LH = 640, y0 + 50, 142, 88
F.rect(LX, LY, LW, LH, rx=5, fill=WHITE, stroke=GRID, sw=1)
th = LY + 26
F.line(LX + 8, th, LX + LW - 8, th, stroke=GREY[0], sw=1, dash="3 2")
F.text(LX + LW - 10, th - 4, "θ", size=FS_XS, color=SUB, anchor="end")
vpts = []
v, base = 0.0, LY + LH - 14
ins = [10, 22, 30, 38, 70, 78, 86, 92, 116]
for t in range(0, 126):
    v *= 0.965
    if t in ins:
        v += 0.32
    if v >= 1.0:
        vpts.append((LX + 8 + t, th))
        F.line(LX + 8 + t, th, LX + 8 + t, LY + 6, stroke=ORANGE[0], sw=1.8)
        v = 0.0
    vpts.append((LX + 8 + t, base - v * (base - th)))
F.poly(vpts, closed=False, stroke=INK, sw=1.1)
for t in ins:
    F.line(LX + 8 + t, LY + LH - 4, LX + 8 + t, LY + LH - 10, stroke=EV_ON, sw=1.4)
F.text(LX + 8, LY + 14, "membrane $V(t)$", size=FS_XS - 1.5, color=SUB, anchor="start")
note(LX + LW / 2, LY + LH + 14, ["integrate → fire → reset"], size=FS_XS - 1)
takeaway(y0 + 96, "compute only on spikes", "static scene → ≈ no compute")

F.save("out/fig5_events.svg")
print("ok")
