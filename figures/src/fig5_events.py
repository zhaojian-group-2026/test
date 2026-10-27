"""Fig. 5: frame camera + vision encoder vs. DVS (accumulated) + vision encoder vs. DVS + SNN.

Visual data come from src/sim_events.py (simulated camera pan + thrown ball on a real photo).
"""
import math
import os
import numpy as np
from PIL import Image
from figlib import *

D = np.load("assets/fig5_sim.npz")
EV = D["events"]
W_, H_ = 320, 240
os.makedirs("assets/tmp", exist_ok=True)
DARK = np.array([0.075, 0.085, 0.11])
ON_C = np.array([0.867, 0.541, 0.243])
OFF_C = np.array([0.43, 0.63, 0.88])


def save_rgb(arr, name):
    p = f"assets/tmp/{name}.png"
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(p)
    return p


def window(t0, t1):
    m = (EV[:, 2] >= t0) & (EV[:, 2] < t1)
    acc = np.zeros((H_, W_))
    np.add.at(acc, (EV[m, 1].astype(int), EV[m, 0].astype(int)), EV[m, 3])
    return acc, EV[m]


def event_frame(t0, t1, name):
    acc, _ = window(t0, t1)
    img = np.ones((H_, W_, 3)) * DARK
    img[acc > 0] = ON_C
    img[acc < 0] = OFF_C
    return save_rgb(img, name)


RGB = [save_rgb(D[f"rgb{i}"], f"rgb{i}") for i in range(3)]

F = Fig(1000, 606)
ROW_H, GAP = 194, 9
rows_y = [6 + i * (ROW_H + GAP) for i in range(3)]
titles = [("a", "Frame camera → vision encoder"),
          ("b", "DVS → accumulate events into a frame → fuse with RGB → vision encoder"),
          ("c", "DVS → spiking neural network: purely event-driven")]
for (lab, t), y in zip(titles, rows_y):
    F.panel(6, y, 988, ROW_H, label=lab, title=t, title_size=FS_S + 1, pad=11)

ENC_X, ENC_W, TOK_X, TXT_X = 604, 112, 736, 790


def caption(x, y, s, color=SUB, size=FS_XS - 0.5):
    F.text(x, y, s, size=size, color=color)


def arrow_label(x0, x1, y, text):
    F.arrow([(x0, y), (x1, y)], sw=1.4)
    if text:
        F.text((x0 + x1) / 2, y - 7, text, size=FS_XS - 1, color=SUB)


def encoder(y, label="vision encoder", color=BLUE, h=58):
    F.rect(ENC_X, y - h / 2, ENC_W, h, rx=8, fill=color[1], stroke=color[0], sw=1.4)
    F.text(ENC_X + ENC_W / 2, y + 5, label, size=FS_S, weight="bold", color=BLUE_D if color is BLUE else ORANGE_D)


def features(y):
    F.arrow([(ENC_X + ENC_W, y), (TOK_X - 4, y)], sw=1.4)
    F.tokens(TOK_X, y - 34, 5, BLUE, s=10, gap=4, vertical=True)


def takeaway(y, l1, l2):
    F.text(TXT_X, y - 6, l1, size=FS_XS + 0.5, anchor="start", color=INK, weight="bold")
    F.text(TXT_X, y + 12, l2, size=FS_XS, anchor="start", color=SUB)


def tile_stack(x, y, w, h, paths, dx=18, dy=-12, labels=None):
    """Front tile at (x, y); older tiles up-left behind it."""
    n = len(paths)
    for k, p in enumerate(paths):
        off = n - 1 - k
        xx, yy = x - off * dx, y + off * dy
        F.rect(xx - 1.5, yy - 1.5, w + 3, h + 3, rx=4, fill=WHITE, stroke=GRID, sw=1)
        F.image(xx, yy, w, h, p, rx=3)
        if labels:
            F.rect(xx + 4, yy + 4, est_width(labels[k], FS_XS - 2) + 10, 15, rx=3, fill=INK, opacity=0.72)
            F.text(xx + 9, yy + 15, labels[k], size=FS_XS - 2, color=WHITE, anchor="start")


def event_cloud(x, y, w=128, h=96, dx=70, dy=26, n=4200):
    """x-y-t point cloud: RGB frame at t = 0, time runs to the lower right."""
    sx, sy = w / W_, h / H_
    F.g(opacity=0.5)
    F.image(x, y, w, h, RGB[0], rx=0)
    F.end()
    F.rect(x, y, w, h, rx=0, fill="none", stroke=GRID, sw=1)
    F.rect(x + dx, y + dy, w, h, rx=0, fill="none", stroke=GRID, sw=1, dash="3 3")
    for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)]:
        F.line(x + cx, y + cy, x + cx + dx, y + cy + dy, stroke=GRID, sw=1, dash="3 3")
    rng = np.random.default_rng(1)
    idx = rng.choice(len(EV), size=n, replace=False)
    pts = {1: [], -1: []}
    for ex, ey, et, ep in EV[idx]:
        u = et / 100.0
        pts[int(ep)].append((x + ex * sx + u * dx, y + ey * sy + u * dy))
    for pol, col in ((-1, EV_OFF), (1, EV_ON)):
        d = "".join(f"M{f(px)},{f(py)}h0.01" for px, py in pts[pol])
        F.path(d, stroke=col, sw=1.25, cap="round", opacity=0.75)
    F.arrow([(x + w + 4, y + h + 6), (x + w + dx - 2, y + h + dy + 4)], color=LINE, sw=1, hsize=5)
    F.text(x + w + dx / 2 + 10, y + h + dy / 2 + 18, "t", size=FS_XS, color=SUB, italic=True)
    F.text(x + w / 2, y - 5, "x", size=FS_XS - 1, color=SUB, italic=True)
    F.text(x - 6, y + h / 2, "y", size=FS_XS - 1, color=SUB, italic=True, anchor="end")


# ------------------------------------------------------------------ (a)
y0 = rows_y[0]
tile_stack(66, y0 + 70, 136, 102, RGB, labels=None)
caption(134, y0 + 188, "RGB frames every 33 ms (30 fps)", color=INK)
mid = y0 + 116
arrow_label(214, 330, mid, "")
GX, GY, GC = 340, mid - 42, 10.5
for r in range(8):
    for c in range(12):
        F.rect(GX + c * GC, GY + r * GC, GC - 1.2, GC - 1.2, rx=1, fill=BLUE[1], stroke=BLUE[0], sw=0.7)
caption(GX + 6 * GC, mid + 58, "all H×W pixels, every frame", size=FS_XS - 1)
arrow_label(GX + 12 * GC + 10, ENC_X - 10, mid, "")
encoder(mid)
features(mid)
takeaway(mid, "33 ms resolution, motion blur", "dense compute on every frame")

# ------------------------------------------------------------------ (b)
y0 = rows_y[1]
mid = y0 + 112
event_cloud(26, y0 + 48)
caption(118, y0 + 188, "event stream (x, y, t, p)", color=INK)
arrow_label(228, 280, mid, "sum ΔT")
EVF = event_frame(44, 50, "evframe")
F.image(288, mid - 39, 104, 78, EVF, rx=3)
caption(340, mid + 56, "event frame (ΔT = 6 ms)", size=FS_XS - 1)
F.text(410, mid + 7, "⊕", size=FS_T + 4, color=INK)
F.image(428, mid - 39, 104, 78, RGB[1], rx=3)
caption(480, mid + 56, "RGB frame", size=FS_XS - 1)
arrow_label(540, ENC_X - 10, mid, "concat")
encoder(mid)
features(mid)
takeaway(mid, "events re-binned into frames", "sparsity and μs timing lost")

# ------------------------------------------------------------------ (c)
y0 = rows_y[2]
mid = y0 + 112
TW, TH = 152, 114
FX, FY = 92, y0 + 62
wins = [(20, 26), (50, 56), (80, 86)]
slices = [event_frame(a, b, f"evslice{k}") for k, (a, b) in enumerate(wins)]
tile_stack(FX, FY, TW, TH, slices, dx=24, dy=-10, labels=["$T_{0}$", "…", "$T_{j}$"])
# pixel grid on the front slice
NC, NR = 16, 12
cw, ch = TW / NC, TH / NR
for i in range(1, NC):
    F.line(FX + i * cw, FY, FX + i * cw, FY + TH, stroke=WHITE, sw=0.5, opacity=0.28)
for j in range(1, NR):
    F.line(FX, FY + j * ch, FX + TW, FY + j * ch, stroke=WHITE, sw=0.5, opacity=0.28)
caption(FX + TW / 2 - 10, y0 + 188, "event stream on the pixel grid", color=INK)

# choose active cells (many events in the last window) and one silent cell
_, ew = window(*wins[2])
cnt = np.zeros((NR, NC))
np.add.at(cnt, ((ew[:, 1] * NR / H_).astype(int), (ew[:, 0] * NC / W_).astype(int)), 1)
order = np.dstack(np.unravel_index(np.argsort(-cnt, axis=None), cnt.shape))[0]
picked = []
for r, c in order:
    if 1 <= c <= NC - 2 and 1 <= r <= NR - 2 and all(abs(r - r2) + abs(c - c2) >= 4 for r2, c2 in picked):
        picked.append((r, c))
    if len(picked) == 5:
        break
quiet = [(r, c) for r in range(2, NR - 2) for c in range(1, 6) if cnt[r, c] == 0]
silent = quiet[len(quiet) // 2] if quiet else (NR - 3, 2)
cells = sorted(picked + [silent], key=lambda rc: rc[0])

# SNN block with input neurons on its left edge
SX0, SX1 = 516, ENC_X + ENC_W
SY0, SY1 = y0 + 36, y0 + 186
F.rect(SX0, SY0, SX1 - SX0, SY1 - SY0, rx=10, fill=ORANGE[1], stroke=ORANGE[0], sw=1.4)
F.text((SX0 + SX1) / 2 + 26, SY0 + 22, "SNN", size=FS_S + 1, weight="bold", color=ORANGE_D)
hid = [(SX0 + 92 + 40 * a, SY0 + 46 + 26 * b) for a in range(2) for b in range(4)]
nys = [SY0 + 20 + k * (SY1 - SY0 - 40) / (len(cells) - 1) for k in range(len(cells))]
neur = [(SX0 + 26, ny) for ny in nys]
for k, (r, c) in enumerate(cells):
    active = (r, c) != silent
    for hx, hy in hid[:4]:
        F.line(*neur[k], hx, hy, stroke=ORANGE[0] if active else GRID, sw=0.7, opacity=0.5 if active else 0.8)
for a in range(4):
    for b in range(4, 8):
        F.line(*hid[a], *hid[b], stroke=GRID, sw=0.6)
for hx, hy in hid:
    F.circle(hx, hy, 5, fill=WHITE, stroke=ORANGE[0], sw=1)
F.arrow([(SX1, mid), (TOK_X - 4, mid)], sw=1.4)
F.tokens(TOK_X, mid - 34, 5, ORANGE, s=10, gap=4, vertical=True)


def bez(p0, p1, p2, p3, t):
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1])


rng = np.random.default_rng(4)
for k, (r, c) in enumerate(cells):
    active = (r, c) != silent
    px, py = FX + (c + 0.5) * cw, FY + (r + 0.5) * ch
    F.rect(FX + c * cw, FY + r * ch, cw, ch, rx=1, fill="none",
           stroke=ORANGE[0] if active else WHITE, sw=1.6, dash=None if active else "2 1.5")
    nx, ny = neur[k]
    p0, p3 = (px + cw / 2, py), (nx - 9, ny)
    p1, p2 = (px + 120, py), (nx - 120, ny)
    d = f"M{f(p0[0])},{f(p0[1])} C{f(p1[0])},{f(p1[1])} {f(p2[0])},{f(p2[1])} {f(p3[0])},{f(p3[1])}"
    F.path(d, stroke=GREY[0] if active else GRID, sw=1.2, dash=None if active else "4 3")
    if active:
        n = int(np.clip(2 + cnt[r, c] / cnt.max() * 5, 2, 6))
        ts = np.sort(rng.uniform(0.32, 0.9, n))
        for t in ts:
            qx, qy = bez(p0, p1, p2, p3, t)
            F.line(qx, qy - 6, qx, qy + 6, stroke=ORANGE[0], sw=2.2, cap="butt")
    F.circle(nx, ny, 7.5, fill=ORANGE[0] if active else WHITE, stroke=ORANGE[0] if active else GREY[0], sw=1.2)
    if not active:
        F.text(nx - 13, ny - 4, "no spikes", size=FS_XS - 1.5, color=SUB, anchor="end")
F.text(420, SY0 + 2, "one spike train per pixel", size=FS_XS - 1, color=ORANGE_D)
takeaway(mid, "compute only on spikes", "static pixels: no spikes, no compute")

F.save("out/fig5_events.svg")
print("ok", cells, silent)
