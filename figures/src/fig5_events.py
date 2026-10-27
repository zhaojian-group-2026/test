"""Fig. 5: frame-based encoding (RGB frames or accumulated DVS events) vs. DVS + convolutional SNN.

Visual data come from src/sim_events.py (simulated camera pan + thrown ball on a real photo).
"""
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


def activity_map(t0, t1, name, block=10, frac=0.10):
    """Sparse spiking activity of a conv-SNN layer: blocks with the most input events fire."""
    _, e = window(t0, t1)
    gh, gw = H_ // block, W_ // block
    cnt = np.zeros((gh, gw))
    np.add.at(cnt, ((e[:, 1] // block).astype(int), (e[:, 0] // block).astype(int)), 1)
    thr = np.quantile(cnt, 1 - frac)
    act = cnt >= max(thr, 1)
    img = np.ones((gh, gw, 3)) * DARK
    img[act] = ON_C
    img = np.repeat(np.repeat(img, block, 0), block, 1)
    return save_rgb(img, name), act.mean()


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


def event_cloud(x, y, w, h, dx, dy, n=3600):
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
        F.path(d, stroke=col, sw=1.2, cap="round", opacity=0.75)
    F.arrow([(x + w + 4, y + h + 5), (x + w + dx - 2, y + h + dy + 3)], color=LINE, sw=1, hsize=5)
    F.text(x + w + dx + 4, y + h + dy + 8, "t", size=FS_XS - 1, color=SUB, italic=True, anchor="start")


def takeaway(x, y, l1, l2):
    F.text(x, y - 6, l1, size=FS_XS + 0.5, anchor="start", color=INK, weight="bold")
    F.text(x, y + 12, l2, size=FS_XS, anchor="start", color=SUB)


def arrow_label(x0, x1, y, text):
    F.arrow([(x0, y), (x1, y)], sw=1.4)
    if text:
        F.text((x0 + x1) / 2, y - 7, text, size=FS_XS - 1.5, color=SUB)


RGB = [save_rgb(D[f"rgb{i}"], f"rgb{i}") for i in range(3)]

F = Fig(1000, 582)
PA_Y, PA_H = 6, 284
PB_Y, PB_H = PA_Y + PA_H + 10, 276
F.panel(6, PA_Y, 988, PA_H, label="a", title="Frame-based: RGB frames or accumulated events → vision encoder",
        title_size=FS_S + 1, pad=11)
F.panel(6, PB_Y, 988, PB_H, label="b", title="DVS → spiking neural network: purely event-driven",
        title_size=FS_S + 1, pad=11)

# ================================================================== (a) two lanes, one encoder
L1, L2 = PA_Y + 98, PA_Y + 212
MX = 252                      # middle column (dense grid / event frame)
EX0, EX1 = 420, 534           # shared vision encoder
# lane 1: RGB frames -> all pixels
tile_stack(66, L1 - 42, 116, 87, RGB, dx=14, dy=-9)
F.text(124, L1 + 60, "RGB frames, 30 fps", size=FS_XS - 1, color=INK)
arrow_label(192, MX - 8, L1, "")
GC = 9
for r in range(8):
    for c in range(12):
        F.rect(MX + c * GC, L1 - 36 + r * GC, GC - 1.2, GC - 1.2, rx=1, fill=BLUE[1], stroke=BLUE[0], sw=0.7)
F.text(MX + 54, L1 + 52, "every pixel, every frame", size=FS_XS - 1, color=SUB)
arrow_label(MX + 12 * GC + 6, EX0 - 4, L1, "")
# lane 2: DVS events -> accumulated event frame
event_cloud(40, L2 - 42, 96, 72, 50, 20)
F.text(100, L2 + 66, "event stream (x, y, t, p)", size=FS_XS - 1, color=INK)
arrow_label(196, MX - 8, L2, "sum ΔT")
EVF = event_frame(44, 50, "evframe")
F.image(MX, L2 - 40, 108, 81, EVF, rx=3)
F.text(MX + 54, L2 + 56, "event frame", size=FS_XS - 1, color=SUB)
arrow_label(MX + 114, EX0 - 4, L2, "")
# shared encoder + features
F.rect(EX0, L1 - 46, EX1 - EX0, L2 - L1 + 92, rx=10, fill=BLUE[1], stroke=BLUE[0], sw=1.4)
F.text((EX0 + EX1) / 2, (L1 + L2) / 2 + 5, "vision encoder", size=FS_S, weight="bold", color=BLUE_D)
F.text((EX0 + EX1) / 2, (L1 + L2) / 2 + 22, "dense, per frame", size=FS_XS - 1.5, color=SUB)
FM = (L1 + L2) / 2
F.arrow([(EX1, FM), (EX1 + 26, FM)], sw=1.4)
F.tokens(EX1 + 32, FM - 34, 5, BLUE, s=10, gap=4, vertical=True)
takeaway(610, L1, "RGB: 33 ms resolution, motion blur", "every pixel is recomputed every frame")
takeaway(610, L2, "DVS as frames: events re-binned over ΔT", "sparsity and μs timing are lost before the encoder")

# ================================================================== (b) DVS -> conv SNN
MID = PB_Y + 176
TW, TH = 150, 112
FX, FY = 92, MID - 56
wins = [(20, 26), (50, 56), (80, 86)]
slices = [event_frame(a, b, f"evslice{k}") for k, (a, b) in enumerate(wins)]
tile_stack(FX, FY, TW, TH, slices, dx=24, dy=-10, labels=["$T_{0}$", "…", "$T_{j}$"])
NC, NR = 16, 12
cw, ch = TW / NC, TH / NR
for i in range(1, NC):
    F.line(FX + i * cw, FY, FX + i * cw, FY + TH, stroke=WHITE, sw=0.5, opacity=0.28)
for j in range(1, NR):
    F.line(FX, FY + j * ch, FX + TW, FY + j * ch, stroke=WHITE, sw=0.5, opacity=0.28)
F.text(FX + TW / 2 - 10, PB_Y + PB_H - 10, "event stream on the pixel grid", size=FS_XS - 1, color=INK)

# active cells (many events in the last window) and one silent cell
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


# conv-SNN slabs (sheets seen at an angle), large -> small
def slab(x, ymid, h, depth=30, skew=14, thick=7):
    y0, y1 = ymid - h / 2, ymid + h / 2
    F.poly([(x + depth, y0 + skew), (x + depth + thick, y0 + skew - 3), (x + depth + thick, y1 - skew - 3),
            (x + depth, y1 - skew)], fill="#E9B98C", stroke=ORANGE[0], sw=1)
    F.poly([(x, y0), (x + thick, y0 - 3), (x + depth + thick, y0 + skew - 3), (x + depth, y0 + skew)],
           fill="#F4D5B8", stroke=ORANGE[0], sw=1)
    F.poly([(x, y0), (x + depth, y0 + skew), (x + depth, y1 - skew), (x, y1)], fill=ORANGE[1], stroke=ORANGE[0], sw=1.2)
    return (x, y0, y1, depth, skew)


S1 = slab(452, MID, 156)
S2 = slab(518, MID, 118)
S3 = slab(578, MID, 84)
for a, b in ((S1, S2), (S2, S3)):
    F.line(a[0] + a[3], a[1] + a[4], b[0], b[1], stroke=ORANGE[0], sw=0.8, opacity=0.6)
    F.line(a[0] + a[3], a[2] - a[4], b[0], b[2], stroke=ORANGE[0], sw=0.8, opacity=0.6)
F.text(530, PB_Y + PB_H - 10, "convolutional SNN", size=FS_XS, weight="bold", color=ORANGE_D)

# neurons on the first slab: one per drawn pixel, ⋮ for the many others
x1n = S1[0] + 9
nys = [S1[1] + 22 + k * (S1[2] - S1[1] - 44) / (len(cells) - 1) for k in range(len(cells))]
for k in range(len(cells) - 1):
    F.text(x1n + 1, (nys[k] + nys[k + 1]) / 2 + 4, "⋮", size=FS_XS - 1, color=ORANGE_D)


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
    nx, ny = x1n, nys[k]
    p0, p3 = (px + cw / 2, py), (nx - 6, ny)
    p1, p2 = (px + 100, py), (nx - 100, ny)
    d = f"M{f(p0[0])},{f(p0[1])} C{f(p1[0])},{f(p1[1])} {f(p2[0])},{f(p2[1])} {f(p3[0])},{f(p3[1])}"
    F.path(d, stroke=GREY[0] if active else GRID, sw=1.2, dash=None if active else "4 3")
    if active:
        n = int(np.clip(2 + cnt[r, c] / cnt.max() * 5, 2, 6))
        for t in np.sort(rng.uniform(0.34, 0.9, n)):
            qx, qy = bez(p0, p1, p2, p3, t)
            F.line(qx, qy - 6, qx, qy + 6, stroke=ORANGE[0], sw=2.2, cap="butt")
    F.circle(nx, ny, 5, fill=ORANGE[0] if active else WHITE, stroke=ORANGE[0] if active else GREY[0], sw=1.1)
    if not active:
        F.text(nx - 12, ny - 5, "no spikes", size=FS_XS - 1.5, color=SUB, anchor="end")
F.text(352, PB_Y + 44, "one spike train per pixel", size=FS_XS - 1, color=ORANGE_D)

# callout: a single spiking neuron
CX, CY, CRX, CRY = 660, PB_Y + 74, 92, 40
src = (S3[0] + S3[3] / 2, MID - 22)
F.line(*src, CX - CRX * 0.62, CY + CRY * 0.78, stroke=LINE, sw=0.9, dash="2 2")
F.line(*src, CX + CRX * 0.05, CY + CRY, stroke=LINE, sw=0.9, dash="2 2")
F.circle(*src, 2.5, fill=INK)
F.ellipse(CX, CY, CRX, CRY, fill=WHITE, stroke=LINE, sw=1.1)
F.text(CX, CY - CRY + 15, "spiking neuron", size=FS_XS - 1, weight="bold", color=INK)
NCX, NCY = CX + 10, CY + 6
for k, dy in enumerate((-16, 0, 16)):
    x0 = CX - 74
    F.path(f"M{x0},{NCY + dy} C{x0 + 40},{NCY + dy} {NCX - 34},{NCY + dy * 0.4} {NCX - 13},{NCY + dy * 0.2}",
           stroke=LINE, sw=1)
    for tt in ((0.18, 0.5), (0.3,), (0.12, 0.42))[k]:
        tx = x0 + tt * 90
        F.line(tx, NCY + dy - 5, tx, NCY + dy + 5, stroke=ORANGE[0], sw=1.8, cap="butt")
F.circle(NCX, NCY, 13, fill=ORANGE[1], stroke=ORANGE[0], sw=1.4)
vx = [NCX - 8, NCX - 4, NCX - 4, NCX, NCX, NCX + 3, NCX + 3, NCX + 7]
vy = [NCY + 6, NCY + 2, NCY + 5, NCY - 1, NCY + 3, NCY - 7, NCY + 6, NCY + 6]
F.poly(list(zip(vx, vy)), closed=False, stroke=INK, sw=1)
F.line(NCX + 13, NCY, CX + 80, NCY, stroke=LINE, sw=1)
F.line(CX + 58, NCY - 6, CX + 58, NCY + 6, stroke=ORANGE[0], sw=2.2, cap="butt")

# spiking activity map
AM, frac_on = activity_map(*wins[2], "spikemap")
AX0, AW, AH = 648, 112, 84
F.arrow([(S3[0] + S3[3] + 10, MID + 30), (AX0 - 6, MID + 30)], sw=1.4)
F.image(AX0, MID + 30 - AH / 2, AW, AH, AM, rx=3)
F.text(AX0 + AW / 2, MID + 30 + AH / 2 + 15, f"spiking activity (~{round(frac_on * 100)}% active)",
       size=FS_XS - 1, color=SUB)
takeaway(790, MID - 4, "compute only on spikes", "static pixels: no spikes,")
F.text(790, MID + 25, "no computation", size=FS_XS, anchor="start", color=SUB)

F.save("out/fig5_events.svg")
print("ok", round(frac_on, 3))
