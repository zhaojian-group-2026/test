"""Fig. 5: (a) frame camera + vision encoder, (b) DVS event frames + vision encoder, (c) DVS + SNN.

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

F = Fig(1000, 444)
R1_Y, R1_H = 6, 214
R2_Y, R2_H = R1_Y + R1_H + 10, 208
PW = 489
A0, B0 = 6, 6 + PW + 10
F.panel(A0, R1_Y, PW, R1_H, label="a", title="Frame camera → vision encoder", title_size=FS_S + 1, pad=11)
F.panel(B0, R1_Y, PW, R1_H, label="b", title="DVS → event frame → vision encoder", title_size=FS_S + 1, pad=11)
F.panel(6, R2_Y, 988, R2_H, label="c", title="DVS → spiking neural network: purely event-driven",
        title_size=FS_S + 1, pad=11)

ENC_OFF, ENC_W, ENC_H = 362, 84, 70
MID1 = R1_Y + 100


def encoder_and_tokens(x0, color=BLUE):
    ex = x0 + ENC_OFF
    F.rect(ex, MID1 - ENC_H / 2, ENC_W, ENC_H, rx=8, fill=color[1], stroke=color[0], sw=1.4)
    F.text(ex + ENC_W / 2, MID1 - 3, "vision", size=FS_S, weight="bold", color=BLUE_D)
    F.text(ex + ENC_W / 2, MID1 + 14, "encoder", size=FS_S, weight="bold", color=BLUE_D)
    F.arrow([(ex + ENC_W, MID1), (ex + ENC_W + 18, MID1)], sw=1.4)
    F.tokens(ex + ENC_W + 22, MID1 - 34, 5, BLUE, s=10, gap=4, vertical=True)


def bottom_note(x0, l1, l2):
    F.text(x0 + PW / 2, R1_Y + R1_H - 30, l1, size=FS_XS + 0.5, weight="bold", color=INK)
    F.text(x0 + PW / 2, R1_Y + R1_H - 13, l2, size=FS_XS, color=SUB)


# ------------------------------------------------------------------ (a) RGB frames
tile_stack(A0 + 48, MID1 - 46, 124, 93, RGB, dx=14, dy=-9)
F.text(A0 + 110, MID1 + 62, "RGB frames, 30 fps", size=FS_XS - 1, color=INK)
arrow_label(A0 + 180, A0 + 206, MID1, "")
GX, GC = A0 + 212, 9
for r in range(8):
    for c in range(11):
        F.rect(GX + c * GC, MID1 - 36 + r * GC, GC - 1.2, GC - 1.2, rx=1, fill=BLUE[1], stroke=BLUE[0], sw=0.7)
F.text(GX + 49, MID1 + 52, "every pixel, every frame", size=FS_XS - 1.5, color=SUB)
arrow_label(GX + 11 * GC + 4, A0 + ENC_OFF - 4, MID1, "")
encoder_and_tokens(A0)
bottom_note(A0, "33 ms resolution, motion blur", "dense compute on every pixel of every frame")

# ------------------------------------------------------------------ (b) DVS -> event frame
event_cloud(B0 + 28, MID1 - 42, 92, 69, 48, 18)
F.text(B0 + 92, MID1 + 62, "event stream (x, y, t, p)", size=FS_XS - 1, color=INK)
arrow_label(B0 + 180, B0 + 214, MID1, "sum ΔT")
EVF = event_frame(44, 50, "evframe")
F.image(B0 + 220, MID1 - 40, 107, 80, EVF, rx=3)
F.text(B0 + 273, MID1 + 56, "event frame (ΔT = 6 ms)", size=FS_XS - 1.5, color=SUB)
arrow_label(B0 + 333, B0 + ENC_OFF - 4, MID1, "")
encoder_and_tokens(B0)
bottom_note(B0, "events re-binned into frames", "sparsity and μs timing lost before the encoder")

# ------------------------------------------------------------------ (c) DVS -> SNN
y0 = R2_Y
mid = y0 + 112
TW, TH = 152, 114
FX, FY = 92, y0 + 62
wins = [(20, 26), (50, 56), (80, 86)]
slices = [event_frame(a, b, f"evslice{k}") for k, (a, b) in enumerate(wins)]
tile_stack(FX, FY, TW, TH, slices, dx=24, dy=-10, labels=["$T_{0}$", "…", "$T_{j}$"])
NC, NR = 16, 12
cw, ch = TW / NC, TH / NR
for i in range(1, NC):
    F.line(FX + i * cw, FY, FX + i * cw, FY + TH, stroke=WHITE, sw=0.5, opacity=0.28)
for j in range(1, NR):
    F.line(FX, FY + j * ch, FX + TW, FY + j * ch, stroke=WHITE, sw=0.5, opacity=0.28)
F.text(FX + TW / 2 - 10, y0 + R2_H - 10, "event stream on the pixel grid", size=FS_XS - 0.5, color=INK)

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

# SNN: every layer has the same slots (6 neurons, ⋮, 1 more neuron); ⋯ between layers
SX0, SY0, SY1 = 508, y0 + 36, y0 + 192
LX = [SX0 + 26, SX0 + 96, SX0 + 196]            # input, hidden, last hidden
SX1 = LX[2] + 30
F.rect(SX0, SY0, SX1 - SX0, SY1 - SY0, rx=10, fill=ORANGE[1], stroke=ORANGE[0], sw=1.4)
F.text((SX0 + SX1) / 2, SY0 - 6, "SNN", size=FS_S + 1, weight="bold", color=ORANGE_D)
NSLOT = 8
slot = [SY0 + 16 + k * (SY1 - SY0 - 32) / (NSLOT - 1) for k in range(NSLOT)]
SHOWN = [0, 1, 2, 3, 4, 5, 7]                     # slot 6 is the ⋮
act_in = {k for k, rc in enumerate(cells) if rc != silent}
act_h = {1, 2, 4}
act_h2 = {2, 5}
for i in SHOWN:                                   # input -> hidden (full connections)
    for j in SHOWN:
        on = i in act_in and j in act_h
        F.line(LX[0], slot[i], LX[1], slot[j], stroke=ORANGE[0] if on else GRID, sw=0.9 if on else 0.5,
               opacity=0.8 if on else 0.7)
for j in SHOWN:                                   # stubs around the ⋯ between layers
    F.line(LX[1], slot[j], LX[1] + 26, slot[j] + (slot[3] - slot[j]) * 0.25, stroke=GRID, sw=0.6)
    F.line(LX[2] - 26, slot[j] + (slot[3] - slot[j]) * 0.25, LX[2], slot[j], stroke=GRID, sw=0.6)
F.text((LX[1] + LX[2]) / 2, slot[3] + 5, "⋯", size=FS_T + 2, color=ORANGE_D)
for L, act, r in ((1, act_h, 5.5), (2, act_h2, 5.5)):
    for j in SHOWN:
        on = j in act
        F.circle(LX[L], slot[j], r, fill=ORANGE[0] if on else WHITE, stroke=ORANGE[0], sw=1)
    F.text(LX[L], slot[6] + 4, "⋮", size=FS_S, color=ORANGE_D)
F.text(LX[0], slot[6] + 4, "⋮", size=FS_S, color=ORANGE_D)
F.circle(LX[0], slot[7], 7.5, fill=WHITE, stroke=ORANGE[0], sw=1.2)
TOK_X = SX1 + 26
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
    nx, ny = LX[0], slot[k]
    p0, p3 = (px + cw / 2, py), (nx - 9, ny)
    p1, p2 = (px + 120, py), (nx - 120, ny)
    d = f"M{f(p0[0])},{f(p0[1])} C{f(p1[0])},{f(p1[1])} {f(p2[0])},{f(p2[1])} {f(p3[0])},{f(p3[1])}"
    F.path(d, stroke=GREY[0] if active else GRID, sw=1.2, dash=None if active else "4 3")
    if active:
        n = int(np.clip(2 + cnt[r, c] / cnt.max() * 5, 2, 6))
        for t in np.sort(rng.uniform(0.32, 0.9, n)):
            qx, qy = bez(p0, p1, p2, p3, t)
            F.line(qx, qy - 6, qx, qy + 6, stroke=ORANGE[0], sw=2.2, cap="butt")
    F.circle(nx, ny, 7.5, fill=ORANGE[0] if active else WHITE, stroke=ORANGE[0] if active else GREY[0], sw=1.2)
F.text(400, SY0 + 2, "one spike train per pixel", size=FS_XS - 1, color=ORANGE_D)
TX = TOK_X + 32
F.text(TX, mid - 26, "compute only on spikes", size=FS_XS + 0.5, anchor="start", color=INK, weight="bold")
F.text(TX, mid - 8, "static pixels: no spikes,", size=FS_XS, anchor="start", color=SUB)
F.text(TX, mid + 8, "no compute", size=FS_XS, anchor="start", color=SUB)
LY = mid + 40
F.line(TX, LY, TX + 30, LY, stroke=GREY[0], sw=1.2)
for xx in (TX + 9, TX + 19):
    F.line(xx, LY - 6, xx, LY + 6, stroke=ORANGE[0], sw=2.2, cap="butt")
F.text(TX + 38, LY + 4, "spike train (active pixel)", size=FS_XS - 1.5, anchor="start", color=INK)
F.line(TX, LY + 20, TX + 30, LY + 20, stroke=GRID, sw=1.2, dash="4 3")
F.text(TX + 38, LY + 24, "static pixel: no spikes", size=FS_XS - 1.5, anchor="start", color=INK)

F.save("out/fig5_events.svg")
print("ok", cells, silent)
