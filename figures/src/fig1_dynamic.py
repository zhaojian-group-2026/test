"""Fig. 1: dynamic manipulation tasks (photos) + perception/planning/execution under latency."""
from figlib import *

F = Fig(1000, 616)

# ------------------------------------------------------------------ photos
photos = [("a", "Conveyor sorting", "assets/fig1_conveyor.png", "target keeps moving: grasp on the move"),
          ("b", "Catching a thrown object", "assets/fig1_catch.png", "ballistic target: reach the catch point in time"),
          ("c", "Table-tennis striking", "assets/fig1_pingpong.png", "fast incoming ball: strike at the right instant")]
PW, PH, GAP, X0, Y0 = 322, 226, 11, 6, 6
for i, (lab, title, path, note) in enumerate(photos):
    x = X0 + i * (PW + GAP)
    F.image(x, Y0, PW, PH, path, rx=6)
    tw = est_width(f"({lab}) {title}", FS_S) * 1.02 + 16
    F.rect(x + 8, Y0 + 8, tw, 22, rx=4, fill=INK, opacity=0.78)
    F.text(x + 16, Y0 + 23.5, f"({lab}) {title}", size=FS_S, anchor="start", color=WHITE, weight="bold")
    nw = est_width(note, FS_XS) * 0.98 + 14
    F.rect(x + 8, Y0 + PH - 30, nw, 21, rx=4, fill=INK, opacity=0.62)
    F.text(x + 15, Y0 + PH - 15, note, size=FS_XS, anchor="start", color=WHITE)

# ------------------------------------------------------------------ timeline
PY = 244
F.h = 616
F.panel(6, PY, 988, 366, label="d", title="Perception → planning → execution under latency")
t_obs, t_in, t_out, t_act = 268, 386, 626, 690
xa0, xa1 = 168, 980
LX = 22

# phase brackets
yb = PY + 60
for (a, b, n, name, col) in [(t_obs, t_in, "1", "perception", BLUE), (t_in, t_out, "2", "planning", BLUE),
                             (t_out, 975, "3", "execution", ORANGE)]:
    F.rect(a + 1, yb, b - a - 2, 6, rx=2, fill=col[0], opacity=0.85)
    w = est_width(name, FS_S) * 1.05 + 18
    cx = (a + b) / 2 if b < 900 else a + w / 2 + 6
    F.badge(cx - w / 2 + 7, yb - 11, n, BLUE, r=7.5, size=10.5)
    F.text(cx - w / 2 + 18, yb - 6.5, name, size=FS_S, anchor="start",
           color=ORANGE_D if col is ORANGE else BLUE_D, weight="bold")

# time axis, tick labels above the axis
ya = yb + 40
for x, lab in [(t_obs, "$t_{\\rm{obs}}$"), (t_in, "$t_{\\rm{in}}$"), (t_out, "$t_{\\rm{out}}$"), (t_act, "$t_{\\rm{act}}$")]:
    F.line(x, ya + 4, x, PY + 300, stroke=GRID, sw=1.1, dash="4 3")
    F.line(x, ya - 4, x, ya + 4, stroke=LINE, sw=1.4)
    F.text(x, ya - 9, lab, size=FS_S, color=INK)
F.arrow([(xa0, ya), (xa1, ya)], color=LINE, sw=1.2)
F.text(xa1, ya - 8, "time", size=FS_XS, anchor="end", color=SUB, italic=True)

rows = {"rgb": ya + 40, "prop": ya + 76, "pol": ya + 120, "act": ya + 182, "tgt": ya + 232}
F.text(LX, rows["rgb"] + 5, "RGB camera", size=FS_S, anchor="start", weight="bold")
F.text(LX, rows["prop"] + 5, "proprioception", size=FS_S, anchor="start", weight="bold")
F.text(LX, rows["pol"] + 5, "policy", size=FS_S, anchor="start", weight="bold")
F.text(LX, rows["act"] + 5, "actions", size=FS_S, anchor="start", weight="bold")
F.text(LX, rows["tgt"] - 2, "moving target", size=FS_S, anchor="start", weight="bold")
F.text(LX, rows["tgt"] + 13, "(position)", size=FS_XS, anchor="start", color=SUB)


def frame_icon(cx, cy, active=False):
    col = BLUE[0] if active else GREY[0]
    F.rect(cx - 11, cy - 8, 22, 16, rx=2.5, fill=BLUE[1] if active else WHITE, stroke=col, sw=1.2)
    F.poly([(cx - 8, cy + 5), (cx - 2, cy - 2), (cx + 2, cy + 2), (cx + 5, cy - 1), (cx + 8, cy + 5)],
           fill="none", stroke=col, sw=1.0, closed=False)
    F.circle(cx + 5, cy - 4, 1.6, fill=col)


import math as _m

# RGB frames (~30 Hz); the frame captured at t_obs feeds this inference
y = rows["rgb"]
F.line(xa0, y, xa1, y, stroke=GRID, sw=1)
for x in range(t_obs - 59 * 1, xa1 - 10, 59):
    frame_icon(x, y, active=(x == t_obs))
frame_icon(t_obs, y, active=True)
F.arrow([(t_obs + 14, y - 16), (t_in - 2, y - 16)], color=BLUE[0], sw=1.2, hsize=6)
F.text((t_obs + t_in) / 2 + 6, y - 21, "obs. latency", size=FS_XS - 0.5, color=BLUE_D)

# proprioception (~100 Hz)
y = rows["prop"]
wave = lambda x: y + 6 * _m.sin((x - xa0) / 55.0)  # noqa: E731
F.poly([(x, wave(x)) for x in range(xa0, xa1 - 8, 2)], closed=False, stroke=GRID, sw=1.2)
for x in range(xa0 + 4, xa1 - 10, 14):
    F.circle(x, wave(x), 2.0, fill=GREY[0])
xs = xa0 + 4 + 14 * round((t_in - xa0 - 4) / 14)
F.circle(xs, wave(xs), 3.4, fill=BLUE[0])

# policy inference block (+ previous / next, faded)
y = rows["pol"]
F.rect(xa0 + 2, y - 16, t_in - xa0 - 10, 32, rx=5, fill=PANEL, stroke=GRID, sw=1)
F.text((xa0 + t_in) / 2 - 2, y + 5, "previous inference", size=FS_XS, color=SUB)
F.rect(t_in, y - 19, t_out - t_in, 38, rx=6, fill=BLUE[1], stroke=BLUE[0], sw=1.4)
F.text((t_in + t_out) / 2, y - 2, "policy inference (VLA)", size=FS_S, weight="bold", color=BLUE_D)
F.text((t_in + t_out) / 2, y + 13, "encode observation → generate chunk", size=FS_XS - 0.5, color=SUB)
F.arrow([(t_in + 2, y + 28), (t_out - 2, y + 28)], color=BLUE[0], sw=1.0, hsize=6, tail=True)
F.text((t_in + t_out) / 2, y + 42, "inference latency", size=FS_XS - 0.5, color=BLUE_D)
F.rect(t_out + 8, y - 16, xa1 - t_out - 14, 32, rx=5, fill=PANEL, stroke=GRID, sw=1, dash="4 3")
F.text((t_out + xa1) / 2 + 2, y + 5, "next inference …", size=FS_XS, color=SUB)

# actions: the stale chunk keeps executing until the new chunk arrives
y = rows["act"]
aw = lambda x: y + 5 * _m.sin((x - xa0) / 70.0)  # noqa: E731
for x in range(xa0 + 6, t_act - 6, 18):
    F.rect(x - 4, aw(x) - 4, 8, 8, rx=1.5, fill=WHITE, stroke=GREY[0], sw=1.1)
F.text((xa0 + t_in) / 2 - 4, y + 24, "executing previous chunk", size=FS_XS - 0.5, color=SUB)
F.arrow([(t_out + 2, y - 17), (t_act - 2, y - 17)], color=ORANGE[0], sw=1.0, hsize=6, tail=True)
F.text((t_out + t_act) / 2, y - 23, "exec. latency", size=FS_XS - 0.5, color=ORANGE_D)
cw = lambda x: y + 9 * _m.sin((x - t_act) / 60.0 + 0.6) - 5  # noqa: E731
F.poly([(x, cw(x)) for x in range(t_act, xa1 - 10, 2)], closed=False, stroke=ORANGE[0], sw=1.8)
for j, x in enumerate(range(t_act, xa1 - 10, 18)):
    F.rect(x - 4, cw(x) - 4, 8, 8, rx=1.5, fill=ORANGE[0] if j == 0 else ORANGE[1], stroke=ORANGE[0], sw=1.1)
F.text((t_act + xa1) / 2, y + 26, "new chunk $a_{t:t+H}$, planned from the frame at $t_{\\rm{obs}}$",
       size=FS_XS - 0.5, color=ORANGE_D)

# moving target: seen at t_obs, elsewhere when the first new action lands at t_act
y = rows["tgt"]
F.line(xa0, y, xa1, y, stroke=GRID, sw=1)
F.circle(t_obs, y, 7, fill=INK)
F.text(t_obs, y + 22, "seen here", size=FS_XS, color=INK)
F.circle(t_act, y, 7, fill=WHITE, stroke=INK, sw=1.3, dash="2.5 2")
F.text(t_act, y + 22, "actually here", size=FS_XS, color=INK)
F.arrow([(t_obs + 10, y), (t_act - 10, y)], color=RED, sw=1.6, dash="5 3")
F.text((t_obs + t_act) / 2, y - 10, "target moves $v$Δ during Δ = $t_{\\rm{act}} − t_{\\rm{obs}}$",
       size=FS_XS, color=RED, weight="bold")
F.text(t_act + 40, y - 4, "e.g. Δ = 0.2 s, $v$ = 1 m/s → 20 cm offset", size=FS_XS, anchor="start", color=INK)
F.text(t_act + 40, y + 13, "⇒ plan for the predicted state $\\hat{p}(t_{\\rm{act}})$", size=FS_XS,
       anchor="start", color=INK, weight="bold")

F.save("out/fig1_dynamic.svg")
print("ok")
