"""Fig. 3: slow-fast asynchronous VLA, four ways the slow system conditions the fast system.

Top: asynchronous execution through a buffer (slow writes, fast reads the latest entry).
Bottom: what is written into the buffer - (a) latent vector, (b) mid-layer tokens,
(c) nothing explicit: shared parameters, (d) a coarse action chunk.
"""
import math
from figlib import *

F = Fig(1000, 526)

# ================================================================ top: timing through a buffer
TY = 6
F.panel(6, TY, 988, 176, title="Asynchronous slow–fast execution through a buffer", title_size=FS)
X0, X1 = 150, 982
ys, yb, yf = TY + 58, TY + 104, TY + 148
F.text(20, ys + 2, "slow system", size=FS_S, anchor="start", weight="bold", color=BLUE_D)
F.text(20, ys + 17, "VLM, ≈1–10 Hz", size=FS_XS - 0.5, anchor="start", color=SUB)
F.text(20, yb + 2, "buffer", size=FS_S, anchor="start", weight="bold", color=INK)
F.text(20, yb + 17, "latest slow output", size=FS_XS - 0.5, anchor="start", color=SUB)
F.text(20, yf + 2, "fast system", size=FS_S, anchor="start", weight="bold", color=ORANGE_D)
F.text(20, yf + 17, "≈25–200 Hz", size=FS_XS - 0.5, anchor="start", color=SUB)

seg = 206
ends = []
for k in range(4):
    a = X0 + k * seg
    b = min(a + seg - 6, X1)
    F.rect(a, ys - 12, b - a, 24, rx=5, fill=BLUE[1], stroke=BLUE[0], sw=1.2)
    F.text((a + b) / 2, ys + 4.5, f"VLM inference on $o_{{t_{{{k}}}}}$" if k < 3 else "…", size=FS_XS, color=BLUE_D)
    ends.append(b)
# buffer versions: b_k becomes available when inference k finishes
edges = [X0] + ends[:3] + [X1]
for k in range(4):
    a, b = edges[k], edges[k + 1]
    F.rect(a + 1, yb - 11, b - a - 2, 22, rx=4, fill=PANEL, stroke=GRID, sw=1)
    F.text((a + b) / 2, yb + 5, "$b_{-1}$" if k == 0 else f"$b_{{{k - 1}}}$", size=FS_XS + 0.5, color=INK)
for k, e in enumerate(ends[:3]):
    F.arrow([(e - 2, ys + 12), (e - 2, yb - 12)], color=BLUE[0], sw=1.5, hsize=7)
F.text(ends[0] + 6, (ys + yb) / 2 + 4, "write", size=FS_XS - 0.5, color=BLUE_D, anchor="start")
# fast steps read the latest buffer entry
step = 18.5
xs = [X0 + 6 + j * step for j in range(int((X1 - X0 - 10) / step) + 1)]
for j, x in enumerate(xs):
    F.rect(x - 5, yf - 7, 10, 14, rx=2, fill=ORANGE[1], stroke=ORANGE[0], sw=1)
    if j % 4 == 2:
        F.line(x, yf - 8, x, yb + 12, stroke=ORANGE[0], sw=1, dash="2 2")
F.text(xs[2] + 6, (yb + yf) / 2 + 3, "read latest", size=FS_XS - 0.5, color=ORANGE_D, anchor="start")
F.text(X1, TY + 22, "each fast step: latest buffer entry + fresh proprio / obs", size=FS_XS - 0.5,
       anchor="end", color=SUB)

# ================================================================ bottom: four interfaces
CW, CG, CY, CH = 238, 12.67, 192, 328
cols = [
    ("a", "Latent vector → fast", "compact global intent, low bandwidth", "", ""),
    ("b", "Mid-layer tokens → fast", "rich spatial–semantic tokens", "", ""),
    ("c", "Shared parameters", "no explicit message between systems", "", ""),
    ("d", "Action chunk → fast", "explicit motion prior, refined fast", "", ""),
]
NL = 6  # drawn VLM layers


def vlm_stack(x, y, w, kind):
    """VLM drawn as a stack of layers; returns centre x and the layer y-positions (bottom → top)."""
    F.rect(x, y, w, 92, rx=7, fill=BLUE[1], stroke=BLUE[0], sw=1.3)
    F.text(x + 10, y + 16, "slow VLM", size=FS_XS, anchor="start", weight="bold", color=BLUE_D)
    lys = []
    for i in range(NL):
        ly = y + 82 - i * 11
        top = i >= NL - 2
        shared = kind == "c" and top
        tap = kind == "b" and i == 2
        F.rect(x + 14, ly - 4.5, w - 28, 8, rx=2.5,
               fill=ORANGE[1] if shared else (BLUE[0] if tap else WHITE),
               stroke=ORANGE[0] if shared else BLUE[0], sw=0.9)
        lys.append(ly)
    return x + w / 2, lys


for i, (lab, title, n1, n2, eg) in enumerate(cols):
    x = 6 + i * (CW + CG)
    cx = x + CW / 2
    F.panel(x, CY, CW, CH, label=lab, title=title, title_size=FS_S + 0.5, pad=11)
    # slow system
    vy = CY + 34
    _, lys = vlm_stack(x + 16, vy, CW - 32, lab)
    # buffer zone
    by0, by1 = vy + 116, vy + 182
    F.rect(x + 16, by0, CW - 32, by1 - by0, rx=6, fill=WHITE, stroke=GREY[0], sw=1.1, dash="4 3")
    F.text(x + 24, by0 + 13, "buffer" if lab != "c" else "cached in the VLM", size=FS_XS - 1.5,
           anchor="start", color=SUB)
    # fast system
    fy0, fy1 = vy + 208, vy + 252
    F.rect(x + 16, fy0, CW - 32, fy1 - fy0, rx=7, fill=ORANGE[1], stroke=ORANGE[0], sw=1.3)
    fast_title = "fast policy" if lab != "c" else "last 2 VLM blocks"
    fast_sub = {"a": "proprio + latest obs", "b": "cross-attends to tokens", "c": "same weights θ, fast inputs",
                "d": "refines / tracks Â"}[lab]
    F.text(cx, fy0 + 18, fast_title, size=FS_XS + 0.5, weight="bold", color=ORANGE_D)
    F.text(cx, fy0 + 34, fast_sub, size=FS_XS - 1, color=SUB)
    # output
    F.arrow([(cx, fy1), (cx, fy1 + 16)], color=ORANGE[0], sw=1.3, hsize=6)
    F.text(cx + 8, fy1 + 14, "$a_{t}$", size=FS_XS + 0.5, color=ORANGE_D, anchor="start")
    # write / read arrows
    if lab != "c":
        F.arrow([(cx, vy + 92), (cx, by0 - 1)], color=BLUE[0], sw=2.2, hsize=8)
        F.text(cx + 8, vy + 106, "write", size=FS_XS - 1.5, color=BLUE_D, anchor="start")
    F.arrow([(cx, by1 + 1), (cx, fy0 - 1)], color=ORANGE[0], sw=2.2, hsize=8)
    F.text(cx + 8, by1 + 16, "read", size=FS_XS - 1.5, color=ORANGE_D, anchor="start")
    # footer
    F.text(cx, fy1 + 34, n1, size=FS_XS - 0.5, color=SUB)

    bcx, bcy = cx, (by0 + by1) / 2 + 6
    if lab == "a":
        F.tokens(bcx - 3 * 14 + 2, bcy - 6, 6, BLUE, s=12, gap=2, fills=[BLUE[0]] * 6)
        F.text(bcx, bcy + 22, "one vector $z ∈ R^{d}$", size=FS_XS - 0.5, color=INK)
    elif lab == "b":
        F.tokens(bcx - 4 * 16 + 2, bcy - 6, 8, BLUE, s=12, gap=4)
        F.text(bcx, bcy + 22, "N tokens $h^{(l)}_{1:N}$ of layer $l$", size=FS_XS - 0.5, color=INK)
        # tap from the highlighted middle layer
        ly = lys[2]
        F.arrow([(x + CW - 30, ly), (x + CW - 8, ly), (x + CW - 8, bcy - 1), (bcx + 4 * 16 + 4, bcy - 1)],
                color=BLUE[0], sw=1.5, radius=6, hsize=6)
        F.text(x + CW - 34, ly + 3.5, "layer $l$", size=FS_XS - 1.5, color=WHITE, anchor="end")
    elif lab == "c":
        F.tokens(bcx - 4 * 16 + 2, bcy - 6, 8, GREY, s=12, gap=4, fills=[BLUE[1]] * 8)
        F.text(bcx, bcy + 22, "slow features of lower layers (KV)", size=FS_XS - 0.5, color=INK)
        # the fast box IS the top VLM blocks: dashed link to the orange layers
        top = lys[-1]
        F.path(f"M{x + 18},{top} C{x + 2},{top} {x + 2},{fy0 + 22} {x + 16},{fy0 + 22}",
               stroke=ORANGE[0], sw=1.3, dash="4 3")
        F.text(x + CW - 30, lys[-2] + 3.5, "shared", size=FS_XS - 2, color=ORANGE_D, anchor="end")
        F.arrow([(cx, vy + 92), (cx, by0 - 1)], color=BLUE[0], sw=1.4, hsize=6, dash="3 3")
        F.text(cx + 8, vy + 106, "cache", size=FS_XS - 1.5, color=BLUE_D, anchor="start")
    elif lab == "d":
        pts = []
        for j in range(40):
            u = j / 39
            pts.append((bcx - 82 + 164 * u, bcy - 6 - 14 * math.sin(math.pi * (u * 1.2 + 0.1))))
        F.poly(pts, closed=False, stroke=BLUE[0], sw=1.6, dash="5 3")
        for j in range(0, 40, 8):
            F.circle(*pts[j], 3.2, fill=BLUE[0])
        F.circle(*pts[39], 3.2, fill=BLUE[0])
        F.text(bcx, bcy + 24, "coarse chunk $\\hat{A}$ (waypoints)", size=FS_XS - 0.5, color=INK)

F.save("out/fig3_async.svg")
print("ok")
