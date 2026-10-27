"""Fig. 4: flow matching (top) and three routes to few-/one-step action generation (bottom).

All trajectories are computed, not hand-drawn: 1-D toy with noise eps ~ N(0,1), actions
A ~ 0.5 N(-mu, s^2) + 0.5 N(+mu, s^2), linear path x_tau = (1-tau) eps + tau A.
The marginal velocity v(x, tau) = E[A - eps | x_tau = x] has a closed form for this mixture.
"""
import math
import numpy as np
from figlib import *

F = Fig(1000, 590)
MU, S = 1.35, 0.20
COMPS = [(-MU, 0.5), (MU, 0.5)]


def v_marg(x, tau):
    s2 = S * S
    var = (1 - tau) ** 2 + tau ** 2 * s2
    logs, vks = [], []
    for mu, pi in COMPS:
        logs.append(math.log(pi) - (x - tau * mu) ** 2 / (2 * var) - 0.5 * math.log(var))
        vks.append(mu + (tau * s2 - (1 - tau)) / var * (x - tau * mu))
    m = max(logs)
    w = [math.exp(l - m) for l in logs]
    return sum(wi * vi for wi, vi in zip(w, vks)) / sum(w)


def ode(x0, n=400, t0=0.0, t1=1.0):
    xs, x, h = [(t0, x0)], x0, (t1 - t0) / n
    t = t0
    for _ in range(n):
        k1 = v_marg(x, t)
        k2 = v_marg(x + h / 2 * k1, t + h / 2)
        k3 = v_marg(x + h / 2 * k2, t + h / 2)
        k4 = v_marg(x + h * k3, min(t + h, 1.0))
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        xs.append((t, x))
    return xs


def euler(x0, n, t0=0.0):
    pts, x, t = [(t0, x0)], x0, t0
    h = (1 - t0) / n
    for _ in range(n):
        x += h * v_marg(x, t)
        t += h
        pts.append((t, x))
    return pts


class Plot:
    """Axes for the 1-D toy: horizontal = flow time tau, vertical = action value x."""

    def __init__(self, x, y, w, h, xr=2.6):
        self.x, self.y, self.w, self.h, self.xr = x, y, w, h, xr

    def P(self, tau, xv):
        return (self.x + tau * self.w, self.y + self.h / 2 - xv / self.xr * self.h / 2)

    def frame(self, densities=True, labels=True):
        x, y, w, h = self.x, self.y, self.w, self.h
        F.line(x, y + h + 6, x + w, y + h + 6, stroke=GRID, sw=1)
        F.line(x, y + h + 3, x, y + h + 9, stroke=SUB, sw=1)
        F.line(x + w, y + h + 3, x + w, y + h + 9, stroke=SUB, sw=1)
        if labels:
            F.text(x, y + h + 21, "τ = 0", size=FS_XS - 1, color=SUB)
            F.text(x + w, y + h + 21, "τ = 1", size=FS_XS - 1, color=SUB)
        if densities:
            ys = np.linspace(-self.xr, self.xr, 120)
            pn = np.exp(-ys ** 2 / 2)
            pts = [(x - 4 - 30 * p, self.P(0, yy)[1]) for yy, p in zip(ys, pn / pn.max())]
            F.poly([(x - 4, self.P(0, ys[0])[1])] + pts + [(x - 4, self.P(0, ys[-1])[1])], fill=GREY[1],
                   stroke=GREY[0], sw=1)
            pa = sum(pi * np.exp(-(ys - mu) ** 2 / (2 * S * S)) for mu, pi in COMPS)
            pts = [(x + w + 4 + 30 * p, self.P(1, yy)[1]) for yy, p in zip(ys, pa / pa.max())]
            F.poly([(x + w + 4, self.P(1, ys[0])[1])] + pts + [(x + w + 4, self.P(1, ys[-1])[1])],
                   fill=ORANGE[1], stroke=ORANGE[0], sw=1)

    def curve(self, pts, **kw):
        F.poly([self.P(t, xv) for t, xv in pts], closed=False, **kw)


rng = np.random.default_rng(7)


def streamlines(pl, n=15, op=0.4):
    """Faint background: integral curves of the (learned) marginal field."""
    for q in np.linspace(-2.25, 2.25, n):
        pl.curve(ode(q, 150), stroke=GREY[0], sw=0.8, opacity=op)


# ============================================================ (a) flow matching
F.panel(6, 6, 988, 210, label="a", title="Flow matching: from noise to an action chunk", title_size=FS)
CAP_Y = 202
# (i) conditional straight paths
p1 = Plot(72, 54, 196, 122)
p1.frame(labels=False)
eps = rng.standard_normal(28)
A = np.where(rng.random(28) < 0.5, -MU, MU) + S * rng.standard_normal(28)
for e, a in zip(eps, A):
    F.line(*p1.P(0, e), *p1.P(1, a), stroke=BLUE[0], sw=0.8, opacity=0.5)
F.text(p1.x - 18, p1.y - 4, "noise", size=FS_XS - 1, color=SUB)
F.text(p1.x + p1.w + 20, p1.y - 4, "actions", size=FS_XS - 1, color=ORANGE_D)
F.text(p1.x + p1.w / 2, CAP_Y, "training: straight paths between random (noise, action) pairs", size=FS_XS - 1)

# (ii) learned field + 10 Euler steps
p2 = Plot(404, 54, 196, 122)
p2.frame(labels=False)
for tau in np.linspace(0.06, 0.94, 9):
    for xv in np.linspace(-2.2, 2.2, 9):
        vv = v_marg(xv, tau)
        dx, dy = 1.0 * p2.w, -vv * p2.h / 2 / p2.xr
        L = math.hypot(dx, dy)
        x0_, y0_ = p2.P(tau, xv)
        F.line(x0_, y0_, x0_ + dx / L * 8, y0_ + dy / L * 8, stroke=GREY[0], sw=0.9)
for q in np.linspace(-2.0, 2.0, 13):
    p2.curve(ode(q, 200), stroke=BLUE[0], sw=0.9, opacity=0.55)
eu = euler(0.35, 10)
p2.curve(eu, stroke=ORANGE[0], sw=1.6)
for t, xv in eu:
    F.circle(*p2.P(t, xv), 2.6, fill=ORANGE[0])
F.text(p2.x + p2.w / 2, CAP_Y, "sampling: follow the learned field, N = 10 steps (dots)", size=FS_XS - 1)

# (iii) a 2-D action chunk along the same process
Hn = 50
u = np.linspace(0, 1, Hn)
s0, g0 = np.array([0.12, 0.22]), np.array([0.88, 0.80])
dirv = g0 - s0
perp = np.array([-dirv[1], dirv[0]]) / np.linalg.norm(dirv)
clean = s0[None] + u[:, None] * dirv[None] + 0.20 * np.sin(np.pi * u)[:, None] * perp[None]
noise = 0.5 + 0.22 * rng.standard_normal((Hn, 2))
BXW, BXH, BGAP = 86, 104, 20
BX0, BY0 = 690, 54
for k, tau in enumerate((0.0, 0.5, 1.0)):
    bx = BX0 + k * (BXW + BGAP)
    F.rect(bx, BY0, BXW, BXH, rx=5, fill=WHITE, stroke=ORANGE[0] if tau == 1 else GRID, sw=1)
    pts = np.clip((1 - tau) * noise + tau * clean, 0.02, 0.98)
    P = [(bx + 4 + q[0] * (BXW - 8), BY0 + BXH - 4 - q[1] * (BXH - 8)) for q in pts]
    col = ORANGE[0] if tau == 1 else (GREY[0] if tau == 0 else "#C9A27E")
    F.poly(P, closed=False, stroke=col, sw=0.9 if tau < 1 else 1.5, opacity=0.8)
    for q in P[::2]:
        F.circle(*q, 1.6, fill=col)
    gx, gy = bx + 4 + g0[0] * (BXW - 8), BY0 + BXH - 4 - g0[1] * (BXH - 8)
    F.circle(gx, gy, 5, fill="none", stroke=INK, sw=1.1)
    F.line(gx - 3, gy, gx + 3, gy, stroke=INK, sw=1)
    F.line(gx, gy - 3, gx, gy + 3, stroke=INK, sw=1)
    sx_, sy_ = bx + 4 + s0[0] * (BXW - 8), BY0 + BXH - 4 - s0[1] * (BXH - 8)
    F.rect(sx_ - 4, sy_ - 4, 8, 8, rx=1.5, fill=INK)
    F.text(bx + BXW / 2, BY0 + BXH + 15, {0.0: "τ = 0 (noise)", 0.5: "τ = 0.5", 1.0: "τ = 1 (action)"}[tau],
           size=FS_XS - 1.5, color=INK)
    if k < 2:
        F.arrow([(bx + BXW + 3, BY0 + BXH / 2), (bx + BXW + BGAP - 3, BY0 + BXH / 2)], color=GREY[0], sw=1.1, hsize=5)
F.text(BX0 + (3 * BXW + 2 * BGAP) / 2, CAP_Y, "2-D gripper path of one chunk (■ start, ⊕ target)", size=FS_XS - 1)

# ============================================================ bottom row
BY, BH, BW = 224, 362, 322
heads = [("b", "Distillation: N steps → 1 step"),
         ("c", "One-step objective: mean velocity"),
         ("d", "Warm start and cross-step iteration")]
for i, (lab, title) in enumerate(heads):
    F.panel(6 + i * (BW + 11), BY, BW, BH, label=lab, title=title, title_size=FS_S + 0.5, pad=11)

# ---- (b) distillation
pb = Plot(58, BY + 42, 218, 186)
pb.frame()
streamlines(pb)
for x0, emph in [(-1.75, False), (-0.6, False), (0.25, True), (1.3, False)]:
    tch = euler(x0, 10)
    pb.curve(tch, stroke=BLUE[0], sw=1.2 if emph else 0.9, opacity=1 if emph else 0.6)
    for t, xv in tch:
        F.circle(*pb.P(t, xv), 2.5 if emph else 1.9, fill=BLUE[0], opacity=1 if emph else 0.6)
    a, b = pb.P(0, x0), pb.P(1, tch[-1][1])
    F.arrow([a, b], color=ORANGE[0], sw=2.4 if emph else 1.4, opacity=None if emph else 0.7, hsize=8)
    if emph:
        F.text(b[0] + 6, b[1] + 16, "$\\hat{A}$", size=FS_XS + 1, color=ORANGE_D, anchor="start")
        F.text(a[0] - 6, a[1] + 4, "$x_{0}$", size=FS_XS + 1, color=INK, anchor="end")
LGY = BY + 266
F.line(18, LGY, 34, LGY, stroke=BLUE[0], sw=1.2)
F.circle(26, LGY, 2.5, fill=BLUE[0])
F.text(40, LGY + 4, "teacher: 10 Euler steps", size=FS_XS - 0.5, anchor="start")
F.line(192, LGY, 208, LGY, stroke=ORANGE[0], sw=2.4)
F.text(214, LGY + 4, "student: 1 step", size=FS_XS - 0.5, anchor="start")
F.rect(18, BY + 282, BW - 24, 68, rx=6, fill=PANEL, stroke=GRID, sw=1)
F.text(6 + BW / 2, BY + 308, "$L(φ) = ‖S_{φ}(x_{0}, o) - Φ^{(N)}_{θ}(x_{0}, o)‖^{2}$", size=FS_S)
F.text(6 + BW / 2, BY + 334, "NFE: N → 1, needs a trained teacher", size=FS_XS - 0.5, color=SUB)

# ---- (c) mean velocity
xc0 = 6 + BW + 11
pc = Plot(xc0 + 52, BY + 42, 218, 186)
pc.frame()
streamlines(pc)
x0 = 0.06
traj = ode(x0, 400)
pc.curve(traj, stroke=BLUE[0], sw=1.4)
for tau in (0.08, 0.32, 0.56, 0.8):
    idx = int(round(tau * 400))
    t, xv = traj[idx]
    vv = v_marg(xv, t)
    dx, dy = pc.w, -vv * pc.h / 2 / pc.xr
    L = math.hypot(dx, dy)
    a = pc.P(t, xv)
    F.arrow([a, (a[0] + dx / L * 26, a[1] + dy / L * 26)], color=BLUE[0], sw=1.2, hsize=5.5)
a, b = pc.P(0, x0), pc.P(1, traj[-1][1])
F.arrow([a, b], color=ORANGE[0], sw=2.4, hsize=8)
F.text(a[0] - 6, a[1] + 4, "ε", size=FS_XS + 1, color=INK, anchor="end")
F.text(b[0] + 6, b[1] + 16, "$\\hat{A}$", size=FS_XS + 1, color=ORANGE_D, anchor="start")
F.line(xc0 + 22, LGY, xc0 + 38, LGY, stroke=BLUE[0], sw=1.4)
F.text(xc0 + 44, LGY + 4, "instantaneous $v$ (tangent)", size=FS_XS - 0.5, anchor="start")
F.line(xc0 + 200, LGY, xc0 + 216, LGY, stroke=ORANGE[0], sw=2.4)
F.text(xc0 + 222, LGY + 4, "average $u$ (chord)", size=FS_XS - 0.5, anchor="start")
F.rect(xc0 + 12, BY + 282, BW - 24, 68, rx=6, fill=PANEL, stroke=GRID, sw=1)
F.text(xc0 + BW / 2, BY + 308, "one step:  $\\hat{A} = ε + u_{θ}(ε, 0, 1 \\mid o)$", size=FS_S)
F.text(xc0 + BW / 2, BY + 334, "learn the average velocity directly: no teacher, NFE = 1", size=FS_XS - 0.5, color=SUB)

# ---- (d) warm start + cross-step iteration
xd0 = 6 + 2 * (BW + 11)
nodes = {
    "prev": (xd0 + 16, BY + 42, 136, 46, ORANGE, "previous chunk", "$A_{k-1}$"),
    "shift": (xd0 + BW - 152, BY + 42, 136, 46, GREY, "shift by s, pad tail", "$\\tilde{A}$"),
    "noise": (xd0 + BW - 152, BY + 122, 136, 46, GREY, "re-noise to $τ_{0}$", "$(1-τ_{0})ε + τ_{0}\\tilde{A}$"),
    "den": (xd0 + 16, BY + 122, 136, 46, ORANGE, "K = 1–2 steps, $τ_{0}$ → 1", "new chunk $A_{k}$"),
}
for key, (x, y, w, h, col, t1, t2) in nodes.items():
    F.rect(x, y, w, h, rx=7, fill=col[1], stroke=col[0], sw=1.2)
    F.text(x + w / 2, y + 19, t1, size=FS_XS - 0.5, color=INK, weight="bold" if key in ("prev", "den") else "normal")
    F.text(x + w / 2, y + 36, t2, size=FS_XS, color=INK)
P_, S_, N_, D_ = nodes["prev"], nodes["shift"], nodes["noise"], nodes["den"]
F.arrow([(P_[0] + P_[2], P_[1] + 23), (S_[0], S_[1] + 23)], color=LINE, sw=1.3)
F.arrow([(S_[0] + 68, S_[1] + 46), (N_[0] + 68, N_[1])], color=LINE, sw=1.3)
F.arrow([(N_[0], N_[1] + 23), (D_[0] + D_[2], D_[1] + 23)], color=LINE, sw=1.3)
F.arrow([(D_[0] + 68, D_[1]), (P_[0] + 68, P_[1] + 46)], color=ORANGE[0], sw=1.5)
F.text(D_[0] + 74, (D_[1] + P_[1] + 46) / 2 + 4, "execute s actions, k ← k+1", size=FS_XS - 1.5,
       color=ORANGE_D, anchor="start")
F.arrow([(D_[0] + 40, D_[1] + 66), (D_[0] + 40, D_[1] + 47)], color=BLUE[0], sw=1.4, hsize=6)
F.text(D_[0] + 46, D_[1] + 64, "new observation $o_{k}$", size=FS_XS - 0.5, color=BLUE_D, anchor="start")

CY0 = BY + 196
F.rect(xd0 + 12, CY0, BW - 24, 156, rx=6, fill=WHITE, stroke=RED, sw=1.4, dash="5 3")
F.text(xd0 + BW / 2, CY0 + 22, "$‖T_{o}(A) - T_{o}(A')‖ ≤ ρ‖A - A'‖,$   ρ < 1", size=FS_S)
F.text(xd0 + BW / 2, CY0 + 42, "holds only while the observation $o$ is fixed", size=FS_XS, color=RED, weight="bold")


def draw_star(cx, cy, col, r=6, op=1.0):
    pts = []
    for j in range(10):
        rr = r if j % 2 == 0 else r * 0.45
        a = -math.pi / 2 + j * math.pi / 5
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    F.poly(pts, fill=col, opacity=op)


def mini(x, y, w, h, moving):
    F.rect(x, y, w, h, rx=5, fill=PANEL, stroke=GRID, sw=1)
    rho, ang = 0.62, 1.1
    if not moving:
        star = (x + w * 0.55, y + h * 0.52)
        z = complex(-w * 0.38, -h * 0.30)
        pts = []
        for _ in range(7):
            pts.append((star[0] + z.real, star[1] + z.imag))
            z *= rho * complex(math.cos(ang), math.sin(ang))
        F.poly(pts, closed=False, stroke=BLUE[0], sw=1.0)
        for p in pts:
            F.circle(*p, 2.4, fill=BLUE[0])
        draw_star(*star, ORANGE[0])
        F.text(x + w / 2, y + h + 14, "fixed $o$: converges to $A^{*}(o)$", size=FS_XS - 1.5, color=INK)
    else:
        stars = [(x + w * (0.22 + 0.25 * j), y + h * (0.70 - 0.20 * j)) for j in range(3)]
        for j, s_ in enumerate(stars):
            draw_star(*s_, ORANGE[0], op=0.45 + 0.27 * j)
        F.poly(stars, closed=False, stroke=ORANGE[0], sw=1.0, dash="3 2")
        pts = [(x + w * 0.10, y + h * 0.20)]
        cur = complex(*pts[0])
        for j in range(6):
            tgt = complex(*stars[min(j // 2, 2)])
            cur = tgt + (cur - tgt) * rho * complex(math.cos(ang * 0.6), math.sin(ang * 0.6))
            pts.append((cur.real, cur.imag))
        F.poly(pts, closed=False, stroke=BLUE[0], sw=1.0)
        for p in pts:
            F.circle(*p, 2.4, fill=BLUE[0])
        F.text(x + w / 2, y + h + 14, "$o_{k}$ changes: $A^{*}(o_{k})$ moves", size=FS_XS - 1.5, color=INK)


mini(xd0 + 24, CY0 + 56, 128, 72, moving=False)
mini(xd0 + BW - 152, CY0 + 56, 128, 72, moving=True)

F.save("out/fig4_flow.svg")
print("ok")
