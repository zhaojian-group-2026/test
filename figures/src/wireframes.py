"""Mid-fidelity structure sketches (layout + labels) for Figs. 1-5."""
from figlib import *


def ph(F, x, y, w, h, rows, size=FS_XS):
    """Placeholder for an illustration to be drawn in the final figure."""
    F.rect(x, y, w, h, rx=6, fill=WHITE, stroke=GREY[0], sw=1.0, dash="5 4")
    lh = size * 1.3
    y0 = y + h / 2 - (len(rows) - 1) * lh / 2 + size * 0.35
    for i, r in enumerate(rows):
        F.text(x + w / 2, y0 + i * lh, r, size=size, color=SUB, italic=(i == 0))


# ====================================================================== Fig 1
def fig1():
    F = Fig(1000, 580)
    tasks = [
        ("a", "Conveyor sorting",
         ["illustration: side view", "overhead camera + FOV cone", "belt with boxes, target box tracked",
          "ghost box at t+Δ (predicted)", "arm intercepts on the move"],
         ["Track box, estimate $\\hat{v}$", "Predict pose at $t+Δ$, plan grasp", "Sync with belt, grasp on the move"]),
        ("b", "Catching a moving object",
         ["illustration: side view", "observed ball samples (fading dots)", "predicted parabola (dashed)",
          "catch point $p^{*}$ + uncertainty ellipse", "ghost start pose → catch pose"],
         ["Detect ball, fit trajectory", "Predict catch point $(p^{*}, t^{*})$", "Reach $p^{*}$ before $t^{*}$"]),
        ("c", "Table-tennis striking",
         ["illustration: side view", "event camera, table + net", "incoming ball, bounce, hit point",
          "racket arm: back-swing → strike", "swing arc (orange)"],
         ["High-rate ball tracking (events)", "Predict bounce & hitting point", "Timed swing, velocity profile"]),
    ]
    for i, (lab, title, phrows, caps) in enumerate(tasks):
        x = 10 + i * 330
        F.panel(x, 10, 320, 330, label=lab, title=title)
        ph(F, x + 12, 44, 296, 190, phrows)
        for j, (c, cap) in enumerate(zip([BLUE, PURPLE, ORANGE], caps)):
            yy = 262 + j * 25
            F.badge(x + 26, yy - 4, str(j + 1), c)
            F.text(x + 42, yy, cap, size=FS_S, anchor="start")
    # (d) pipeline + latency
    F.panel(10, 352, 980, 218, label="d", title="Closed loop under latency: perception → prediction & planning → execution")
    y0 = 400
    ph(F, 25, y0 + 10, 95, 100, ["sensors", "RGB cam", "event cam", "proprio."])
    stages = [(BLUE, "① Perception", ["detect & track", "$\\hat{x}_{t} = (\\hat{p}_{t}, \\hat{v}_{t})$"]),
              (PURPLE, "② Prediction & planning", ["forecast $\\hat{p}(t+Δ)$", "intercept $(p^{*}, t^{*})$"]),
              (ORANGE, "③ Execution", ["action chunk $a_{t:t+H}$", "high-rate control"])]
    xs = [140, 300, 475]
    ws = [140, 155, 140]
    for (c, t, sub), x, w in zip(stages, xs, ws):
        F.box(x, y0 + 18, w, 84, c, t, sub, title_size=FS_S, sub_size=FS_XS + 0.5)
    F.arrow([(120, y0 + 60), (140, y0 + 60)])
    F.arrow([(280, y0 + 60), (300, y0 + 60)])
    F.arrow([(455, y0 + 60), (475, y0 + 60)])
    F.arrow([(615, y0 + 60), (628, y0 + 60), (628, y0 + 138), (72, y0 + 138), (72, y0 + 110)],
            dash="5 4", color=SUB, radius=8)
    F.text(350, y0 + 155, "closed loop: robot + moving target → new observation", size=FS_XS, color=SUB)
    # latency timeline
    tx0, tx1, ty = 655, 975, y0 + 110
    F.text(815, y0 + 6, "Latency budget", size=FS_S, weight="bold")
    F.arrow([(tx0, ty), (tx1, ty)], color=LINE, sw=1.2)
    segs = [(tx0 + 10, tx0 + 85, BLUE, "sense+perceive"), (tx0 + 85, tx0 + 205, PURPLE, "inference"),
            (tx0 + 205, tx1 - 15, ORANGE, "execute")]
    for a, b, c, t in segs:
        F.rect(a, ty - 14, b - a, 10, rx=2, fill=c[1], stroke=c[0], sw=1)
        F.text((a + b) / 2, ty + 18, t, size=FS_XS, color=c[0])
    F.line(tx0 + 10, ty - 22, tx0 + 10, ty + 4, stroke=INK, sw=1)
    F.line(tx0 + 205, ty - 22, tx0 + 205, ty + 4, stroke=INK, sw=1)
    F.text(tx0 + 10, ty - 26, "$t_{k}$", size=FS_S)
    F.text(tx0 + 205, ty - 26, "$t_{k}+Δ$", size=FS_S)
    F.circle(tx0 + 40, y0 + 38, 7, fill="#E9B949", stroke="#9A7419", sw=1)
    F.circle(tx0 + 175, y0 + 38, 7, fill="none", stroke="#9A7419", sw=1, dash="2 2")
    F.arrow([(tx0 + 50, y0 + 38), (tx0 + 165, y0 + 38)], color=RED, sw=1.3, dash="4 3")
    F.text(tx0 + 107, y0 + 32, "target moves $vΔ$", size=FS_XS, color=RED)
    F.text(815, ty + 42, "e.g. $v$ = 1 m/s, Δ = 100 ms → 10 cm offset", size=FS_XS, color=SUB)
    F.text(815, ty + 58, "⇒ act on the predicted state, not the last observation", size=FS_XS, color=SUB)
    return F


# ====================================================================== Fig 2
def fig2():
    F = Fig(1000, 730)
    T = 120  # top of main blocks
    # inputs
    F.text(95, T + 6, "Inputs $o_{t}$, ℓ", size=FS, weight="bold")
    for k, name in enumerate(["base cam", "left wrist", "right wrist"]):
        ph(F, 20 + k * 52, T + 18, 48, 40, [""])
        F.text(44 + k * 52, T + 72, name, size=10.5, color=SUB)
    F.rect(20, T + 86, 150, 40, rx=6, fill=GREEN[1], stroke=GREEN[0], sw=1)
    F.text(95, T + 102, "prompt ℓ:", size=FS_XS, color=SUB)
    F.text(95, T + 118, "\"clean the bedroom\"", size=FS_XS, italic=True)
    F.rect(20, T + 136, 150, 40, rx=6, fill=GREEN[1], stroke=GREEN[0], sw=1)
    F.text(95, T + 152, "state $q_{t}$ → 256 bins", size=FS_XS)
    F.text(95, T + 168, "(as text tokens)", size=FS_XS, color=SUB)
    # encoders
    F.box(195, T + 18, 95, 56, BLUE, "SigLIP", "ViT, 400M", title_size=FS_S, sub_size=FS_XS)
    F.box(195, T + 102, 95, 56, GREEN, "Tokenizer", "Gemma", title_size=FS_S, sub_size=FS_XS)
    F.arrow([(172, T + 46), (195, T + 46)])
    F.arrow([(172, T + 112), (195, T + 120)])
    F.arrow([(172, T + 156), (195, T + 142)])
    # prefix tokens column
    F.tokens(305, T + 14, 6, BLUE, s=10, gap=3, vertical=True)
    F.tokens(305, T + 98, 4, GREEN, s=10, gap=3, vertical=True)
    F.tokens(305, T + 150, 2, GREEN, s=10, gap=3, vertical=True, fills=["#C8E6D2", "#C8E6D2"])
    F.arrow([(290, T + 46), (303, T + 46)], sw=1.2)
    F.arrow([(290, T + 130), (303, T + 130)], sw=1.2)
    F.text(310, T + 192, "prefix", size=FS_XS, color=SUB)
    # VLM
    vx, vy, vw, vh = 335, T, 210, 330
    F.rect(vx, vy, vw, vh, rx=10, fill=PURPLE[1], stroke=PURPLE[0], sw=1.6)
    F.text(vx + vw / 2, vy + 26, "VLM backbone", size=FS, weight="bold")
    F.text(vx + vw / 2, vy + 44, "PaliGemma · Gemma 2B", size=FS_XS, color=SUB)
    for k in range(6):
        F.rect(vx + 20, vy + 62 + k * 40, vw - 40, 28, rx=4, fill=WHITE, stroke=PURPLE[0], sw=0.9)
        F.text(vx + vw / 2, vy + 81 + k * 40, f"layer {['1', '2', '3', '…', '17', '18'][k]}", size=FS_XS, color=SUB)
    F.arrow([(318, T + 100), (335, T + 100)], sw=1.2)
    # action expert
    ax, ay, aw, ah = 600, T, 170, 330
    F.rect(ax, ay, aw, ah, rx=10, fill=ORANGE[1], stroke=ORANGE[0], sw=1.6)
    F.text(ax + aw / 2, ay + 26, "Action expert", size=FS, weight="bold")
    F.text(ax + aw / 2, ay + 44, "300M · same depth", size=FS_XS, color=SUB)
    for k in range(6):
        F.rect(ax + 18, ay + 62 + k * 40, aw - 36, 28, rx=4, fill=WHITE, stroke=ORANGE[0], sw=0.9)
        F.text(ax + aw / 2, ay + 81 + k * 40, "attn + adaRMSNorm(τ)" if k != 3 else "…", size=FS_XS - 1, color=SUB)
        F.line(vx + vw - 20, ay + 76 + k * 40, ax + 18, ay + 76 + k * 40, stroke=SUB, sw=1, dash="3 3")
    F.text((vx + vw + ax) / 2, vy + vh + 18, "shared self-attention in every layer", size=FS_XS, color=SUB)
    F.text((vx + vw + ax) / 2, vy + vh + 33, "(actions attend to prefix, not vice versa)", size=FS_XS, color=SUB)
    # noisy action input
    F.tokens(ax + 22, ay + ah + 52, 8, GREY, s=12, gap=4)
    F.text(ax + aw / 2, ay + ah + 84, "noisy chunk $A^{τ}$ + flow time τ", size=FS_XS)
    F.arrow([(ax + aw / 2, ay + ah + 50), (ax + aw / 2, ay + ah + 2)])
    # language outputs above the VLM
    F.text(vx + 50, 36, "① subtask $\\hat{ℓ}$ (autoregressive)", size=FS_XS, color=GREEN[0], weight="bold")
    F.text(vx + 50, 52, "\"pick up the pillow\"", size=FS_XS, italic=True)
    F.tokens(vx + 8, 62, 6, GREEN, s=11, gap=3)
    F.arrow([(vx + 50, T), (vx + 50, 76)], color=GREEN[0])
    F.text(vx + 202, 36, "FAST tokens", size=FS_XS, color=PURPLE[0], weight="bold")
    F.text(vx + 202, 52, "(pre-training only)", size=FS_XS, color=SUB)
    F.tokens(vx + 160, 62, 6, PURPLE, s=11, gap=3, fills=[WHITE] * 6)
    F.arrow([(vx + 202, T), (vx + 202, 76)], color=PURPLE[0], dash="4 3", sw=1.2)
    F.arrow([(vx + 6, 67), (310, 67), (310, T + 12)], color=GREEN[0], sw=1.2, dash="4 3", radius=8)
    F.text(300, 90, "② $\\hat{ℓ}$ appended", size=FS_XS, color=GREEN[0], anchor="end")
    F.text(300, 105, "to the prefix", size=FS_XS, color=GREEN[0], anchor="end")
    # action outputs on the right
    ox = 800
    F.box(ox, T + 120, 185, 56, ORANGE, "② velocity $v_{θ}$", "Euler, 10 steps: $A^{τ+δ} = A^{τ} + δ v_{θ}$",
          title_size=FS_S, sub_size=FS_XS - 1)
    F.arrow([(ax + aw, T + 140), (ox, T + 140)], color=ORANGE[0])
    ph(F, ox, T + 200, 185, 80, ["action chunk $a_{t:t+H}$", "H = 50 (≈1 s @ 50 Hz)", "joint curves over time"])
    F.arrow([(ox + 92, T + 176), (ox + 92, T + 200)], color=ORANGE[0])
    F.arrow([(ox, T + 166), (786, T + 166), (786, T + ah + 58), (754, T + ah + 58)],
            color=ORANGE[0], sw=1.1, dash="3 3", radius=8)
    F.text(794, T + ah + 40, "repeat 10× (τ: 0 → 1)", size=FS_XS, color=ORANGE[0], anchor="start")
    # bottom: attention mask + recipe
    yb = 562
    F.panel(10, yb, 330, 160, title="Attention mask (blockwise)")
    ph(F, 30, yb + 35, 290, 92, ["3×3 block matrix", "[images+prompt+state | $\\hat{ℓ}$ tokens | action tokens]"])
    F.text(175, yb + 148, "■ attend  □ masked: prefix bidirectional, text causal", size=10.5, color=SUB)
    F.panel(350, yb, 640, 160, title="Training & inference")
    F.box(365, yb + 35, 300, 105, PURPLE, "Pre-training (VLM only)",
          ["robot data → FAST tokens + subtask text", "web data: captioning, VQA, detection",
           "loss: next-token cross-entropy"], title_size=FS_S, sub_size=FS_XS)
    F.box(675, yb + 35, 300, 105, ORANGE, "Post-training (+ action expert)",
          ["flow matching on continuous chunks", "$L = L_{CE} + α L_{FM}$",
           "inference: ① subtask → ② actions"], title_size=FS_S, sub_size=FS_XS)
    return F


# ====================================================================== Fig 3
def fig3():
    F = Fig(1000, 600)
    # timing strip
    F.panel(10, 10, 980, 112, title="Asynchronous execution (shared by all four designs)")
    y = 70
    F.text(80, y - 4, "slow", size=FS_S, anchor="end", color=PURPLE[0], weight="bold")
    F.text(80, y + 10, "1–10 Hz", size=FS_XS - 1, anchor="end", color=PURPLE[0])
    F.text(80, y + 32, "fast", size=FS_S, anchor="end", color=ORANGE[0], weight="bold")
    F.text(80, y + 46, "50–200 Hz", size=FS_XS - 1, anchor="end", color=ORANGE[0])
    for k in range(4):
        x0 = 90 + k * 220
        F.rect(x0, y - 18, 200, 18, rx=4, fill=PURPLE[1], stroke=PURPLE[0], sw=1)
        F.text(x0 + 100, y - 5, f"VLM inference  ($k$={k})" if k < 3 else "…", size=FS_XS)
    for j in range(43):
        x = 92 + j * 20
        F.rect(x, y + 22, 12, 18, rx=2, fill=ORANGE[1], stroke=ORANGE[0], sw=0.9)
    for k in range(3):
        xe = 290 + k * 220
        F.arrow([(xe, y + 2), (xe + 6, y + 20)], color=PURPLE[0], sw=1.2)
    F.text(500, y + 56, "fast steps reuse the latest slow output until the next one arrives", size=FS_XS, color=SUB)
    # four columns
    cols = [("a", "Slow latent → fast", "one / few latent vectors $z$", "e.g. Helix, HiRT, LCB",
             ["compact global intent", "low bandwidth, loose coupling"]),
            ("b", "Mid-layer tokens → fast", "token sequence from layer ℓ", "e.g. GR00T N1, RoboDual",
             ["rich spatial–semantic tokens", "fast policy cross-attends"]),
            ("c", "Shared parameters", "no explicit message", "e.g. FiS-VLA",
             ["fast path = last blocks of VLM", "reuses cached slow features"]),
            ("d", "Slow action chunk → fast", "coarse chunk $\\hat{A}$", "e.g. Hume, RoboDual",
             ["explicit motion prior", "fast system refines / tracks"])]
    for i, (lab, title, iface, eg, notes) in enumerate(cols):
        x = 10 + i * 247.5
        w = 237.5
        F.panel(x, 134, w, 456, label=lab, title=title, title_size=FS)
        cx = x + w / 2
        F.box(x + 20, 172, w - 40, 70, PURPLE, "Slow system (VLM)", "images + instruction, 1–10 Hz",
              title_size=FS_S, sub_size=FS_XS)
        # interface region
        F.rect(x + 20, 262, w - 40, 120, rx=8, fill=WHITE, stroke=PURPLE[0], sw=1.2, dash="5 4")
        F.text(cx, 280, "interface", size=FS_XS, color=PURPLE[0], weight="bold")
        ph(F, x + 34, 290, w - 68, 80, ["sketch of what is passed:", iface])
        F.arrow([(cx, 242), (cx, 262)], color=PURPLE[0], sw=2.6)
        F.arrow([(cx, 382), (cx, 402)], color=PURPLE[0], sw=2.6)
        F.box(x + 20, 402, w - 40, 70, ORANGE, "Fast system", "proprio + latest obs, 50–200 Hz",
              title_size=FS_S, sub_size=FS_XS)
        F.arrow([(cx, 472), (cx, 494)], color=ORANGE[0])
        F.text(cx, 508, "$a_{t}$", size=FS_S, color=ORANGE[0])
        F.lines(cx, 532, notes, size=FS_XS, color=SUB, lh=15)
        F.text(cx, 578, eg, size=FS_XS, color=INK, italic=True)
    return F


# ====================================================================== Fig 4
def fig4():
    F = Fig(1000, 800)
    F.panel(10, 10, 980, 330, label="a", title="Flow matching for action generation")
    ph(F, 25, 50, 300, 210, ["training: conditional paths", "noise ε ∼ N(0, I) (grey cloud, left)",
                            "action distribution (orange, bimodal, right)",
                            "straight lines $x_{τ} = (1-τ)ε + τA$"])
    ph(F, 345, 50, 300, 210, ["sampling: learned velocity field", "grey arrows = $v_{θ}(x, τ | o)$",
                             "curved ODE trajectories,", "Euler dots, N = 10 steps"])
    ph(F, 665, 50, 310, 210, ["same process on an action chunk", "chunk curves at τ = 0, 0.3, 0.7, 1",
                             "noise → smooth trajectory", "(x-axis: time step 1…H)"])
    F.rect(25, 272, 950, 56, rx=6, fill=WHITE, stroke=GRID, sw=1)
    F.text(500, 294, "train:  $\\min_{θ}\\, E_{τ,ε,A}\\, ‖v_{θ}(x_{τ}, τ \\mid o) - (A - ε)‖^{2}$".replace("\\min", "min").replace("\\,", " "),
           size=FS)
    F.text(500, 318, "sample:  $x_{0} = ε,\\;\\; x_{τ+δ} = x_{τ} + δ\\, v_{θ}(x_{τ}, τ \\mid o)$,  δ = 1/N".replace("\\;", " ").replace("\\,", " "),
           size=FS)
    cols = [("b", "Distillation: N steps → 1 step",
             ["2-D plane, same noise $x_{0}$", "teacher: curved path, 10 Euler dots",
              "student: one straight jump", "to teacher endpoint"],
             ["$L = ‖S_{φ}(x_{0}, o) - Φ_{T}^{(N)}(x_{0}, o)‖^{2}$", "NFE: N → 1, needs teacher"]),
            ("c", "One-step objective: mean velocity",
             ["true path with tangent $v$ (thin)", "chord = average velocity", "$u(x_{0}, 0, 1)$ (thick)",
              "one jump: ε → action"],
             ["$u(x_{τ}, τ, τ') = \\frac{1}{τ'-τ}∫ v\\,ds$".replace("\\frac{1}{τ'-τ}", "(τ'−τ)^{−1}").replace("\\,", " "),
              "$\\hat{A} = ε + u_{θ}(ε, 0, 1 \\mid o)$"]),
            ("d", "Warm start & cross-step iteration",
             ["loop over control cycles k", "shift $A_{k-1}$ → re-noise to $τ_{0}$",
              "K = 1–2 denoise steps with $o_{k}$", "execute s actions → next cycle"],
             ["$‖T_{o}(A) - T_{o}(A')‖ ≤ ρ‖A - A'‖$, ρ < 1",
              "holds only for a fixed observation $o$",
              "$o_{k}$ changes ⇒ fixed point $A^{*}(o_{k})$ moves"])]
    for i, (lab, title, phr, eqs) in enumerate(cols):
        x = 10 + i * 330
        F.panel(x, 352, 320, 438, label=lab, title=title, title_size=FS)
        ph(F, x + 14, 392, 292, 250, phr)
        if lab == "d":
            F.rect(x + 14, 654, 292, 120, rx=6, fill=WHITE, stroke=RED, sw=1.4, dash="4 3")
            F.lines(x + 160, 684, eqs, size=FS_S - 0.5, lh=26)
        else:
            F.rect(x + 14, 654, 292, 120, rx=6, fill=WHITE, stroke=GRID, sw=1)
            F.lines(x + 160, 688, eqs, size=FS_S, lh=30)
    return F


# ====================================================================== Fig 5
def fig5():
    F = Fig(1000, 600)
    heads = ["Sensor", "Raw data", "Encoder input", "Computation", "Properties"]
    xs = [25, 160, 362, 564, 776]
    ws = [115, 182, 182, 192, 200]
    for h, x, w in zip(heads, xs, ws):
        F.text(x + w / 2, 30, h, size=FS, weight="bold")
    rows = [("a", "Frame camera + RGB encoder", BLUE,
             ["RGB camera", "sync, 30–60 fps"],
             ["frames $I_{t}$ every 33 ms", "H×W×3, motion blur,", "static background repeated"],
             ["same frames", "(dense tensor)"],
             ["CNN / ViT", "dense MAC on every pixel,", "every frame"],
             ["temporal res.: frame rate", "latency ≥ frame interval", "ops: MAC, dense, GPU"]),
            ("b", "DVS + RGB (ANN) encoder", BLUE,
             ["event camera", "async per pixel, μs"],
             ["event stream $e_{i} = (x_{i}, y_{i}, t_{i}, p_{i})$", "x–y–t point cloud", "red/blue polarity"],
             ["binned to frame / voxel", "grid B×H×W over ΔT", "sparsity & timing lost"],
             ["CNN / ViT (as above)", "dense MAC incl. zeros"],
             ["temporal res.: bin ΔT", "latency ≥ window ΔT", "ops: MAC, dense, GPU"]),
            ("c", "DVS + SNN", TEAL,
             ["event camera", "async per pixel, μs"],
             ["same event stream", "as spike trains per pixel", "(raster)"],
             ["spikes fed directly", "(time steps T)"],
             ["LIF neurons: $V$ integrates,", "spike at $V ≥ θ$, reset", "AC ops only on spikes"],
             ["temporal res.: μs–ms", "event-driven, low latency", "ops: AC, sparse, neuromorphic"])]
    for r, (lab, title, c, *cells) in enumerate(rows):
        y = 48 + r * 180
        F.panel(10, y, 980, 170, fill=PANEL)
        F.text(20, y + 22, f"({lab}) {title}", size=FS, anchor="start", weight="bold", color=c[0])
        for k, (cell, x, w) in enumerate(zip(cells, xs, ws)):
            if k == 4:
                F.rect(x, y + 36, w, 120, rx=6, fill=WHITE, stroke=GRID, sw=1)
                F.lines(x + w / 2, y + 70, cell, size=FS_XS, color=INK, lh=22)
            else:
                ph(F, x, y + 36, w, 120, cell)
            if k < 4:
                F.arrow([(x + w + 2, y + 96), (xs[k + 1] - 2, y + 96)], sw=1.2)
        F.text(20, y + 60, "", size=1)
    F.text(500, 594, "", size=1)
    return F


if __name__ == "__main__":
    for name, fn in [("fig1_wire", fig1), ("fig2_wire", fig2), ("fig3_wire", fig3),
                     ("fig4_wire", fig4), ("fig5_wire", fig5)]:
        fn().save(f"out/wire/{name}.svg")
    print("ok")
