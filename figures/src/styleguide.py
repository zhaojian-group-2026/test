"""Style guide sheet: palette, typography, components, illustration samples."""
from figlib import *

F = Fig(1000, 560)
F.text(20, 32, "Unified visual style for Figs. 1–5", size=20, anchor="start", weight="bold")
F.text(20, 52, "All figures: 1000-unit canvas = 7.0 in (double column); Arial for text, Times italic for math; "
       "light fill + saturated stroke for modules.", size=FS_S, anchor="start", color=SUB)

# ---- palette
F.panel(15, 66, 470, 200, title="Semantic palette")
sw = [(BLUE, "Perception / vision", "images, cameras, encoders"),
      (PURPLE, "Slow system / VLM", "reasoning, planning"),
      (ORANGE, "Fast system / action", "action expert, execution"),
      (GREEN, "Language / state tokens", "prompt, proprioception"),
      (TEAL, "Spiking / neuromorphic", "SNN, LIF neurons"),
      (GREY, "Noise / neutral", "ε, inactive, background")]
for i, (c, name, note) in enumerate(sw):
    col, row = i % 2, i // 2
    x, y = 30 + col * 230, 106 + row * 48
    F.rect(x, y, 34, 34, rx=6, fill=c[1], stroke=c[0], sw=1.6)
    F.rect(x + 40, y + 5, 10, 24, rx=2, fill=c[0])
    F.text(x + 58, y + 15, name, size=FS_S, anchor="start", weight="bold")
    F.text(x + 58, y + 31, note, size=FS_XS, anchor="start", color=SUB)
F.circle(36, 252, 5, fill=EV_ON); F.circle(50, 252, 5, fill=EV_OFF)
F.text(62, 257, "event polarity ON / OFF", size=FS_XS, anchor="start", color=SUB)
F.rect(232, 246, 22, 12, rx=3, fill="none", stroke=RED, sw=1.6, dash="4 3")
F.text(262, 257, "caution / key constraint (red, dashed)", size=FS_XS, anchor="start", color=SUB)

# ---- typography
F.panel(500, 66, 485, 200, title="Typography")
F.text(516, 120, "(a) Panel title — 9 pt bold", size=FS_T, anchor="start", weight="bold")
F.text(516, 146, "Module label — 8 pt bold", size=FS, anchor="start", weight="bold")
F.text(516, 168, "Secondary text — 7 pt, grey", size=FS_S, anchor="start", color=SUB)
F.text(516, 188, "Annotation — 6.3 pt (minimum size)", size=FS_XS, anchor="start", color=SUB)
F.text(516, 222, "Math:  $x_{τ} = (1-τ)ε + τA$,   $v_{θ}(x_{τ}, τ \\mid o)$,   $a_{t:t+H}$",
       size=17, anchor="start")
F.text(516, 250, "Convention used everywhere: τ = 0 noise → τ = 1 action", size=FS_XS,
       anchor="start", color=SUB)

# ---- components
F.panel(15, 280, 470, 265, title="Components")
F.box(30, 320, 130, 50, PURPLE, "VLM backbone", "Gemma 2B")
F.box(175, 320, 130, 50, ORANGE, "Action expert", "300M, flow matching")
F.box(320, 320, 150, 50, BLUE, "Vision encoder", "SigLIP")
y = 392
for i, (c, lab) in enumerate([(BLUE, "image tokens"), (GREEN, "text / state tokens"),
                              (ORANGE, "action tokens"), (GREY, "noisy action tokens")]):
    xx = 30 + (i % 2) * 230
    yy = y + (i // 2) * 24
    F.tokens(xx, yy, 6, c, s=11, gap=3)
    F.text(xx + 90, yy + 10, lab, size=FS_XS, anchor="start", color=SUB)
F.arrow([(30, 462), (130, 462)])
F.text(140, 466, "data flow", size=FS_XS, anchor="start", color=SUB)
F.arrow([(30, 484), (130, 484)], dash="5 4", color=SUB)
F.text(140, 488, "feedback / optional", size=FS_XS, anchor="start", color=SUB)
F.arrow([(260, 462), (360, 462)], color=PURPLE[0], sw=3)
F.text(370, 466, "interface (thick)", size=FS_XS, anchor="start", color=SUB)
F.arrow([(260, 484), (360, 484)], color=ORANGE[0], sw=1.6, dash="2 3")
F.text(370, 488, "planned motion", size=FS_XS, anchor="start", color=SUB)
for i, c in enumerate([BLUE, PURPLE, ORANGE]):
    F.badge(40 + i * 24, 520, str(i + 1), c)
F.text(110, 524, "stage badges: perception · planning · execution", size=FS_XS, anchor="start", color=SUB)

# ---- illustration samples
F.panel(500, 280, 485, 265, title="Illustration style (flat, side view)")
floor = 505
F.line(515, floor + 22, 970, floor + 22, stroke=GRID, sw=1.2)
# conveyor
F.rect(520, 455, 170, 12, rx=6, fill="#B9C0CB", stroke=ARM_DARK, sw=1.1)
for cx in (526, 684):
    F.circle(cx, 461, 5, fill="#7B8494", stroke=ARM_DARK, sw=0.9)
for lx in (540, 670):
    F.line(lx, 467, lx, floor + 22, stroke=ARM_DARK, sw=2.4, cap="butt")
F.rect(545, 433, 22, 22, rx=3, fill=BLUE[1], stroke=BLUE[0], sw=1.2)
F.rect(600, 433, 22, 22, rx=3, fill=ORANGE[1], stroke=ORANGE[0], sw=1.2)
F.rect(596, 429, 30, 30, rx=4, fill="none", stroke=BLUE[0], sw=1.1, dash="3 2")
F.arrow([(598, 420), (628, 420)], color=BLUE[0], sw=1.4)
F.text(590, 424, "$\\hat{v}$", size=FS_S, color=BLUE[0])
F.rect(650, 433, 22, 22, rx=3, fill="none", stroke=ORANGE[0], sw=1.2, dash="3 2")
camera(F, 585, 335, ang=-90)
fov(F, 585, 350, -90, half=20, length=80)
base = (780, 452)
L = (72, 66, 22)
ang = arm_ik(base, (661, 426), -90, L)
robot_arm(F, base, ang, L, ped_h=floor + 22 - 458)
# catching sample
camera(F, 815, 335, ang=-15, kind="event", color=TEAL)
F.text(815, 362, "event cam", size=FS_XS, color=SUB)
xs = [850, 868, 886, 904]
for i, x in enumerate(xs):
    yy = 330 + 0.012 * (x - 930) ** 2 * -1 + 60
    F.circle(x, yy, 3.2, fill=BLUE[0], opacity=0.35 + 0.15 * i)
F.circle(922, 390 - 0.012 * 64, 6, fill="#E9B949", stroke="#9A7419", sw=1)
F.path("M928,389 Q950,392 958,420", stroke=PURPLE[0], sw=1.5, dash="4 3")
F.circle(958, 424, 7, fill="none", stroke=PURPLE[0], sw=1.4)
F.text(958, 446, "$p^{*}, t^{*}$", size=FS_XS, color=PURPLE[0])
F.save("out/wire/style_guide.svg")
print("ok")
