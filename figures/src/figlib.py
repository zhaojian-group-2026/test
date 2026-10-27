"""Shared style + SVG primitives for all paper figures.

Coordinate system: every figure is 1000 units wide, printed at 7.0 in
(double-column width), so 1 unit = 0.504 pt.  Font size 16 -> ~8 pt.
"""
import math
from xml.sax.saxutils import escape

# ---------------------------------------------------------------- style
SANS = "Arial, Helvetica, 'Liberation Sans', sans-serif"
SERIF = "'Times New Roman', Times, 'Liberation Serif', serif"

INK = "#1F2430"      # main text
SUB = "#5B6170"      # secondary text
LINE = "#3F4652"     # arrows / connectors
GRID = "#C9CED6"     # light borders
PANEL = "#F6F7F9"    # panel background
WHITE = "#FFFFFF"
RED = "#C8423B"      # caution / emphasis

# semantic colour pairs: (stroke, light fill)
BLUE = ("#2F6DB5", "#E4EEF9")     # perception / vision
PURPLE = ("#6C4FB8", "#EEE9F8")   # slow system / VLM / planning
ORANGE = ("#D9772B", "#FCEEDF")   # action / fast system / execution
GREEN = ("#3B9A5F", "#E4F3E9")    # language / state tokens
TEAL = ("#178A87", "#DFF2F1")     # spiking / neuromorphic
GREY = ("#8A919E", "#EEF0F3")     # noise / neutral
EV_ON = "#D6453D"                 # event polarity +
EV_OFF = "#2F5FC4"                # event polarity -

FS = 16        # body label
FS_S = 14      # small label
FS_XS = 12.5   # annotation
FS_T = 18      # panel title
SW = 1.3       # box stroke
ASW = 1.5      # arrow stroke


def f(v):
    """compact number formatting"""
    if isinstance(v, float):
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return s if s not in ("-0", "") else "0"
    return str(v)


# ---------------------------------------------------------------- rich text
import re

_MACROS = {r"\mid": "|", r"\cdot": "·", r"\to": "→", r"\approx": "≈", r"\leq": "≤",
           r"\geq": "≥", r"\infty": "∞", r"\int": "∫", r"\|": "‖", r"\times": "×",
           r"\sim": "∼", r"\nabla": "∇", r"\neq": "≠", r"\in": "∈"}


def _macros(s):
    s = re.sub(r"\\hat\{(.)\}", lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\bar\{(.)\}", lambda m: m.group(1) + "\u0304", s)
    s = re.sub(r"\\tilde\{(.)\}", lambda m: m.group(1) + "\u0303", s)
    s = re.sub(r"\\dot\{(.)\}", lambda m: m.group(1) + "\u0307", s)
    for k in sorted(_MACROS, key=len, reverse=True):
        s = s.replace(k, _MACROS[k])
    return s


def _parse(s, math_mode=False, shift=0, upright=False, bold=False, out=None):
    """Tiny markup: $..$ math, _{..} sub, ^{..} sup, \\rm{..}, \\bf{..}, \\hat{x}."""
    if out is None:
        out = []
        s = _macros(s)
    i = 0
    n = len(s)

    def read_group(j):
        depth = 0
        k = j
        while k < n:
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0:
                    return s[j + 1:k], k + 1
            k += 1
        raise ValueError("unbalanced braces in " + s)

    while i < n:
        c = s[i]
        if c == "$":
            math_mode = not math_mode
            i += 1
            continue
        if c in "_^" and i + 1 < n and s[i + 1] == "{":
            grp, i = read_group(i + 1)
            _parse(grp, math_mode, shift + (1 if c == "^" else -1), upright, bold, out)
            continue
        if s.startswith("\\rm{", i):
            grp, i = read_group(i + 3)
            _parse(grp, math_mode, shift, True, bold, out)
            continue
        if s.startswith("\\bf{", i):
            grp, i = read_group(i + 3)
            _parse(grp, math_mode, shift, upright, True, out)
            continue
        if math_mode:
            fam = "serif"
            ital = (c.isalpha() or c in "̂̄̃̇") and not upright
        else:
            fam = "sans"
            ital = False
        out.append([c, fam, ital, bold, shift])
        i += 1
    # merge runs
    merged = []
    for seg in out:
        if merged and merged[-1][1:] == seg[1:]:
            merged[-1][0] += seg[0]
        else:
            merged.append(list(seg))
    return merged


def text(x, y, s, size=FS, anchor="middle", color=INK, weight="normal",
         italic=False, family=None, opacity=None, rotate=None, spacing=None):
    """Return an SVG <text> element with rich-text tspans."""
    segs = _parse(s)
    attrs = [f'x="{f(x)}"', f'y="{f(y)}"', f'font-size="{f(size)}"',
             f'text-anchor="{anchor}"', f'fill="{color}"',
             f'font-family="{family or SANS}"', 'xml:space="preserve"']
    if weight != "normal":
        attrs.append(f'font-weight="{weight}"')
    if italic:
        attrs.append('font-style="italic"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    if spacing is not None:
        attrs.append(f'letter-spacing="{spacing}"')
    if rotate is not None:
        attrs.append(f'transform="rotate({f(rotate)} {f(x)} {f(y)})"')
    cur = 0.0
    parts = []
    for txt, fam, ital, bold, sh in segs:
        fs = size * (0.7 if sh else 1.0) * (0.85 if abs(sh) > 1 else 1.0)
        target = 0.0 if sh == 0 else (size * 0.26 if sh < 0 else -size * 0.36) * (1.4 if abs(sh) > 1 else 1)
        dy = target - cur
        cur = target
        a = []
        if abs(dy) > 1e-6:
            a.append(f'dy="{f(dy)}"')
        if fs != size:
            a.append(f'font-size="{f(fs)}"')
        if fam == "serif":
            a.append(f'font-family="{SERIF}"')
        if ital:
            a.append('font-style="italic"')
        if bold:
            a.append('font-weight="bold"')
        if a:
            parts.append(f'<tspan {" ".join(a)}>{escape(txt)}</tspan>')
        else:
            parts.append(escape(txt))
    if abs(cur) > 1e-6:  # keep later siblings on baseline (no-op visually)
        parts.append(f'<tspan dy="{f(-cur)}"></tspan>')
    return f'<text {" ".join(attrs)}>{"".join(parts)}</text>'


def est_width(s, size=FS):
    """Rough width estimate of rendered rich text."""
    w = 0.0
    for txt, fam, ital, bold, sh in _parse(s):
        fs = size * (0.7 if sh else 1.0)
        for ch in txt:
            if ch in "il.,:;|!'()[]{} ":
                k = 0.30
            elif ch in "mwMW":
                k = 0.85
            elif ch.isupper():
                k = 0.66
            else:
                k = 0.53
            w += k * fs * (1.08 if bold else 1.0)
    return w


# ---------------------------------------------------------------- figure
class Fig:
    def __init__(self, w=1000, h=600, bg=WHITE):
        self.w, self.h, self.bg = w, h, bg
        self.items = []
        self.defs = []

    def add(self, s):
        self.items.append(s)
        return s

    # -- groups
    def g(self, transform=None, opacity=None, id=None):
        a = []
        if id:
            a.append(f'id="{id}"')
        if transform:
            a.append(f'transform="{transform}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        self.add(f'<g {" ".join(a)}>')

    def end(self):
        self.add("</g>")

    # -- shapes
    def rect(self, x, y, w, h, rx=6, fill="none", stroke=None, sw=SW, dash=None,
             opacity=None, fill_opacity=None):
        a = [f'x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}"']
        if rx:
            a.append(f'rx="{f(rx)}"')
        a.append(f'fill="{fill}"')
        if fill_opacity is not None:
            a.append(f'fill-opacity="{fill_opacity}"')
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{f(sw)}"')
            if dash:
                a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        return self.add(f'<rect {" ".join(a)}/>')

    def circle(self, cx, cy, r, fill="none", stroke=None, sw=SW, dash=None,
               opacity=None, fill_opacity=None):
        a = [f'cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="{fill}"']
        if fill_opacity is not None:
            a.append(f'fill-opacity="{fill_opacity}"')
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{f(sw)}"')
            if dash:
                a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        return self.add(f'<circle {" ".join(a)}/>')

    def ellipse(self, cx, cy, rx, ry, fill="none", stroke=None, sw=SW, opacity=None,
                dash=None, rotate=None):
        a = [f'cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx)}" ry="{f(ry)}" fill="{fill}"']
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{f(sw)}"')
            if dash:
                a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        if rotate is not None:
            a.append(f'transform="rotate({f(rotate)} {f(cx)} {f(cy)})"')
        return self.add(f'<ellipse {" ".join(a)}/>')

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=ASW, dash=None, cap="round",
             opacity=None):
        a = [f'x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}"',
             f'stroke="{stroke}" stroke-width="{f(sw)}" stroke-linecap="{cap}"']
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        return self.add(f'<line {" ".join(a)}/>')

    def path(self, d, fill="none", stroke=LINE, sw=ASW, dash=None, cap="round",
             join="round", opacity=None, fill_opacity=None):
        a = [f'd="{d}" fill="{fill}"']
        if fill_opacity is not None:
            a.append(f'fill-opacity="{fill_opacity}"')
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{f(sw)}" '
                     f'stroke-linecap="{cap}" stroke-linejoin="{join}"')
            if dash:
                a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        return self.add(f'<path {" ".join(a)}/>')

    def poly(self, pts, fill="none", stroke=None, sw=SW, closed=True, opacity=None,
             join="round", dash=None, fill_opacity=None):
        p = " ".join(f"{f(x)},{f(y)}" for x, y in pts)
        tag = "polygon" if closed else "polyline"
        a = [f'points="{p}" fill="{fill}"']
        if fill_opacity is not None:
            a.append(f'fill-opacity="{fill_opacity}"')
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{f(sw)}" stroke-linejoin="{join}"')
            if dash:
                a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'opacity="{opacity}"')
        return self.add(f'<{tag} {" ".join(a)}/>')

    def text(self, x, y, s, **kw):
        return self.add(text(x, y, s, **kw))

    def lines(self, x, y, rows, size=FS, lh=None, **kw):
        """Stack of text rows; y is the first baseline."""
        lh = lh or size * 1.25
        for i, r in enumerate(rows):
            if r:
                self.text(x, y + i * lh, r, size=size, **kw)

    # -- composite helpers
    def box(self, x, y, w, h, color=GREY, title=None, sub=None, rx=7, sw=SW,
            title_size=FS, sub_size=FS_S, dash=None, fill=None, bold=True,
            title_color=None, sub_color=SUB, valign=0.0):
        stroke, light = color
        self.rect(x, y, w, h, rx=rx, fill=fill or light, stroke=stroke, sw=sw, dash=dash)
        rows = []
        if title:
            rows += [(t, title_size, "bold" if bold else "normal", title_color or INK)
                     for t in (title if isinstance(title, list) else [title])]
        if sub:
            rows += [(t, sub_size, "normal", sub_color)
                     for t in (sub if isinstance(sub, list) else [sub])]
        if rows:
            total = sum(r[1] * 1.22 for r in rows)
            yy = y + h / 2 - total / 2 + rows[0][1] * 0.86 + valign
            for t, s, wgt, col in rows:
                self.text(x + w / 2, yy, t, size=s, weight=wgt, color=col)
                yy += s * 1.22

    def arrowhead(self, tip, frm, color=LINE, sw=ASW, size=None):
        size = size or (5.2 + 2.2 * sw)
        ang = math.atan2(tip[1] - frm[1], tip[0] - frm[0])
        w = size * 0.48
        bx, by = tip[0] - size * math.cos(ang), tip[1] - size * math.sin(ang)
        p1 = (bx + w * math.sin(ang), by - w * math.cos(ang))
        p2 = (bx - w * math.sin(ang), by + w * math.cos(ang))
        self.poly([tip, p1, p2], fill=color)
        return size

    def arrow(self, pts, color=LINE, sw=ASW, dash=None, head=True, tail=False,
              radius=0, opacity=None, hsize=None):
        """Polyline arrow (optionally with rounded corners)."""
        pts = [tuple(p) for p in pts]
        hs = hsize or (5.2 + 2.2 * sw)
        a, b = pts[-2], pts[-1]
        L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
        end = (b[0] - (b[0] - a[0]) / L * hs * 0.85, b[1] - (b[1] - a[1]) / L * hs * 0.85) if head else b
        a0, b0 = pts[1], pts[0]
        L0 = math.hypot(b0[0] - a0[0], b0[1] - a0[1]) or 1
        start = (b0[0] - (b0[0] - a0[0]) / L0 * hs * 0.85, b0[1] - (b0[1] - a0[1]) / L0 * hs * 0.85) if tail else b0
        body = [start] + pts[1:-1] + [end]
        if radius and len(body) > 2:
            d = f"M{f(body[0][0])},{f(body[0][1])}"
            for i in range(1, len(body) - 1):
                p0, p1, p2 = body[i - 1], body[i], body[i + 1]
                l1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                l2 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                r = min(radius, l1 / 2, l2 / 2)
                q0 = (p1[0] - (p1[0] - p0[0]) / l1 * r, p1[1] - (p1[1] - p0[1]) / l1 * r)
                q2 = (p1[0] + (p2[0] - p1[0]) / l2 * r, p1[1] + (p2[1] - p1[1]) / l2 * r)
                d += f" L{f(q0[0])},{f(q0[1])} Q{f(p1[0])},{f(p1[1])} {f(q2[0])},{f(q2[1])}"
            d += f" L{f(body[-1][0])},{f(body[-1][1])}"
        else:
            d = "M" + " L".join(f"{f(x)},{f(y)}" for x, y in body)
        self.path(d, stroke=color, sw=sw, dash=dash, opacity=opacity)
        if opacity is not None:
            self.g(opacity=opacity)
        if head:
            self.arrowhead(b, a, color, sw, hs)
        if tail:
            self.arrowhead(b0, a0, color, sw, hs)
        if opacity is not None:
            self.end()

    def curve(self, p0, c1, c2, p1, color=LINE, sw=ASW, dash=None, head=True,
              opacity=None, hsize=None, tail=False):
        """Cubic Bezier arrow."""
        hs = hsize or (5.2 + 2.2 * sw)
        L = math.hypot(p1[0] - c2[0], p1[1] - c2[1]) or 1
        end = (p1[0] - (p1[0] - c2[0]) / L * hs * 0.85, p1[1] - (p1[1] - c2[1]) / L * hs * 0.85) if head else p1
        L0 = math.hypot(p0[0] - c1[0], p0[1] - c1[1]) or 1
        st = (p0[0] - (p0[0] - c1[0]) / L0 * hs * 0.85, p0[1] - (p0[1] - c1[1]) / L0 * hs * 0.85) if tail else p0
        d = (f"M{f(st[0])},{f(st[1])} C{f(c1[0])},{f(c1[1])} "
             f"{f(c2[0])},{f(c2[1])} {f(end[0])},{f(end[1])}")
        self.path(d, stroke=color, sw=sw, dash=dash, opacity=opacity)
        if head:
            self.arrowhead(p1, c2, color, sw, hs)
        if tail:
            self.arrowhead(p0, c1, color, sw, hs)

    def badge(self, cx, cy, label, color, r=8.5, size=12):
        self.circle(cx, cy, r, fill=color[0])
        self.text(cx, cy + size * 0.36, label, size=size, color=WHITE, weight="bold")

    def tokens(self, x, y, n, color, s=11, gap=3, vertical=False, fills=None,
               sw=1.0, rx=2):
        """Row (or column) of token squares; returns end coordinate."""
        for i in range(n):
            xx = x + (0 if vertical else i * (s + gap))
            yy = y + (i * (s + gap) if vertical else 0)
            fill = fills[i] if fills else color[1]
            self.rect(xx, yy, s, s, rx=rx, fill=fill, stroke=color[0], sw=sw)
        return (x + (s if vertical else n * (s + gap) - gap),
                y + (n * (s + gap) - gap if vertical else s))

    def panel(self, x, y, w, h, label=None, title=None, fill=PANEL, rx=10,
              stroke=None, title_size=FS_T, pad=14):
        self.rect(x, y, w, h, rx=rx, fill=fill, stroke=stroke, sw=1.0)
        if title or label:
            s = (f"({label}) " if label else "") + (title or "")
            self.text(x + pad, y + pad + title_size * 0.62, s, size=title_size,
                      anchor="start", weight="bold")

    # -- output
    def svg(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {f(self.w)} {f(self.h)}" width="{f(self.w)}" height="{f(self.h)}">')
        bg = f'<rect x="0" y="0" width="{f(self.w)}" height="{f(self.h)}" fill="{self.bg}"/>'
        defs = f"<defs>{''.join(self.defs)}</defs>" if self.defs else ""
        return "\n".join([head, defs, bg] + self.items + ["</svg>"])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.svg())


# ---------------------------------------------------------------- illustrations
ARM_DARK = "#3A404C"
ARM_LIGHT = "#D9DEE5"
ARM_JOINT = "#5E6776"


def arm_fk(base, angles, lengths):
    """2-D forward kinematics; angles are relative (deg, CCW, y-up)."""
    pts = [base]
    phi = 0.0
    x, y = base
    for th, L in zip(angles, lengths):
        phi += th
        x += L * math.cos(math.radians(phi))
        y -= L * math.sin(math.radians(phi))
        pts.append((x, y))
    return pts, phi


def arm_ik(base, ee, phi_ee, lengths, elbow="up"):
    """Relative joint angles placing the tool tip at `ee` with heading phi_ee."""
    L1, L2, L3 = lengths
    wx = ee[0] - L3 * math.cos(math.radians(phi_ee))
    wy = ee[1] + L3 * math.sin(math.radians(phi_ee))
    dx, dy = wx - base[0], base[1] - wy  # y-up
    d = min(math.hypot(dx, dy), L1 + L2 - 1e-6)
    a0 = math.degrees(math.atan2(dy, dx))
    alpha = math.degrees(math.acos((L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d)))
    beta = math.degrees(math.acos((L1 ** 2 + L2 ** 2 - d ** 2) / (2 * L1 * L2)))
    sols = []
    for sgn in (1, -1):
        t1 = a0 + sgn * alpha
        t2 = -sgn * (180 - beta)
        sols.append((t1, t2))
    # elbow height (y-up) for each solution
    def elbow_y(t1):
        return math.sin(math.radians(t1))
    t1, t2 = max(sols, key=lambda s: elbow_y(s[0])) if elbow == "up" else min(sols, key=lambda s: elbow_y(s[0]))
    t3 = phi_ee - (t1 + t2)
    return [t1, t2, t3]


def robot_arm(F, base, angles, lengths, width=13, opacity=None, grip=7,
              tool="gripper", accent=None, pedestal=True, ped_h=24):
    """Side-view stylised manipulator.  Returns (joint points, ee heading)."""
    pts, phi = arm_fk(base, angles, lengths)
    if opacity is not None:
        F.g(opacity=opacity)
    bx, by = base
    if pedestal:
        F.rect(bx - 15, by + 6, 30, ped_h, rx=3, fill="#A3ABB8", stroke=ARM_DARK, sw=1.1)
        F.poly([(bx - 16, by + 7), (bx + 16, by + 7), (bx + 11, by - 6), (bx - 11, by - 6)],
               fill="#A3ABB8", stroke=ARM_DARK, sw=1.1)
    d = "M" + " L".join(f"{f(x)},{f(y)}" for x, y in pts)
    F.path(d, stroke=ARM_DARK, sw=width, cap="round")
    F.path(d, stroke=ARM_LIGHT, sw=width - 3.2, cap="round")
    for i, (x, y) in enumerate(pts[:-1]):
        r = width * 0.62 if i < len(pts) - 2 else width * 0.5
        F.circle(x, y, r, fill=ARM_JOINT, stroke=ARM_DARK, sw=1.0)
        F.circle(x, y, r * 0.42, fill=accent or "#C5CBD4")
    # tool
    ex, ey = pts[-1]
    c, s = math.cos(math.radians(phi)), -math.sin(math.radians(phi))
    nx, ny = -s, c  # normal
    if tool == "gripper":
        palm = 11
        F.line(ex + nx * palm, ey + ny * palm, ex - nx * palm, ey - ny * palm,
               stroke=ARM_DARK, sw=5, cap="round")
        for sgn in (1, -1):
            ox, oy = ex + nx * grip * sgn, ey + ny * grip * sgn
            F.line(ox, oy, ox + c * 13, oy + s * 13, stroke=ARM_DARK, sw=3.6, cap="round")
    elif tool == "paddle":
        hx, hy = ex + c * 8, ey + s * 8
        F.line(ex, ey, hx, hy, stroke="#7A4A2A", sw=4.5)
        F.ellipse(hx + c * 13, hy + s * 13, 15, 12, fill="#C8423B", stroke="#7E2420", sw=1.2,
                  rotate=math.degrees(math.atan2(s, c)))
    elif tool == "cup":
        F.line(ex + nx * 10, ey + ny * 10, ex - nx * 10, ey - ny * 10,
               stroke=ARM_DARK, sw=5, cap="round")
        # open basket
        p = [(ex + nx * 12, ey + ny * 12), (ex + nx * 16 + c * 16, ey + ny * 16 + s * 16),
             (ex - nx * 16 + c * 16, ey - ny * 16 + s * 16), (ex - nx * 12, ey - ny * 12)]
        F.poly(p, fill="#F3D2B5", stroke=ARM_DARK, sw=1.4, closed=False)
    if opacity is not None:
        F.end()
    return pts, phi


def camera(F, x, y, ang=0, scale=1.0, color=BLUE, kind="rgb"):
    """Small camera icon centred at (x,y) looking along `ang` (deg, y-up)."""
    F.g(transform=f"translate({f(x)},{f(y)}) rotate({f(-ang)}) scale({f(scale)})")
    body = "#2E3440" if kind == "rgb" else "#1D4E4C"
    F.rect(-14, -9, 22, 18, rx=3, fill=body)
    F.rect(8, -6, 7, 12, rx=1.5, fill=body)
    F.circle(15, 0, 4.2, fill=color[0], stroke="#11151C", sw=1)
    F.circle(15, 0, 1.6, fill="#DCE8F6")
    if kind == "event":
        F.rect(-11, -6, 8, 4, rx=1, fill=EV_ON)
        F.rect(-11, 1, 8, 4, rx=1, fill="#6E9BFF")
    F.end()


def fov(F, x, y, ang, half=22, length=120, color=BLUE, opacity=0.18):
    a1, a2 = math.radians(ang - half), math.radians(ang + half)
    p1 = (x + length * math.cos(a1), y - length * math.sin(a1))
    p2 = (x + length * math.cos(a2), y - length * math.sin(a2))
    F.poly([(x, y), p1, p2], fill=color[0], fill_opacity=opacity)
    F.line(x, y, *p1, stroke=color[0], sw=0.8, dash="3 3", opacity=0.7)
    F.line(x, y, *p2, stroke=color[0], sw=0.8, dash="3 3", opacity=0.7)
