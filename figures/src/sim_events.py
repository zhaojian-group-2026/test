"""Simulate frames and DVS events for Fig. 5.

The camera pans slowly across a real background photo (assets/fig5_bg.png) while a ball is
thrown through the scene, so events fire on edges across the whole image (ego-motion) and
along the ball.  Sub-frames are rendered at 1 kHz; RGB frames integrate 8 ms of exposure;
events use the standard per-pixel log-intensity contrast-threshold model.
Outputs are cached in assets/fig5_sim.npz.
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.join(os.path.dirname(__file__), "..", "assets")
W, H = 320, 240
T_MS = 100                 # simulated duration
C = 0.25                   # contrast threshold (log units)
R = 14                     # ball radius in px
PAN = (-30.0, -10.0)       # scene shift in px over T_MS (camera pans right/down)
PAD = 34


def ball_pos(t_ms):
    u = t_ms / T_MS
    return 30 + 262 * u, 30 + 40 * u + 36 * u * u      # thrown left to right


def shifted(bgp, dx, dy):
    """Bilinear sub-pixel shift of the padded background, cropped to W x H."""
    x0, y0 = PAD - dx, PAD - dy
    ix, iy = int(np.floor(x0)), int(np.floor(y0))
    fx, fy = x0 - ix, y0 - iy
    a = bgp[iy:iy + H, ix:ix + W]
    b = bgp[iy:iy + H, ix + 1:ix + 1 + W]
    c = bgp[iy + 1:iy + 1 + H, ix:ix + W]
    d = bgp[iy + 1:iy + 1 + H, ix + 1:ix + 1 + W]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def render(bgp, t_ms):
    u = t_ms / T_MS
    img = shifted(bgp, PAN[0] * u, PAN[1] * u)
    cx, cy = ball_pos(t_ms)
    yy, xx = np.mgrid[0:H, 0:W]
    dd = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    a = np.clip(R + 0.5 - dd, 0, 1)[..., None]                    # anti-aliased disk
    shade = 1.0 + 1.6 * np.clip(1 - ((xx - cx + 4) ** 2 + (yy - cy + 4) ** 2) / (R * R), 0, 1)[..., None]
    ball = np.array([0.16, 0.17, 0.20])[None, None, :] * shade
    return img * (1 - a) + ball * a


def main():
    bg = Image.open(os.path.join(HERE, "fig5_bg.png")).convert("RGB").resize((W, H), Image.LANCZOS)
    bg = np.asarray(bg).astype(np.float32) / 255.0
    bgp = np.pad(bg, ((PAD + 1, PAD + 1), (PAD + 1, PAD + 1), (0, 0)), mode="edge")[1:, 1:]
    rng = np.random.default_rng(0)
    gray = lambda im: im @ np.array([0.299, 0.587, 0.114])  # noqa: E731
    ref = np.log(gray(render(bgp, 0)) + 0.02)
    ref += rng.uniform(-C / 2, C / 2, ref.shape)      # random initial phase, avoids banding
    ev = []
    for t in range(1, T_MS + 1):
        L = np.log(gray(render(bgp, t)) + 0.02)
        diff = L - ref
        for sgn in (1, -1):
            mask = sgn * diff >= C
            if mask.any():
                n = np.floor(sgn * diff[mask] / C).astype(int)
                ys, xs = np.nonzero(mask)
                for k in range(1, n.max() + 1):
                    sel = n >= k
                    tt = t - 1 + rng.random(sel.sum())          # sub-ms timestamps
                    ev.append(np.stack([xs[sel], ys[sel], tt, np.full(sel.sum(), sgn)], 1))
                ref[mask] += sgn * n * C
    nn = 400                                              # background activity noise
    ev.append(np.stack([rng.integers(0, W, nn), rng.integers(0, H, nn), rng.random(nn) * T_MS,
                        rng.choice([-1, 1], nn)], 1))
    ev = np.concatenate(ev).astype(np.float32)
    frames = {}
    for f, t0 in enumerate((0, 33, 66)):                  # 30 fps, 8 ms exposure
        frames[f"rgb{f}"] = np.mean([render(bgp, t0 + dt) for dt in np.linspace(0, 8, 9)], axis=0)
    np.savez_compressed(os.path.join(HERE, "fig5_sim.npz"), events=ev, **frames)
    print("events", len(ev), "per pixel per ms", round(len(ev) / (W * H * T_MS), 4))


if __name__ == "__main__":
    main()
