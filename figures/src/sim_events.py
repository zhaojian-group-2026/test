"""Simulate frames and DVS events for Fig. 5.

A synthetic ball moves fast across a real background photo (assets/fig5_bg.png).
Sub-frames are rendered at 1 kHz; RGB frames integrate 8 ms of exposure (motion blur);
events use the standard per-pixel log-intensity contrast-threshold model.
Outputs are cached in assets/fig5_sim.npz.
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.join(os.path.dirname(__file__), "..", "assets")
W, H = 320, 240
T_MS = 100                 # simulated duration
C = 0.18                   # contrast threshold (log units)
R = 16                     # ball radius in px


def ball_pos(t_ms):
    u = t_ms / T_MS
    return 30 + 262 * u, 30 + 40 * u + 36 * u * u      # thrown left to right over the bright wall


def render(bg, t_ms):
    img = bg.copy()
    cx, cy = ball_pos(t_ms)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    a = np.clip(R + 0.5 - d, 0, 1)[..., None]                    # anti-aliased disk
    shade = 1.0 + 1.6 * np.clip(1 - ((xx - cx + 4) ** 2 + (yy - cy + 4) ** 2) / (R * R), 0, 1)[..., None]
    ball = np.array([0.16, 0.17, 0.20])[None, None, :] * shade
    return img * (1 - a) + ball * a


def main():
    bg = Image.open(os.path.join(HERE, "fig5_bg.png")).convert("RGB").resize((W, H), Image.LANCZOS)
    bg = np.asarray(bg).astype(np.float32) / 255.0
    rng = np.random.default_rng(0)
    frames = {}
    ev = []
    gray = lambda im: im @ np.array([0.299, 0.587, 0.114])  # noqa: E731
    ref = np.log(gray(render(bg, 0)) + 0.02)
    for t in range(0, T_MS + 1):
        im = render(bg, t)
        L = np.log(gray(im) + 0.02)
        diff = L - ref
        for sgn in (1, -1):
            mask = sgn * diff >= C
            if mask.any():
                n = np.floor(sgn * diff[mask] / C).astype(int)
                ys, xs = np.nonzero(mask)
                for k in range(1, n.max() + 1):
                    sel = n >= k
                    tt = t + rng.random(sel.sum())          # sub-ms timestamps
                    ev.append(np.stack([xs[sel], ys[sel], tt, np.full(sel.sum(), sgn)], 1))
                ref[mask] += sgn * n * C
    # background activity noise
    nn = 260
    ev.append(np.stack([rng.integers(0, W, nn), rng.integers(0, H, nn), rng.random(nn) * T_MS,
                        rng.choice([-1, 1], nn)], 1))
    ev = np.concatenate(ev).astype(np.float32)
    # RGB frames at 30 fps with 10 ms exposure
    for f, t0 in enumerate((0, 33, 66)):
        frames[f"rgb{f}"] = np.mean([render(bg, t0 + dt) for dt in np.linspace(0, 8, 9)], axis=0)
    np.savez_compressed(os.path.join(HERE, "fig5_sim.npz"), events=ev, **frames)
    print("events", len(ev), "density per pixel per ms", len(ev) / (W * H * T_MS))


if __name__ == "__main__":
    main()
