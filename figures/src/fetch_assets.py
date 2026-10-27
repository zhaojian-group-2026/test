"""Download third-party photos used as placeholders (not committed; see REFERENCES.md).
Replace any file in assets/ with your own photo of the same name to swap it in."""
import os
import urllib.request

ASSETS = {
    # AHEAD (Syed et al., 2026, arXiv:2606.02486), Fig. 1
    "fig1_conveyor.png": "https://arxiv.org/html/2606.02486v2/images/hero_conveyor_ducky.png",
    "fig1_catch.png": "https://arxiv.org/html/2606.02486v2/images/hero_catch.png",
    "fig1_pingpong.png": "https://arxiv.org/html/2606.02486v2/images/hero_pingpong_hit.png",
}

here = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(here, exist_ok=True)
for name, url in ASSETS.items():
    dst = os.path.join(here, name)
    if os.path.exists(dst):
        continue
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as fh:
        fh.write(r.read())
    print("fetched", name)


# pi0.5 (Physical Intelligence et al., 2025, arXiv:2504.16054), Fig. 5: the "4x images" camera strip
def _pi05_cams():
    import io
    import numpy as np
    from PIL import Image
    if all(os.path.exists(os.path.join(here, f"fig2_cam{k}.png")) for k in range(4)):
        return
    url = "https://arxiv.org/html/2504.16054v1/robot_system_overview.png"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    a = np.asarray(im).astype(int)
    sub = a[:, :400]
    yel = (sub[:, :, 0] > 220) & (sub[:, :, 1] > 170) & (sub[:, :, 2] < 140)
    ys, xs = np.where(yel)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    strip = a[y0:y1, x0:x1]
    inner = ~((strip[:, :, 0] > 220) & (strip[:, :, 1] > 170) & (strip[:, :, 2] < 140))
    rows = np.where(inner[:, 20:-20].mean(axis=1) > 0.8)[0]
    cols = np.where(inner[20:-20, :].mean(axis=0) > 0.5)[0]
    groups = [g for g in np.split(rows, np.where(np.diff(rows) > 3)[0] + 1) if len(g) >= 30]
    for k, g in enumerate(groups[:4]):
        im.crop((x0 + cols.min() + 3, y0 + g[0] + 3, x0 + cols.max() - 3, y0 + g[-1] - 3)).save(
            os.path.join(here, f"fig2_cam{k}.png"))
    print("cropped pi0.5 camera views")


_pi05_cams()
