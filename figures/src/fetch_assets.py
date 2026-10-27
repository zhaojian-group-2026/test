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
