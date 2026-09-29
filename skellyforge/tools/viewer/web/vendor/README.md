# Vendored viewer libraries

`generate_skeleton_viewer.py` inlines these into the HTML it writes, so the viewer opens
with no network and renders correctly when served from a strict content security policy.
They are committed rather than fetched at build time so that regenerating the viewer never
depends on being online.

| file | source | version |
|---|---|---|
| `three.min.js` | [three.js](https://github.com/mrdoob/three.js) `build/three.min.js` | r128 |
| `OrbitControls.js` | three.js `examples/js/controls/OrbitControls.js` | r128 |

Both are MIT licensed, copyright the three.js authors.

**On the r128 pin.** `examples/js/` holds the pre-module builds, which three.js removed
after r128 - the same files live under `examples/jsm/` as ES modules from r129 on. Moving
off r128 therefore means moving the viewer to `<script type="module">` and an import map,
not just changing a version number here. Fetch a replacement with
`npm pack three@<version>` and copy the two files in.
# Plotly

`plotly-basic-2.35.2.min.js` is Plotly's official basic distribution, downloaded
from https://cdn.plot.ly/plotly-basic-2.35.2.min.js . MIT license is retained in
`PLOTLY_LICENSE`. SHA256:
`138c2e81014b979dc00867a93da55b7605a17495ee78dd7afb433b7f021dfcfa`.
It is bundled into the solver viewer for offline interactive time-series plots.
