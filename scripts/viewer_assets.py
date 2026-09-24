"""Offline assets shared by synthetic and recorded skeleton viewers."""

from pathlib import Path


def vendored_scripts():
    folder = Path(__file__).resolve().parent / "vendor"
    return "\n".join(
        (folder / name).read_text(encoding="utf-8")
        for name in ("three.min.js", "OrbitControls.js")
    )


def geometry_script():
    return Path(__file__).with_name("viewer_geometry.js").read_text(encoding="utf-8")
