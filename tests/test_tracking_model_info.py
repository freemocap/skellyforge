from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

from skellyforge.skellymodels.models.tracking_model_info import ModelInfo


def test_model_info_uses_utf8_on_a_gbk_default_system(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config_path = tmp_path / "模型定义.yaml"
    config_path.write_text(
        "# 中文注释用于模拟 Windows 简体中文环境\n"
        "name: minimal\n"
        "tracker_name: minimal\n"
        "order: [body]\n"
        "aspects:\n"
        "  body:\n"
        "    tracked_points:\n"
        "      type: list\n"
        "      names: [left_hip, right_hip]\n",
        encoding="utf-8",
    )

    real_open = builtins.open

    def open_with_gbk_default(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("encoding", "gbk")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_with_gbk_default)

    model_info = ModelInfo.from_config_path(config_path)

    assert model_info.name == "minimal"
    assert model_info.tracked_point_names == ["left_hip", "right_hip"]
