from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChainConfig:
    start: str
    end: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ChainConfig":
        return cls(**data)
