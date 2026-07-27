"""Frozen configuration for the canonical PWSC-CI experiment path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class PWSCCIConfig:
    """Constants shared by every canonical PWSC-CI benchmark run."""

    weakest_k: int = 5
    method_version: str = "PWSC-CI-v1.0"
    contract_version: str = "pwsc-ci-normalized-holevo-v1"


@dataclass
class ExperimentSettings:
    """Reproducible runner settings; calibration stays outside PWSC-CI."""

    pwsc_ci: PWSCCIConfig = field(default_factory=PWSCCIConfig)
    random_seed: int = 2048
    use_calibration: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)


__all__ = ["ExperimentSettings", "PWSCCIConfig"]
