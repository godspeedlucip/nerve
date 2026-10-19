"""Configuration contracts for the unified enhancement pipeline."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal


DTMMode = Literal["disabled", "monitor", "active"]


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class EnhancementConfig:
    """Immutable, validated configuration shared by pipeline components."""

    enable_anisotropic_attention: bool = False
    enable_tubular_enhancement: bool = False
    projection_angles_deg: tuple[float, ...] = ()
    projection_consistency_weight: float = 0.0
    dtm_mode: DTMMode = "disabled"
    dtm_weight: float = 0.0
    dtm_activate_after_epoch: int = 0
    dtm_active_weight: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enable_anisotropic_attention, bool):
            raise TypeError("enable_anisotropic_attention must be a bool")
        if not isinstance(self.enable_tubular_enhancement, bool):
            raise TypeError("enable_tubular_enhancement must be a bool")
        if not isinstance(self.dtm_activate_after_epoch, int) or isinstance(
            self.dtm_activate_after_epoch, bool
        ):
            raise TypeError("dtm_activate_after_epoch must be an integer")

        angles = tuple(_finite_float("projection_angles_deg", angle) for angle in self.projection_angles_deg)
        object.__setattr__(self, "projection_angles_deg", angles)
        projection_weight = _finite_float(
            "projection_consistency_weight", self.projection_consistency_weight
        )
        dtm_weight = _finite_float("dtm_weight", self.dtm_weight)
        active_weight = _finite_float("dtm_active_weight", self.dtm_active_weight)
        object.__setattr__(self, "projection_consistency_weight", projection_weight)
        object.__setattr__(self, "dtm_weight", dtm_weight)
        object.__setattr__(self, "dtm_active_weight", active_weight)

        if self.dtm_mode not in ("disabled", "monitor", "active"):
            raise ValueError(
                f"dtm_mode must be 'disabled', 'monitor', or 'active', got {self.dtm_mode!r}"
            )
        if projection_weight < 0:
            raise ValueError("projection_consistency_weight must be non-negative")
        if dtm_weight < 0:
            raise ValueError("dtm_weight must be non-negative")
        if active_weight < 0:
            raise ValueError("dtm_active_weight must be non-negative")
        if self.dtm_activate_after_epoch < 0:
            raise ValueError("dtm_activate_after_epoch must be non-negative")
        if not angles and self.projection_consistency_weight != 0:
            raise ValueError(
                "projection_angles_deg cannot be empty when projection_consistency_weight is non-zero"
            )

    @classmethod
    def from_legacy_preset(cls, name: str) -> "EnhancementConfig":
        """Return the configuration represented by an existing trainer name."""

        normalized = name.removesuffix("_A").removesuffix("_B")
        aliases = {
            "aniso_stent": "nnUNetTrainerAnisoStent",
            "aniso_stent_projcons": "nnUNetTrainerAnisoStent_ProjCons",
            "dtm": "nnUNetTrainerDTM",
            "dtm_gsl": "nnUNetTrainerDTM_GSL",
        }
        normalized = aliases.get(normalized, normalized)

        if normalized == "nnUNetTrainerAnisoStent":
            return cls(
                enable_anisotropic_attention=True,
                enable_tubular_enhancement=True,
            )
        if normalized == "nnUNetTrainerAnisoStent_ProjCons":
            return cls(
                enable_anisotropic_attention=True,
                enable_tubular_enhancement=True,
                projection_angles_deg=(0.0, 30.0, 60.0),
                projection_consistency_weight=0.1,
            )
        if normalized == "nnUNetTrainerDTM":
            return cls(
                dtm_mode="monitor",
                dtm_activate_after_epoch=250,
                dtm_active_weight=0.5,
            )
        if normalized == "nnUNetTrainerDTM_GSL":
            return cls(
                dtm_mode="active",
                dtm_weight=1.0,
                dtm_activate_after_epoch=250,
                dtm_active_weight=0.5,
            )
        raise ValueError(f"Unknown legacy preset: {name!r}")
