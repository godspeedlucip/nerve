"""Contracts for optional DTM and validity-mask training targets."""

from dataclasses import dataclass
from typing import Optional, TypeAlias

import torch
from torch import Tensor


TensorTree: TypeAlias = Tensor | list["TensorTree"] | tuple["TensorTree", ...]


def _shape_summary(value: TensorTree) -> tuple[int, ...] | list[object]:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return [_shape_summary(item) for item in value]


def _validate_tree(value: TensorTree, field_name: str, path: str = "root") -> None:
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{field_name} tree at {path} cannot be empty")
        for index, item in enumerate(value):
            _validate_tree(item, field_name, f"{path}[{index}]")
        return
    raise TypeError(
        f"{field_name} tree at {path} must contain only tensors and lists/tuples, "
        f"got {type(value).__name__}"
    )


def _validate_matching_shapes(dtm: TensorTree, valid_mask: TensorTree) -> None:
    if isinstance(dtm, torch.Tensor) and isinstance(valid_mask, torch.Tensor):
        if tuple(dtm.shape) != tuple(valid_mask.shape):
            raise ValueError(
                "DTM and valid_mask shapes must match: "
                f"dtm={tuple(dtm.shape)}, valid_mask={tuple(valid_mask.shape)}"
            )
        return

    if isinstance(dtm, (list, tuple)) and isinstance(valid_mask, (list, tuple)):
        if len(dtm) != len(valid_mask):
            raise ValueError(
                "DTM and valid_mask list lengths must match: "
                f"dtm={len(dtm)}, valid_mask={len(valid_mask)}, "
                f"dtm_shapes={_shape_summary(dtm)}, "
                f"valid_mask_shapes={_shape_summary(valid_mask)}"
            )
        for dtm_item, mask_item in zip(dtm, valid_mask):
            _validate_matching_shapes(dtm_item, mask_item)
        return

    raise ValueError(
        "DTM and valid_mask must have matching tensor/list structure: "
        f"dtm_shapes={_shape_summary(dtm)}, valid_mask_shapes={_shape_summary(valid_mask)}"
    )


def _move_to(value: Optional[TensorTree], device: torch.device | str) -> Optional[TensorTree]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to(item, device) for item in value]  # type: ignore[list-item]
    if isinstance(value, tuple):
        return tuple(_move_to(item, device) for item in value)  # type: ignore[return-value]
    raise TypeError(f"Auxiliary targets must contain tensors, got {type(value).__name__}")


def _first_resolution(value: Optional[TensorTree]) -> Optional[Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if not value:
        raise ValueError("Auxiliary target resolution lists cannot be empty")
    return _first_resolution(value[0])


@dataclass(frozen=True)
class AuxiliaryTargets:
    """Optional DTM and validity-mask tensors, including deep-supervision lists."""

    dtm: Optional[TensorTree] = None
    valid_mask: Optional[TensorTree] = None

    def __post_init__(self) -> None:
        if self.dtm is not None:
            _validate_tree(self.dtm, "dtm")
        if self.valid_mask is not None:
            _validate_tree(self.valid_mask, "valid_mask")
        if self.dtm is not None and self.valid_mask is not None:
            _validate_matching_shapes(self.dtm, self.valid_mask)

    @classmethod
    def empty(cls) -> "AuxiliaryTargets":
        return cls()

    def to(self, device: torch.device | str) -> "AuxiliaryTargets":
        return type(self)(_move_to(self.dtm, device), _move_to(self.valid_mask, device))

    def final_resolution(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
        return _first_resolution(self.dtm), _first_resolution(self.valid_mask)
