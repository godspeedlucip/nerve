import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDTM import nnUNetTrainerDTM, nnUNetTrainerDTM_GSL


class nnUNetTrainerA(nnUNetTrainer):
    """
    Trainer class for network A.
    Kept separate so nnU-Net writes A and B checkpoints into different result folders.
    """
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainerB(nnUNetTrainer):
    """
    Trainer class for network B.
    Kept separate so nnU-Net writes A and B checkpoints into different result folders.
    """
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainerDTM_A(nnUNetTrainerDTM):
    """Dual-training A compatibility name for the DTM monitor preset."""

    pass


class nnUNetTrainerDTM_B(nnUNetTrainerDTM):
    """Dual-training B compatibility name for the DTM monitor preset."""

    pass


class nnUNetTrainerDTM_GSL_A(nnUNetTrainerDTM_GSL):
    """Dual-training A compatibility name for the active DTM preset."""

    pass


class nnUNetTrainerDTM_GSL_B(nnUNetTrainerDTM_GSL):
    """Dual-training B compatibility name for the active DTM preset."""

    pass
