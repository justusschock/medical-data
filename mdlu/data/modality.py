
from __future__ import annotations

from pytorch_lightning.utilities.enums import LightningEnum

__all__ = ["ImageModality"]


class ImageModality(LightningEnum):
    PHOTOGRAPH = 0
    XRAY = 1
    MR = 2
    CT = 3

    @staticmethod
    def get_dimensionality(modality: int | ImageModality) -> int:
        return 2 + int(int(modality) > 1)

    @classmethod
    def from_str(cls, value: str) -> "ImageModality":
        possible_str = super(ImageModality, cls).from_str(value)
        if possible_str is not None:
            return possible_str

        raise ValueError(f"{value} is not a valid image modality")
