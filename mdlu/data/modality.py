from typing import Union

from pytorch_lightning.utilities.enums import LightningEnum

__all__ = ["ImageModality"]


class ImageModality(LightningEnum):
    PHOTOGRAPH = 0
    XRAY = 1
    MR = 2
    CT = 3

    @staticmethod
    def get_dimensionality(modality: Union[int, "ImageModality"]) -> int:
        return 2 + int(int(modality) > 1)
