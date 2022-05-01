# Copyright Justus Schock.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pytorch_lightning.utilities.enums import LightningEnum

__all__ = ["ImageModality"]


class ImageModality(LightningEnum):
    """Enum class for image modalities."""

    PHOTOGRAPH = 0
    XRAY = 1
    MR = 2
    CT = 3

    @staticmethod
    def get_dimensionality(modality: int | ImageModality) -> int:
        """Get the dimensionality of the image modality.

        Args:
            modality: The image modality.

        Returns:
            The dimensionality of the image modality (Usually 2D or 3D).
        """
        return 2 + int(int(modality) > 1)

    @classmethod
    def from_str(cls, value: str) -> ImageModality:
        """Get the image modality from a string.

        Args:
            value: The string value.

        Returns:
            The image modality.

        Raises:
            ValueError: If the string is not a valid image modality.
        """
        possible_str = super().from_str(value)
        if possible_str is not None:
            return possible_str

        raise ValueError(f"{value} is not a valid image modality")
