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

import pytest

from mdlu.data.modality import ImageModality


@pytest.mark.parametrize("modality, expected", [("photograph", 0), ("xray", 1), ("mr", 2), ("ct", 3)])
def test_modality_from_str(modality, expected):
    assert int(ImageModality.from_str(modality).value) == expected


def test_modality_from_str_invalid():
    with pytest.raises(ValueError, match="invalid is not a valid image modality"):
        ImageModality.from_str("invalid")


@pytest.mark.parametrize(
    "modality, expected",
    [(ImageModality.PHOTOGRAPH, 0), (ImageModality.XRAY, 1), (ImageModality.MR, 2), (ImageModality.CT, 3)],
)
def test_modality_int_equality(modality, expected):
    assert modality == expected


def test_modality_int_inequality():
    assert ImageModality.PHOTOGRAPH != 1
