from mdlu.data.modality import ImageModality
import pytest

@pytest.mark.parametrize("modality, expected", [("photograph", 0), ("xray", 1), ("mr", 2), ("ct", 3)])
def test_modality_from_str(modality, expected):
    assert int(ImageModality.from_str(modality).value) == expected

def test_modality_from_str_invalid():
    with pytest.raises(ValueError, match="invalid is not a valid image modality"):
        ImageModality.from_str("invalid")

@pytest.mark.parametrize("modality, expected", [(ImageModality.PHOTOGRAPH, 0), (ImageModality.XRAY, 1), (ImageModality.MR, 2), (ImageModality.CT, 3)])
def test_modality_int_equality(modality, expected):
    assert modality == expected

def test_modality_int_inequality():
    assert ImageModality.PHOTOGRAPH != 1

