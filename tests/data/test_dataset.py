
from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping, Sequence
import pytest
import torch

from mdlu.data.dataset import AbstractDiscreteLabelDataset
import torchio as tio
import os
from pytorch_lightning.utilities.apply_func import apply_to_collection

from mdlu.utils import PyTorchJsonDecoder
class DICOMImageNiftiLabelSegmentationDataset(AbstractDiscreteLabelDataset):
    def parse_subjects(self, *paths: Path | str) -> Sequence[tio.data.Subject]:
        subjects = []
        assert len(paths) == 2
        image_root_path = Path(paths[0])
        label_root_path = Path(paths[1])
        for label_path in label_root_path.rglob("*.nii.gz"):
            image_path = os.path.join(
                image_root_path,
                str(label_path)
                .replace(".nii.gz", "")
                .replace(str(label_root_path), str(image_root_path)),
            )

            if os.path.exists(image_path):
                rem_path, series_id = os.path.split(image_path)
                rem_path, study_id = os.path.split(rem_path)

                patient_id = os.path.split(rem_path)[1]
                subjects.append(
                    tio.data.Subject(
                        dict(
                            data=tio.data.ScalarImage(image_path),
                            label=tio.data.LabelMap(label_path),
                            patient_id=patient_id,
                            study_id=study_id,
                            series_id=series_id,
                            original_data_path=str(image_path),
                            original_label_path=str(label_path),
                        )
                    )
                )

        return subjects

def _dataset_create(dummy_dataset, preprocessed_path: None | str| Path = None, **kwargs):
    return DICOMImageNiftiLabelSegmentationDataset(dummy_dataset['images'], dummy_dataset['labels'], image_modality='MR', image_stat_key='data', label_stat_key='label', preprocessed_path=preprocessed_path, **kwargs)

def test_dataset_length(dummy_dataset):
    dataset = _dataset_create(dummy_dataset)
    assert len(dataset) == 10

def test_dataset_attributes(dummy_dataset):
    dataset = _dataset_create(dummy_dataset)
    for k in (*dataset.image_stat_attr_keys, *dataset.label_stat_attr_keys):
        assert hasattr(dataset, k)

def test_dataset_statedict(dummy_dataset):
    dataset = _dataset_create(dummy_dataset)
    image_state_dict = dataset.image_state_dict()
    label_state_dict = dataset.label_state_dict()
    state_dict = dataset.state_dict()
    assert image_state_dict == state_dict['image']
    assert label_state_dict == state_dict['label']
    assert len(state_dict) == 2

    for k, v in image_state_dict.items():
        assert k in dataset.image_stat_attr_keys
        result = v == getattr(dataset, k)
        if isinstance(result, torch.Tensor):
            result = result.any()
        assert result

    assert len(image_state_dict) == len(dataset.image_stat_attr_keys)

    for k, v in label_state_dict.items():
        result = v == getattr(dataset, k)
        if isinstance(result, torch.Tensor):
            result = result.any()
        assert result

    assert len(label_state_dict) == len(dataset.label_stat_attr_keys)

def _compare_nested_dicts(d1, d2):
    for k, v in d1.items():
        try:
            if isinstance(v, Mapping):
                _compare_nested_dicts(v, d2[k])
            elif isinstance(v, torch.Tensor):
                assert torch.allclose(v, d2[k])
            elif isinstance(v, Sequence):
                assert len(v) == len(d2[k])
                for i in range(len(v)):
                    _compare_nested_dicts(v[i], d2[k][i])
            else:
                assert v == d2[k]
        except AssertionError:
            print(f"{k} differs")
            raise

def test_preprocessed(dummy_dataset, tmpdir):
    preprocessed_path = os.path.join(str(tmpdir), 'preprocessed')

    dataset = _dataset_create(dummy_dataset, preprocessed_path=preprocessed_path, augmentation=None, preprocessing=None)
    assert os.path.isfile(os.path.join(preprocessed_path, 'dataset.json'))

    with open(os.path.join(preprocessed_path, 'dataset.json')) as f:
        state_dict = json.load(f, cls=PyTorchJsonDecoder)


    _compare_nested_dicts(state_dict, dataset.state_dict())

    for i in range(len(dataset)):
        curr_sample_dir = os.path.join(preprocessed_path, str(i).zfill(len(dataset)))
        assert os.path.isdir(curr_sample_dir)
        assert os.path.isfile(os.path.join(curr_sample_dir, 'subject.json'))

        with open(os.path.join(curr_sample_dir, 'subject.json')) as f:
            curr_sample_state = json.load(f, cls=PyTorchJsonDecoder)

        assert curr_sample_state.pop("TORCHIO_SUBJECT_CLASS") == "torchio.data.subject.Subject"
        curr_image_types = curr_sample_state.pop("TORCHIO_IMAGE_TYPES")

        assert len(curr_image_types) == 2
        assert curr_image_types["data"] == 'intensity'
        assert curr_image_types['label'] == 'label'

        assert os.path.isfile(os.path.join(curr_sample_dir, dataset.extension_mapping['data']))
        assert os.path.isfile(os.path.join(curr_sample_dir, dataset.extenion_mapping['label']))

        curr_sample_orig = dataset[i]

        reconstructed_sample = tio.data.Subject(**{k: dataset.restore_mapping[v](os.path.join(curr_sample_dir, k + dataset.extension_mapping[k])) for k, v in curr_sample_state.items()}, **{curr_sample_state})
        assert curr_sample_orig == reconstructed_sample




