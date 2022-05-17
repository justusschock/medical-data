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

import gc
import importlib
import json
import os
from abc import ABCMeta, abstractmethod
from collections import Counter, namedtuple
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from numbers import Number

import torch
import torchio as tio
import tqdm
from tqdm.contrib.concurrent import process_map

from mdlu.data.modality import ImageModality
from mdlu.utils import PyTorchJsonDecoder, PyTorchJsonEncoder

ImageStats = namedtuple("ImageStats", ("spacing", "spatial_shape", "unique_intensities", "intensity_counts", "num_channels"))

__all__ = ["AbstractDataset", "AbstractDiscreteLabelDataset"]


# TODO: Add possibility to also add intensity mean and std manually
class AbstractDataset(tio.data.SubjectsDataset, metaclass=ABCMeta):
    """Abstract Dataset class defining the interface for all datasets. For usage, subclass this class and implement
    at least ``parse_subjects``.

    The workflow within the dataset is as follows:

      ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
      │``__init__``            ┌──────────┐                                                              ┌────────────────┐             │
      │                        │First Run │                                                              │ Subsequent Run │             │
      │                        └──────────┘                                                              └────────────────┘             │
      │                             │                                                                            │                      │
      │                             │                                                                   ┌────────┴──────────┐           │
      │                             ▼                                                                   ▼                   ▼           │
      │                      ┌──────────────┐                                                    ┌──────────────┐  ┌──────────────────┐ │
      │                      │Parse Subjects│                                                    │Parse Subjects│  │Parse Preprocessed│ │
      │                      │              │                                                    │              │  │     Subjects     │ │
      │                      └──────────────┘                                                    └──────────────┘  └──────────────────┘ │
      │                              ┌────────────────────────Not Equal──────────────┐                   │                   │          │
      │                              │                                               │                   └─────────┬─────────┘          │
      │                              ▼                                               │                             ▼                    │
      │ ┌─────────────────────────────────────────────────────────┐                  │                 ┌───────────────────────┐        │
      │ │                    For each Sample:                     │                  │                 │Check whether Number of│        │
      │ │  ┌─────────────┐   (Optionally Parallelized)            │                  └─────────────────│   Subjects is equal   │        │
      │ │  │ Load Sample │──┬─────────────────────┐               │                                    └───────────────────────┘        │
      │ │  └─────────────┘  │                     │               │                                                │                    │
      │ │                   │                     │               │                                             Equal                   │
      │ │              ┌────┘                     │               │                                                │                    │
      │ │              │                          │               │                                                │                    │
      │ │              ▼                          ▼               │                                                ▼                    │
      │ │  ┌──────────────────────┐   ┌──────────────────────┐    │                                    ┌───────────────────────┐        │
      │ │  │    Extract image     │   │    Extract Label     │    │                                    │ Load Image And Label  │        │
      │ │  │      Statistics      │   │      Statistics      │    │                                    │      Statistics       │        │
      │ │  └──────────────────────┘   └──────────────────────┘    │                                    └───────────────────────┘        │
      │ │              │                          │               │                                                │                    │
      │ └──────────────┼──────────────────────────┼───────────────┘                                                │                    │
      │                ▼                          ▼                                                                │                    │
      │    ┌──────────────────────┐   ┌──────────────────────┐                                                     │                    │
      │    │ Aggregate All Image  │   │ Aggregate All Label  │                                                     │                    │
      │    │      Statistics      │   │      Statistics      │                                                     │                    │
      │    │                      │   │                      │                                                     │                    │
      │    └──────────────────────┘   └──────────────────────┘                                                     │                    │
      │                │                          │                                                                │                    │
      │                │                          │                                           ┌────────────────────┤                    │
      │                └────────────┬─────────────┘                                           │                    │                    │
      │                             │                                                         │                    │                    │
      |                             ▼                      ┌─────────────────────────┐        │                    │                    │
      │               ┌──────────────────────────┐         │    Save Preprocessed    │        │                    │                    │
      │               │  Preprocess All Samples  │────────▶│         Samples         │        │                    │                    │
      │               └──────────────────────────┘         │(Optionally Parallelized)│        │                    │                    │
      │                             │                      └─────────────────────────┘        │                    │                    │
      │                             ▼                                   │                     │                    │                    │
      │               ┌──────────────────────────┐                      │                     ▼                    │                    │
      │               │   Save Image And Label   │                      │ ┌──────────────────────────────────────┐ │                    │
      │               │        Statistics        │                      │ │                                      │ │                    │
      │               └──────────────────────────┘                      │ │ Instantiate Augmentation Transforms  │ │                    │
      │                             │                         ┌─────────┼▶│                                      │ │                    │
      │                             │                         │         │ │                                      │ │                    │
      │                             └─────────────────────────┘         │ └──────────────────────────────────────┘ │                    │
      │                                                                 └─────────────────────┬────────────────────┤                    │
      │                                                                                       │                    │                    │
      │                                                                                       │                    │                    │
      │                                                                                       │                    │                    │
      │                                                                                       │                    │                    │
      └───────────────────────────────────────────────────────────────────────────────────────┼────────────────────┼────────────────────┘
                                                                                              │                    │
                                                                                              │                    │
                                ┌─────────────────────────────────────────────────────────────┼────────────────────▼───────────────────┐
    ┌───────────────────┐       │``__getitem__``                                              │ ┌────────────────────────────────────┐ │
    │                   │       │                                                             │ │                                    │ │
    │Dataloader queries │       │                  Fetch Item                                 │ │List of Preprocessed Samples On Disk│ │
    │  Dataset Sample   │───────┼──────────────────From List──────────────────────────────────┼▶│        (Not Cached in RAM)         │ │
    │                   │       │                                                             │ │                                    │ │
    └───────────────────┘       │                                                             │ └────────────────────────────────────┘ │
            ▲                   │                                                             │                    │                   │
            │                   │                                                             │                    │                   │
            │                   │                                                             │                    │                   │
            │                   │                                                             │                    │                   │
            │                   │                                                             │             Load   │                   │
            │                   │                                                             ├─────────Preprocessed                   │
            │                   │                                                             │            Sample                      │
            │                   │                                                             │                                        │
            │                   │                                                             │                                        │
            │                   │                                                             │                                        │
            │                   │                                                             ▼                                        │
            │                   │                         ┌──────────────────────────────────────────────────────────────────────┐     │
            │                   │      Return             │                                                                      │     │
            └───────────────────┼────Augmented ───────────│                          Apply Augmentation                          │     │
                                │      Sample             │                                                                      │     │
                                │                         └──────────────────────────────────────────────────────────────────────┘     │
                                │                                                                                                      │
                                │                                                                                                      │
                                │                                                                                                      │
                                │                                                                                                      │
                                └──────────────────────────────────────────────────────────────────────────────────────────────────────┘

    For extracting custom image and label statistics, you can overwrite the following ``staticmethods`` :

    - ``get_single_image_stats``
    - ``get_single_label_stats``
    - ``aggregate_image_stats``
    - ``aggregate_label_stats``

    And add your extracted statistics to the ``image_stat_attr_keys`` and ``label_stat_attr_keys`` tuples.

    Per default the following statistics are extracted:

    - image spacings
    - spatial image shape
    - image intensity occurence counts

    and based on that the following attributes are calculated:

    - :attr:`median_spacing` : the median spacing of all images
    - :attr:`computed_target_spacing` : the target spacing computed by the dataset
    - :attr:`target_spacing` : the externally passed target spacing if provided else the computed one
    - :attr:`sizes_after_resampling` : the sizes of the images after resampling to target spacing
    - :attr:`median_size_after_resampling` : the median size of all images after resampling to target spacing
    - :attr:`target_size` : the externally passed target size if provided else the computed one
    - :attr:`mean_intensity_value` : the mean intensity value of all images
    - :attr:`std_intensity_value` : the standard deviation of all images
    - :attr:`max_size_after_resampling` : the maximum size of all images after resampling to target spacing
    - :attr:`min_size_after_resampling` : the minimum size of all images after resampling to target spacing
    - :attr:`default_preprocessing` : the default preprocessing applied to all images and labels
    - :attr:`default_augmentation` : the default augmentation applied to all images and labels

    Per default, no label statistics are extracted, as the labels are highly task-dependent.
    """

    extension_mapping = {
        tio.constants.INTENSITY: ".nii.gz",
        tio.constants.LABEL: ".nii.gz",
    }
    restore_mapping = {
        tio.constants.INTENSITY: tio.data.ScalarImage,
        tio.constants.LABEL: tio.data.LabelMap,
    }
    image_stat_attr_keys: tuple[str, ...] = (
        "spacings",
        "spatial_shapes",
        "intensity_counts",
        "num_channels_all_images",
    )
    label_stat_attr_keys: tuple[str, ...] = ()

    pre_stats_trafo = tio.transforms.ToCanonical()

    spacings: torch.Tensor
    spatial_shapes: torch.Tensor
    intensity_counts: Counter
    num_channels: torch.Tensor

    def __init__(
        self,
        *paths: Path | str,
        preprocessed_path: Path | str | None,
        image_modality: ImageModality | int | str,
        image_stat_key: str | None = None,
        label_stat_key: str | None = None,
        preprocessing: (tio.transforms.Transform | Callable[[tio.data.Subject], tio.data.Subject] | None) = "default",
        augmentation: tio.transforms.Transform | Callable[[tio.data.Subject], tio.data.Subject] | None = None,
        statistic_collection_nonzero: bool = False,
        num_stat_collection_procs: int = 0,
        num_save_procs: int = 0,
        anisotropy_threshold: int = 3,
        target_spacing: Sequence[float] | torch.Tensor | None = None,
        target_size: Sequence[int] | torch.Tensor | None = None,
    ):

        """

        Args:
            paths: Paths to the data (images, labels and everything else). Just used to pass 1:1 to ``parse_subjects``.
            preprocessed_path: Path to the preprocessed data. If not None, will attempt to load the preprocessed data
                if the path exists and will sotre the preprocessed data there if the path does not yet exist.
            image_modality: The modality of the image. Can be either an instance of :class:`ImageModality`, an integer or string from the following list:
                - ``PHOTOGRAPH`` | 0
                - ``XRAY``       | 1
                - ``MR``         | 2
                - ``CT``         | 3
            image_stat_key: The key of the image within the subject to extract the statistics from. If None, will use the first image.
            label_stat_key: The key of the label within the subject to extract the statistics from. If None, won't extract any statistics.
            preprocessing: The preprocessing to apply to the subjects. If None, will not apply any preprocessing. If "default", will apply the :attr`default_preprocessing`.
            augmentation: The augmentation to apply to the subjects. If None, will not apply any augmentation. If "default", will apply the :attr`default_augmentation`.
            statistic_collection_nonzero: If True, will crop to nonzero before collecting statistics.
            num_stat_collection_procs: The number of processes to use for collecting the statistics.
                If 0, will only use the main process. If 1, will also just use one process (but a spawned one), if >1, will use multiple processes.
                Using 0 and 1 will be similarly fast, but using 0 won't do any pickling. Switch between 0 and 1 for debugging purposes regarding pickling.
            num_save_procs: The number of processes to use for saving the data.
                If 0, will only use the main process. If 1, will also just use one process (but a spawned one), if >1, will use multiple processes.
            anisotropy_threshold: If largest spacing > (anisotropy_threshold * second largest spacing), an image is considered to be anisotropic.
            target_spacing: The target spacing to use for resampling. If None, will resample to the computed target spacing.
            target_size: The target size to use for resampling. If None, will crop or pad to the computed target size.
        """
        # need to do this early on as torch hacks some custom things (like __getattr__)
        torch.utils.data.Dataset.__init__(self)
        parsed_subjects = self.parse_subjects(*paths)
        if isinstance(image_modality, str) and not isinstance(image_modality, ImageModality):
            image_modality = ImageModality.from_str(image_modality)

        self.image_modality = image_modality
        self.image_stats_key = image_stat_key
        self.label_stats_key = label_stat_key

        self.anisotropy_threshold = anisotropy_threshold

        if not isinstance(target_spacing, torch.Tensor) and target_spacing is not None:
            target_spacing = torch.tensor(target_spacing)
        self._target_spacing = target_spacing

        if not isinstance(target_size, torch.Tensor) and target_size is not None:
            target_size = torch.tensor(target_size, dtype=torch.long)
        self._target_size = target_size

        preprocessed_parsed_subjects: list[tio.data.Subject] | tuple | None = None
        if preprocessed_path is not None and os.path.exists(preprocessed_path):
            try:
                preprocessed_parsed_subjects, dataset_stats = self.restore_preprocessed(preprocessed_path)
            except Exception as e:
                # raise Warning for exception
                preprocessed_parsed_subjects, dataset_stats = (), None

            # Incorrect Saving/Loading -> Trigger warning
            if len(preprocessed_parsed_subjects) != len(parsed_subjects):
                preprocessed_parsed_subjects, dataset_stats = None, None
        else:
            preprocessed_parsed_subjects, dataset_stats = None, None

        if dataset_stats is None:
            dataset_stats = self.collect_dataset_stats(
                *parsed_subjects,
                image_stat_key=image_stat_key,
                label_stat_key=label_stat_key,
                crop_to_nonzero=statistic_collection_nonzero,
                num_procs=num_stat_collection_procs,
            )

        self.set_image_stat_attributes(dataset_stats["image"])
        self.set_label_stat_attributes(dataset_stats["label"])

        transforms = []

        preprocessing_trafo = self.get_preprocessing_transforms(preprocessing)
        if preprocessed_parsed_subjects is None and preprocessed_path is not None:
            self.save_preprocessed(
                *parsed_subjects,
                save_path=preprocessed_path,
                preprocessing_trafo=preprocessing_trafo,
                num_procs=num_save_procs,
            )
            preprocessed_parsed_subjects, _ = self.restore_preprocessed(preprocessed_path)

        elif preprocessed_path is not None and preprocessing_trafo is not None:
            transforms.append(preprocessing_trafo)

        augmentations = self.get_augmentation_transforms(augmentation)

        if augmentations is not None:
            transforms.append(augmentations)

        super().__init__(
            preprocessed_parsed_subjects or parsed_subjects,
            transform=tio.transforms.Compose(transforms),
            load_getitem=True,
        )

    @abstractmethod
    def parse_subjects(self, *paths: Path | str) -> Sequence[tio.data.Subject]:
        """Parses the subjects from the given paths. To be implemented in subclasses by the user.

        Args:
            paths: Paths to the data (images, labels and everything else).
        Returns:
            A sequence of subjects (ideally not yet loaded).
        """
        pass

    def set_image_stat_attributes(self, image_stats: dict[str, torch.Tensor | Counter]):
        """Sets the image statistics attributes.

        Args:
            image_stats: The image statistics to set as attributes.
        """
        for name in self.image_stat_attr_keys:
            setattr(self, name, image_stats[name])

    def set_label_stat_attributes(self, label_stats: dict[str, Any]):
        """Sets the label statistics attributes.

        Args:
            label_stats: The label statistics to set as attributes.
        """
        for name in self.label_stat_attr_keys:
            setattr(self, name, label_stats[name])

    @staticmethod
    def get_single_image_stats(image: tio.data.Image) -> Any | ImageStats:
        """Gets the image statistics for a single image.

        Args:
            image: The image to get the statistics for.
        Returns:
            The image statistics (spacing, spatial_shape, unique values, occurence counts).
        """
        uniques, counts = image.tensor[image.tensor > image.tensor.min()].unique(return_counts=True)
        uniques, counts = uniques.tolist(), counts.tolist()

        return ImageStats(image.spacing, image.spatial_shape, uniques, counts, image.tensor.size(0))

    @staticmethod
    def get_single_label_stats(label: Any, *args: Any, **kwargs: Any) -> Any:
        """Gets the label statistics for a single label.

        Args:
            label: The label to get the statistics for.

        Returns:
            The label statistics (None per default)
        """
        pass

    @staticmethod
    def aggregate_image_stats(*image_stats: Any | ImageStats) -> dict[str, Any | torch.Tensor]:
        """Aggregates the image statistics.

        Args:
            image_stats: The image statistics to aggregate.

        Returns:
            Dictionary with aggregated image statistics.
        """
        intensity_values: Counter[float] = Counter()

        spacings = tuple(map(attrgetter("spacing"), image_stats))
        shapes = tuple(map(attrgetter("spatial_shape"), image_stats))
        unique_intensities = map(attrgetter("unique_intensities"), image_stats)
        intensity_counts = map(attrgetter("intensity_counts"), image_stats)
        num_channels = tuple(map(attrgetter("num_channels"), image_stats))

        for u, c in zip(unique_intensities, intensity_counts):
            intensity_values.update(dict(zip(u, c)))

        return {
            "spacings": torch.tensor(spacings, dtype=torch.float),
            "spatial_shapes": torch.tensor(shapes, dtype=torch.long),
            "intensity_counts": intensity_values,
            "num_channels_all_images": torch.tensor(num_channels, dtype=torch.long)
        }

    @staticmethod
    def aggregate_label_stats(*stats: Any) -> dict[str, Any] | None:
        """Aggregates the label statistics.

        Args:
            stats: The label statistics to aggregate.

        Returns:
            Dictionary with aggregated label statistics.
        """
        pass

    def collect_stats_single_subject(
        self,
        subject: tio.data.Subject,
        image_stat_key: str | None,
        label_stat_key: str | None,
        crop_to_nonzero: bool = False,
    ) -> tuple[ImageStats | Any, Any | None]:
        """Collects the statistics for a single subject.

        Args:
            subject: The subject to collect the statistics for.
            image_stat_key: The key to use for the image statistics.
            label_stat_key: The key to use for the label statistics.
            crop_to_nonzero: Whether to crop the image to non-zero values.
        Returns:
            The image statistics and the label statistics.
        """

        # do all the copying here to avoid loading the acutal subject.
        # loading a copy is fine as this will be garbage collected at the end of the function
        # loading the original subject which is also referenced somewhere else would lead to a memory leak
        subject_copy = deepcopy(subject)

        subject = self.pre_stats_trafo(subject_copy)

        if crop_to_nonzero:
            from mdlu.transforms import CropToNonZero

            trafo = CropToNonZero()
            subject = trafo(subject)

        # TODO: Add Warnings if keys not present
        if image_stat_key:
            image = subject_copy.get(image_stat_key, subject_copy.get_first_image())
        else:
            image = subject_copy.get_first_image()

        if label_stat_key:
            label = subject_copy.get(label_stat_key, None)
        else:
            label = None

        stats = deepcopy((self.get_single_image_stats(image), self.get_single_label_stats(label)))
        del subject_copy
        del image
        del label
        gc.collect()

        return stats

    def collect_dataset_stats(
        self,
        *subjects: tio.data.Subject,
        image_stat_key: str | None,
        label_stat_key: str | None,
        crop_to_nonzero: bool,
        num_procs: int,
    ) -> dict[str, dict[str, Any | None]]:
        """Collects the statistics for the whole dataset.

        Args:
            subjects: The subjects to collect the statistics for.
            image_stat_key: The key to use for the image statistics.
            label_stat_key: The key to use for the label statistics.
            crop_to_nonzero: Whether to crop the image to non-zero values.
            num_procs: The number of processes to use.

        Returns:
            The collected statistics.
        """
        func = partial(
            self.collect_stats_single_subject,
            image_stat_key=image_stat_key,
            label_stat_key=label_stat_key,
            crop_to_nonzero=crop_to_nonzero,
        )

        desc = "Retrieving Subject Statistics"
        if num_procs == 0:
            results = list(map(func, tqdm.tqdm(subjects, desc=desc)))
        else:
            results = process_map(func, subjects, max_workers=num_procs, desc=desc)

        return {
            "image": self.aggregate_image_stats(
                *map(itemgetter(0), results),
            ),
            "label": self.aggregate_label_stats(*map(itemgetter(1), results)),  # type: ignore
        }

    def get_preprocessing_transforms(self, preprocessing) -> tio.transforms.Transform | None:
        """Gets the preprocessing transforms.

        Args:
            preprocessing: The preprocessing to get the transforms for.

        Returns:
            The preprocessing transforms.

        Raises:
            ValueError: If the preprocessing is not supported.
        """
        if preprocessing is None or (isinstance(preprocessing, str) and preprocessing.lower() == "none"):
            return None
        elif callable(preprocessing):
            return preprocessing

        elif isinstance(preprocessing, str) and preprocessing == "default":
            return self.default_preprocessing

        else:
            raise ValueError(f"Invalid Preprocessing: {preprocessing}")

    def get_augmentation_transforms(self, augmentation) -> tio.transforms.Transform | None:
        """Gets the augmentation transforms.

        Args:
            augmentation: The augmentation to get the transforms for.

        Returns:
            The augmentation transforms.

        Raises:
            ValueError: If the augmentation is not supported.
        """
        if augmentation is None or (isinstance(augmentation, str) and augmentation.lower() == "none"):
            return None
        elif callable(augmentation):
            return augmentation

        elif isinstance(augmentation, str) and augmentation == "default":
            return self.default_augmentation

        else:
            raise ValueError(f"Invalid Augmentation: {augmentation}")

    def save_single_preprocessed_subject(
        self,
        idx: int,
        subject: tio.data.Subject,
        save_path: str | Path,
        preprocessing_trafo: tio.transform.Transform | Callable[[tio.data.Subject], tio.data.Subject] | None,
        total_num_subjects: int,
    ) -> None:
        """Preprocesses and saves a single preprocessed subject to a json file for metadata, an image and a label
        file,

        Args:
            idx: The index of the subject.
            subject: The subject to save.
            save_path: The path to save the subject to.
            preprocessing_trafo: The preprocessing transform to use.
            total_num_subjects: The total number of subjects.
        """
        # do all the copying here to avoid loading the acutal subject.
        # loading a copy is fine as this will be garbage collected at the end of the function
        # loading the original subject which is also referenced somewhere else would lead to a memory leak
        subject_copy = deepcopy(subject)
        if preprocessing_trafo is not None:
            subject_copy = preprocessing_trafo(subject_copy)

        save_path = os.path.join(save_path, str(idx).zfill(len(str(total_num_subjects))))
        os.makedirs(save_path, exist_ok=True)

        tio_images = {}
        to_dump = {}
        for k, v in subject_copy.items():
            if isinstance(v, tio.data.Image):
                tio_images[k] = v.type
                v.save(os.path.join(save_path, k + self.extension_mapping[v.type]))
            else:
                to_dump[k] = v

        to_dump["TORCHIO_SUBJECT_CLASS"] = ".".join((type(subject).__module__, type(subject).__qualname__))
        to_dump["TORCHIO_IMAGE_TYPES"] = tio_images

        with open(os.path.join(save_path, "subject.json"), "w") as f:
            json.dump(to_dump, f, indent=4, sort_keys=True, cls=PyTorchJsonEncoder)

    def _wrapped_save_single_preprocessed_subject(
        self,
        args: tuple[int, tio.data.Subject],
        save_path: str | Path,
        preprocessing_trafo: tio.transform.Transform | Callable[[tio.data.Subject], tio.data.Subject] | None,
        total_num_subjects: int,
    ) -> None:
        """Wrapper for the save_single_preprocessed_subject function to enable its usage with multiprocessing.

        Args:
            args: The arguments forward to the function (usually index and Subject).
            save_path: The path to save the subject to.
            preprocessing_trafo: The preprocessing transform to use.
            total_num_subjects: The total number of subjects.
        """
        self.save_single_preprocessed_subject(
            *args, save_path=save_path, preprocessing_trafo=preprocessing_trafo, total_num_subjects=total_num_subjects
        )
        return self.save_single_preprocessed_subject(
            *args,
            save_path=save_path,
            preprocessing_trafo=preprocessing_trafo,
            total_num_subjects=total_num_subjects,
        )

    def image_state_dict(self) -> dict[str, Any]:
        """Returns the state dict of the image statistics across the whole dataset.

        Returns:
            The state dict of the image statistics.
        """
        return {k: getattr(self, k) for k in self.image_stat_attr_keys}

    def label_state_dict(self) -> dict[str, Any]:
        """Returns the state dict of the label statistics across the whole dataset.

        Returns:
            The state dict of the label statistics.
        """

        return {k: getattr(self, k) for k in self.label_stat_attr_keys}

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Returns the combined state dict for images and labels of the statistics across the whole dataset.

        Returns:
            The state dict of the statistics.
        """
        return {"image": self.image_state_dict(), "label": self.label_state_dict()}

    def save_preprocessed(self, *subjects, save_path, preprocessing_trafo, num_procs) -> None:
        """Preprocesses and saves all subjects in the dataset to a json file for metadata, an image and a label
        file each. Also saves a json file containing the state_dict of the dataset.

        Args:
            subjects: The subjects to save.
            save_path: The path to save the subjects to.
            preprocessing_trafo: The preprocessing transform to use.
            num_procs: The number of processes to use for multiprocessing.
        """
        func = partial(
            self._wrapped_save_single_preprocessed_subject,
            save_path=save_path,
            preprocessing_trafo=preprocessing_trafo,
            total_num_subjects=len(subjects),
        )

        desc = "Saving Preprocessed Subjects"
        if num_procs == 0:
            list(map(func, enumerate(tqdm.tqdm(subjects, desc=desc))))
        else:
            process_map(
                func,
                enumerate(subjects),
                max_workers=num_procs,
                desc=desc,
                total=len(subjects),
            )

        stat_dict = self.state_dict()

        with open(os.path.join(save_path, "dataset.json"), "w") as f:
            json.dump(
                stat_dict,
                f,
                indent=4,
                sort_keys=True,
                cls=PyTorchJsonEncoder,
            )

    def restore_preprocessed(self, preprocessed_path) -> tuple[list[tio.data.Subject], dict[str, Any]]:
        """Restores the preprocessed subjects from a json file for metadata, an image and a label file each. Also
        restores the state_dict of the dataset.

        Args:
            preprocessed_path: The path to the preprocessed subjects.

        Returns:
            The subjects and the state dict of the dataset.
        """
        with open(os.path.join(preprocessed_path, "dataset.json")) as f:
            dset_meta = json.load(f, cls=PyTorchJsonDecoder)

        subjects = []
        subs = sorted(
            (
                os.path.join(preprocessed_path, x)
                for x in os.listdir(preprocessed_path)
                if os.path.isdir(os.path.join(preprocessed_path, x))
            ),
            key=lambda _x: int(os.path.split(_x)[1]),
        )

        for sub in subs:
            # load information about subject
            with open(os.path.join(sub, "subject.json")) as f:
                subject_meta = json.load(f, cls=PyTorchJsonDecoder)

            subject_cls_path = subject_meta.pop("TORCHIO_SUBJECT_CLASS", "")

            if subject_cls_path:

                cls_module_path, cls_name = subject_cls_path.rsplit(".", 1)
                cls_module = importlib.import_module(cls_module_path)

                if cls_module is not None:
                    subject_cls = getattr(cls_module, cls_name, None)
                else:
                    subject_cls = None
            else:
                subject_cls = None

            subject_cls = subject_cls or tio.data.Subject

            images = subject_meta.pop("TORCHIO_IMAGE_TYPES", {})
            for k, v in images.items():
                subject_meta[k] = self.restore_mapping[v](os.path.join(sub, k + self.extension_mapping[v]))

            subjects.append(subject_cls(subject_meta))

        return subjects, dset_meta

    @property
    def median_spacing(self) -> torch.Tensor:
        """Computes the median spacing of the dataset.

        Returns:
            The median spacing of the dataset.
        """
        return torch.median(self.spacings, 0).values

    @property
    def computed_target_spacing(self) -> torch.Tensor:
        """Computes the target spacing of the dataset under considereation of possible anisotrophies and the
        expected size after resampling to median spacing.

        Returns:
            The computed target spacing of the dataset.
        """
        target_spacing = self.median_spacing
        target_size = self.median_size_after_resampling
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = torch.argmax(target_spacing)
        other_axes = [i for i in range(len(target_spacing)) if i != worst_spacing_axis]
        other_spacings = [target_spacing[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target_spacing[worst_spacing_axis] > (self.anisotropy_threshold * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = self.spacings[:, worst_spacing_axis]
            target_spacing_of_that_axis = torch.quantile(spacings_of_that_axis, 0.1)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target_spacing[worst_spacing_axis] = target_spacing_of_that_axis

        return target_spacing

    @property
    def target_spacing(self) -> torch.Tensor:
        """Returns the externally given target spacing if provided and falls back to computing it.

        Returns:
            The target spacing for the dataset
        """
        if self._target_spacing is None:
            return self.computed_target_spacing
        return self._target_spacing

    @property
    def num_channels(self) -> int:
        if self.num_channels_all_images.unique(return_counts=False) == 1:
            return self.num_channels_all_images[0].item()
        raise RuntimeError(
            f"""All images need to have the same number of channels, 
            but got images with the following numbers of channels: {self.num_channels_all_images}"""
        )

    @property
    def sizes_after_resampling(self) -> torch.Tensor:
        """Computes the image sizes of the dataset after resampling to the median spacing.

        Returns:
            The image sizes after resampling
        """
        return (self.spatial_shapes.float() / self.spacings * self.median_spacing).round().long()

    @property
    def median_size_after_resampling(self) -> torch.Tensor:
        """Computes the median image size of the dataset after resampling to the median spacing.

        Returns:
            The median image size after resampling
        """
        return torch.median(self.sizes_after_resampling, 0).values

    @property
    def target_size(self) -> torch.Tensor:
        """Returns the externally given target size if provided and falls back to computing it as the median size
        after resampling.

        Returns:
            The target size for the dataset
        """
        if self._target_size is None:
            return self.median_size_after_resampling
        return self._target_size

    @property
    def mean_intensity_value(self) -> torch.Tensor:
        """Computes the mean image intensity value of the dataset.

        Returns:
            The mean intensity value of the dataset.
        """
        return torch.tensor(
            sum(float(k) * float(v) for k, v in self.intensity_counts.items()) / sum(self.intensity_counts.values())
        )

    @property
    def std_intensity_value(self) -> torch.Tensor:
        """Computes the standard deviation of the image intensity values of the dataset.

        Returns:
            The standard deviation of the image intensity values.
        """

        n = sum(self.intensity_counts.values())
        # normal intensity calculation without allocating all the occurences to save memory
        return (
            (
                sum(float(k) * float(v) ** 2 for k, v in self.intensity_counts.items())
                - n * self.mean_intensity_value ** 2
            )
            / (n - 1)
        ).sqrt()

    @property
    def max_size_after_resampling(self) -> torch.Tensor:
        """Computes the maximum image size of the dataset after resampling to the median spacing.

        Returns:
            The maximum image size after resampling.
        """
        return torch.max(self._sizes_after_resampling, 0).values

    @property
    def min_size_after_resampling(self) -> torch.Tensor:
        """Computes the minimum image size of the dataset after resampling to the median spacing.

        Returns:
            The minimum image size after resampling.
        """
        return torch.min(self._sizes_after_resampling, 0).values

    @property
    def default_preprocessing(self) -> Callable[[tio.data.Subject], tio.data.Subject] | tio.transforms.Transform | None:
        """Returns the default preprocessing for the dataset.

        Returns:
            The default preprocessing for the dataset.
        """
        from mdlu.transforms import DefaultPreprocessing

        return DefaultPreprocessing(
            target_spacing=self.target_spacing.tolist(),
            target_size=self.target_size.tolist(),
            modality=self.image_modality,
            dataset_intensity_mean=self.mean_intensity_value,
            dataset_intensity_std=self.std_intensity_value,
            affine_key=self.image_stats_key,
        )

    @property
    def default_augmentation(self) -> Callable[[tio.data.Subject], tio.data.Subject] | tio.transforms.Transform | None:
        """Returns the default augmentation for the dataset.

        Returns:
            The default augmentation for the dataset.
        """
        from mdlu.transforms import DefaultAugmentation

        return DefaultAugmentation(
            self.image_modality,
            include_deformation=False,
        )


class AbstractDiscreteLabelDataset(AbstractDataset):
    """Abstract dataset for a task with discrete labels (like semantic segmentation or classification).

    Additionally to the image statistics it also stores the class values as label statistics and derives the following attributes:

    - :attr:`consecutive_class_mapping`: A mapping from consecutive class values to the original class values.
    - :attr:`inverse_class_mapping`: A mapping from original class values to consecutive class values.
    - :attr`num_classes`: The number of classes in the dataset.

    For more information see :class:`AbstractDataset`.
    """

    class_values: torch.Tensor
    label_stat_attr_keys = ("class_values", "spatial_label_shapes", *AbstractDataset.label_stat_attr_keys)

    @staticmethod
    def get_single_label_stats(label: tio.data.Image | torch.Tensor | Number, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor | Any] | Any:
        """Computes the label statistics for a single label.

        Args:
            label: The label to compute the statistics for.

        Returns:
            The label statistics for the label.
        """
        if isinstance(label, tio.data.Image):
            label_tensor = label.tensor
        elif isinstance(label, torch.Tensor):
            label_tensor = label
        elif isinstance(label, Number):
            label_tensor = torch.tensor(label).view(1,)
        else:
            raise TypeError(f'label has to be either torchio Image, torch.Tensor or Number, got {type(label)}.')
        return {"class_values": label_tensor.unique().tolist(), 'spatial_shape': label_tensor.shape[1:]}

    @staticmethod
    def aggregate_label_stats(*label_stats: Any) -> dict[str, torch.Tensor | Any] | None:
        """Aggregates the label statistics of multiple labels.

        Args:
            label_stats: The label statistics of multiple labels.

        Returns:
            The aggregated label statistics.
        """
        return {
            "class_values": torch.tensor(sorted(set(chain.from_iterable(map(itemgetter("class_values"), label_stats))))),
            "spatial_label_shapes": torch.tensor(sorted(set(chain.from_iterable(map(itemgetter("spatial_shape"), label_stats)))), dtype=torch.long)
        }

    @property
    def target_label_size(self) -> torch.Tensor:
        if torch.allclose(self.spatial_label_shapes, self.spatial_shapes):
            return self.target_size
        
        if self.spatial_label_shape.numel() == 0:
            return torch.tensor([1])
        
        median = self.spatial_label_shapes.median(0)
        if torch.isnan(median).any():
            return torch.tensor([1])
        return median

    @property
    def consecutive_class_mapping(self) -> dict[int, int]:
        """Returns the mapping from consecutive class values to the original class values.

        Returns:
            The mapping from consecutive class values to the original class values.
        """
        return dict(enumerate(self.class_values.tolist()))

    @property
    def inverse_class_mapping(self) -> dict[int, int]:
        """Returns the mapping from original class values to consecutive class values.

        Returns:
            The mapping from original class values to consecutive class values.
        """
        return {v: k for k, v in self.consecutive_class_mapping.items()}

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset.

        Returns:
            The number of classes in the dataset.
        """
        return self.class_values.numel()

    @property
    def default_preprocessing(self) -> Callable[[tio.data.Subject], tio.data.Subject] | tio.transforms.Transform | None:
        """Returns the default preprocessing for the dataset.

        Returns:
            The default preprocessing for the dataset.
        """
        from mdlu.transforms import DefaultPreprocessing

        return tio.transforms.Compose(
            [
                tio.transforms.RemapLabels(self.inverse_class_mapping),
                DefaultPreprocessing(
                    target_spacing=self.target_spacing.tolist(),
                    target_size=self.median_size_after_resampling.long().tolist(),
                    modality=self.image_modality,
                    dataset_intensity_mean=self.mean_intensity_value,
                    dataset_intensity_std=self.std_intensity_value,
                    affine_key=self.image_stats_key,
                    num_classes=self.num_classes,
                ),
            ]
        )
