from __future__ import annotations

import importlib
import json
import os
from abc import ABCMeta, abstractmethod
from collections import Counter, namedtuple
from functools import partial
from itertools import chain
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchio as tio
import tqdm
from tqdm.contrib.concurrent import process_map

from mdlu.data.modality import ImageModality
from mdlu.utils import PyTorchJsonDecoder, PyTorchJsonEncoder

ImageStats = namedtuple(
    "ImageStats", ("spacing", "spatial_shape", "unique_intensities", "intensity_counts")
)

__all__ = ["AbstractDataset"]


# TODO: Add possibility to also add intensity mean and std manually
class AbstractDataset(tio.data.SubjectsDataset, metaclass=ABCMeta):
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
    )
    label_stat_attr_keys: tuple[str, ...] = ()

    pre_stats_trafo = tio.transforms.ToCanonical()

    spacings: torch.Tensor
    spatial_shapes: torch.Tensor
    intensity_counts: Counter

    def __init__(
        self,
        *paths: Path | str,
        preprocessed_path: Path | str | None,
        image_modality: ImageModality | int,
        image_stat_key: str | None = None,
        label_stat_key: str | None = None,
        preprocessing: (
            tio.transform.Transform
            | Callable[[tio.data.Subject], tio.data.Subject]
            | None
        ) = "default",
        augmentation: tio.transform.Transform
        | Callable[[tio.data.Subject], tio.data.Subject]
        | None = None,
        statistic_collection_nonzero: bool = False,
        num_stat_collection_procs: int = 0,
        num_save_procs: int = 0,
        anisotropy_threshold: int = 3,
        target_spacing: Sequence[float] | torch.Tensor | None = None,
        target_size: Sequence[int] | torch.Tensor | None = None,
    ):
        # need to do this early on as torch hacks some custom things (like __getattr__)
        torch.utils.data.Dataset.__init__(self)
        parsed_subjects = self.parse_subjects(*paths)

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
                preprocessed_parsed_subjects, dataset_stats = self.restore_preprocessed(
                    preprocessed_path
                )
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
            preprocessed_parsed_subjects, _ = self.restore_preprocessed(
                preprocessed_path
            )

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
        pass

    def set_image_stat_attributes(self, image_stats: dict[str, torch.Tensor | Counter]):
        for name in self.image_stat_attr_keys:
            setattr(self, name, image_stats[name])

    def set_label_stat_attributes(self, label_stats: dict[str, Any]):
        for name in self.label_stat_attr_keys:
            setattr(self, name, label_stats[name])

    @staticmethod
    def get_single_image_stats(image: tio.data.Image) -> ImageStats:
        uniques, counts = image.tensor[image.tensor > image.tensor.min()].unique(
            return_counts=True
        )
        uniques, counts = uniques.tolist(), counts.tolist()

        return ImageStats(image.spacing, image.spatial_shape, uniques, counts)

    @staticmethod
    def get_single_label_stats(label: Any, *args: Any, **kwargs: Any):
        pass

    @staticmethod
    def aggregate_image_stats(*image_stats: ImageStats):
        intensity_values: Counter[float] = Counter()

        spacings = tuple(map(attrgetter("spacing"), image_stats))
        shapes = tuple(map(attrgetter("spatial_shape"), image_stats))
        unique_intensities = map(attrgetter("unique_intensities"), image_stats)
        intensity_counts = map(attrgetter("intensity_counts"), image_stats)

        for u, c in zip(unique_intensities, intensity_counts):
            intensity_values.update(dict(zip(u, c)))

        return {
            "spacings": torch.tensor(spacings, dtype=torch.float),
            "spatial_shapes": torch.tensor(shapes, dtype=torch.long),
            "intensity_counts": intensity_values,
        }

    @staticmethod
    def aggregate_label_stats(*stats: Any):
        return

    def collect_stats_single_subject(
        self,
        subject: tio.data.Subject,
        image_stat_key: str | None,
        label_stat_key: str | None,
        crop_to_nonzero: bool = False,
    ):

        subject = self.pre_stats_trafo(subject)

        if crop_to_nonzero:
            from mdlu.transforms import CropToNonZero

            trafo = CropToNonZero()
            subject = trafo(subject)

        # TODO: Add Warnings if keys not present
        if image_stat_key:
            image = subject.get(image_stat_key, subject.get_first_image())
        else:
            image = subject.get_first_image()

        if label_stat_key:
            label = subject.get(label_stat_key, None)
        else:
            label = None

        return self.get_single_image_stats(image), self.get_single_label_stats(label)

    def collect_dataset_stats(
        self,
        *subjects: tio.data.Subject,
        image_stat_key: str | None,
        label_stat_key: str | None,
        crop_to_nonzero: bool,
        num_procs: int,
    ):
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
            "label": self.aggregate_label_stats(*map(itemgetter(1), results)),
        }

    def get_preprocessing_transforms(
        self, preprocessing
    ) -> tio.transforms.Transform | None:
        if preprocessing is None or (
            isinstance(preprocessing, str) and preprocessing.lower() == "none"
        ):
            return None
        elif callable(preprocessing):
            return preprocessing

        elif isinstance(preprocessing, str) and preprocessing == "default":
            return self.default_preprocessing

        else:
            raise ValueError(f"Invalid Preprocessing: {preprocessing}")

    def get_augmentation_transforms(
        self, augmentation
    ) -> tio.transforms.Transform | None:
        if augmentation is None or (
            isinstance(augmentation, str) and augmentation.lower() == "none"
        ):
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
        preprocessing_trafo: tio.transform.Transform
        | Callable[[tio.data.Subject], tio.data.Subject]
        | None,
        total_num_subjects: int,
    ):
        if preprocessing_trafo is not None:
            subject = preprocessing_trafo(subject)

        save_path = os.path.join(
            save_path, str(idx).zfill(len(str(total_num_subjects)))
        )
        os.makedirs(save_path, exist_ok=True)

        tio_images = {}
        to_dump = {}
        for k, v in subject.items():
            if isinstance(v, tio.data.Image):
                tio_images[k] = v.type
                v.save(os.path.join(save_path, k + self.extension_mapping[v.type]))
            else:
                to_dump[k] = v

        to_dump["TORCHIO_SUBJECT_CLASS"] = ".".join(
            (type(subject).__module__, type(subject).__qualname__)
        )
        to_dump["TORCHIO_IMAGE_TYPES"] = tio_images

        with open(os.path.join(save_path, "subject.json"), "w") as f:
            json.dump(to_dump, f, indent=4, sort_keys=True, cls=PyTorchJsonEncoder)

    def _wrapped_save_single_preprocessed_subject(
        self,
        args: tuple[int, tio.data.Subject],
        save_path: str | Path,
        preprocessing_trafo: tio.transform.Transform
        | Callable[[tio.data.Subject], tio.data.Subject]
        | None,
        total_num_subjects: int,
    ):
        return self.save_single_preprocessed_subject(
            *args,
            save_path=save_path,
            preprocessing_trafo=preprocessing_trafo,
            total_num_subjects=total_num_subjects,
        )

    def image_state_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.image_stat_attr_keys}

    def label_state_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.label_stat_attr_keys}

    def state_dict(self) -> dict[str, dict[str, Any]]:
        return {"image": self.image_state_dict(), "label": self.label_state_dict()}

    def save_preprocessed(
        self, *subjects, save_path, preprocessing_trafo, num_procs
    ) -> None:
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

    def restore_preprocessed(
        self, preprocessed_path
    ) -> tuple[list[tio.data.Subject], dict[str, Any]]:
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
                subject_meta[k] = self.restore_mapping[v](
                    os.path.join(sub, k + self.extension_mapping[v])
                )

            subjects.append(subject_cls(subject_meta))

        return subjects, dset_meta

    @property
    def median_spacing(self) -> torch.Tensor:
        return torch.median(self.spacings, 0).values

    @property
    def computed_target_spacing(self) -> torch.Tensor:
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

        has_aniso_spacing = target_spacing[worst_spacing_axis] > (
            self.anisotropy_threshold * max(other_spacings)
        )
        has_aniso_voxels = target_size[
            worst_spacing_axis
        ] * self.anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = self.spacings[:, worst_spacing_axis]
            target_spacing_of_that_axis = torch.quantile(spacings_of_that_axis, 0.1)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = (
                    max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
                )
            target_spacing[worst_spacing_axis] = target_spacing_of_that_axis

        return target_spacing

    @property
    def target_spacing(self) -> torch.Tensor:
        return self._target_spacing or self.computed_target_spacing

    @property
    def sizes_after_resampling(self) -> torch.Tensor:
        return (
            (self.spatial_shapes.float() / self.spacings * self.median_spacing)
            .round()
            .long()
        )

    @property
    def median_size_after_resampling(self) -> torch.Tensor:
        return torch.median(self.sizes_after_resampling, 0).values

    @property
    def target_size(self) -> torch.Tensor:
        return self._target_size or self.median_size_after_resampling

    @property
    def mean_intensity_value(self) -> torch.Tensor:
        return torch.tensor(
            sum(float(k) * float(v) for k, v in self.intensity_counts.items())
            / sum(self.intensity_counts.values())
        )

    @property
    def std_intensity_value(self) -> torch.Tensor:

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
    def max_size_after_resampling(self):
        return torch.max(self._sizes_after_resampling, 0).values

    @property
    def min_size_after_resampling(self):
        return torch.min(self.sizes_after_resampling, 0).values

    @property
    def default_preprocessing(self):
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
    def default_augmentation(self):
        from mdlu.transforms import DefaultAugmentation

        return DefaultAugmentation(
            self.image_modality,
            include_deformation=False,
        )


class AbstractDiscreteLabelDataset(AbstractDataset):
    class_values: torch.Tensor
    label_stat_attr_keys = ("class_values", *AbstractDataset.label_stat_attr_keys)

    @staticmethod
    def get_single_label_stats(label: tio.data.Image, *args, **kwargs):
        return {"class_values": label.tensor.unique().values.tolist()}

    def aggregate_label_stats(self, *label_stats):
        return {
            "class_values": torch.tensor(
                sorted(
                    set(
                        chain.from_iterable(
                            map(itemgetter("class_values"), label_stats)
                        )
                    )
                )
            )
        }

    @property
    def consecutive_class_mapping(self) -> dict[int, int]:
        return dict(enumerate(self.class_values.tolist()))

    @property
    def inverse_class_mapping(self) -> dict[int, int]:
        return {v: k for k, v in self.consecutive_class_mapping.items()}

    @property
    def num_classes(self) -> int:
        return self.class_values.numel()

    @property
    def default_preprocessing(self):
        from mdlu.transforms import DefaultPreprocessing

        return tio.transforms.Compose(
            [
                tio.transforms.RemapLabels(self.class_mapping),
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
