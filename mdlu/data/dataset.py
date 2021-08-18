from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import Counter
from typing import Optional, Tuple, Union, Mapping, Any
from mdlu.utils import PyTorchJsonDecoder, PyTorchJsonEncoder
import torchio as tio
import torch
from mdlu.data.modality import ImageModality
from mdlu.transforms import CropToNonZero, DefaultPreprocessing, DefaultAugmentation
import os
import json
from packaging.version import Version
from functools import partial
import warnings
import shutil
from tqdm import tqdm
import psutil

if Version(tio.__version__) <= Version("0.18.45"):
    from mdlu.data.custom_subject import CustomSubject

    tio.data.subject.Subject = CustomSubject
    tio.data.Subject = CustomSubject
    tio.Subject = CustomSubject


class AbstractDataset(tio.data.SubjectsDataset, metaclass=ABCMeta):
    def __init__(
        self,
        root_path,
        image_modality: Union[int, ImageModality],
        save_preprocessed: Optional[str] = None,
        preprocessing: Optional[Union[tio.transforms.Transform, str]] = "default",
        augmentation: Optional[Union[tio.transforms.Transform, str]] = "default",
        load_getitem: bool = True,
        get_classes_key: Optional[str] = "label",
        get_stats_key: Optional[str] = "data",
        median_spacing: Optional[Sequence[float]] = None,
        median_size: Optional[Sequence[int]] = None,
        anisotropy_threshold: int = 3,
        num_saving_procs: Optional[int] = 1,
        verbose: bool = False,
        crop_to_nonzero_stats: bool = True,
    ):
        restored = False
        restored_meta = {}
        if save_preprocessed is not None and os.path.isdir(save_preprocessed):
            try:
                subjects_restored, restored_meta = self._restore_from_preprocessed(
                    save_preprocessed
                )
                restored = True
            except:
                warnings.warn(
                    f"Failed to restore from path {save_preprocessed}. Deleting it and recreating it. This might take some time."
                )
                shutil.rmtree(save_preprocessed)
                restored = False
        subjects_parsed = self.parse_subjects(root_path)
        if restored:
            if len(subjects_parsed) != len(subjects_restored):
                warnings.warn(
                    f"Number of restored samples ({len(subjects_restored)}) is different from the parsed samples ({len(subjects_parsed)}). Ignoring them and using the parsed ones."
                )
                restored = False

        if restored:
            subjects = subjects_restored
        else:
            subjects = subjects_parsed

        super().__init__(subjects, None, load_getitem=load_getitem)

        self._get_classes_key = get_classes_key
        self._get_stats_key = get_stats_key

        self._spacings = None
        self._sizes = None
        self._classes = None
        self._intensity_values = None

        if median_spacing is not None and not isinstance(median_spacing, torch.Tensor):
            median_spacing = torch.tensor(median_spacing)

        if median_size is not None and not isinstance(median_size, torch.Tensor):
            median_size = torch.tensor(median_size)
            
        self._median_spacing = median_spacing
        self._median_size = median_size
        self._median_size_after_resampling = None
        self._max_size_after_resampling = None
        self._min_size_after_resampling = None
        self._intensity_mean = None
        self._intensity_std = None
        self._class_mapping = None
        self._num_saving_procs = num_saving_procs
        self._verbose = verbose
        self._crop_to_nonzero_stats = crop_to_nonzero_stats

        self.image_modality = image_modality
        self.anisotropy_threshold = anisotropy_threshold

        if restored:
            self._restore_meta(restored_meta)

        if (
            preprocessing is not None
            and isinstance(preprocessing, str)
            and preprocessing == "default"
        ):
            preprocessing = self.default_preprocessing

        elif preprocessing is not None and not callable(preprocessing):
            raise ValueError(
                "preprocessing must be a callable, None or a string 'default'"
            )

        if (
            augmentation is not None
            and isinstance(augmentation, str)
            and augmentation == "default"
        ):
            augmentation = self.default_augmentation
        elif augmentation is not None and not callable(augmentation):
            raise ValueError(
                "augmentation must be a callable, None or a string 'default'"
            )

        transforms = []
        if preprocessing is not None and not restored:
            transforms.append(preprocessing)

        # only save when not restored, otherwise it was already saved
        if save_preprocessed and not restored:
            # set only preprocessing transforms
            self.set_transform(
                tio.transforms.Compose(transforms) if transforms else None
            )
            # save all the preprocessed files
            self._save_preprocessed(save_preprocessed)
            subjects, restored_meta = self._restore_from_preprocessed(save_preprocessed)

            # this is part of torchio's SubjectsDataset init, which has to be redone here.
            self._parse_subjects_list(subjects)
            self._subjects = subjects

            # restores the meta information
            self._restore_meta(restored_meta)
            # reset transforms to empty list, since preprocessing was already applied
            transforms = []

        if augmentation is not None:
            transforms.append(augmentation)

        if transforms:
            self.set_transform(tio.transforms.Compose(transforms))

    @abstractmethod
    def parse_subjects(self, root_path):
        raise NotImplementedError

    def _parse_stats(self, get_stats_key, get_classes_key: Optional[str]):
        # disable load_getitem for this method, since otherwise
        # the dataset may be loaded completely to memory and cached there
        prev_load_getitem = self.load_getitem
        prev_transforms = self._transform

        curr_transform = None
        if self._crop_to_nonzero_stats:
            curr_transform = CropToNonZero()
        self.set_transform(curr_transform)

        self.load_getitem = False
        if self._spacings is None:
            self._spacings = []
            add_spacings = True
        else:
            add_spacings = False

        if self._sizes is None:
            self._sizes = []
            add_sizes = True
        else:
            add_sizes = False

        if self._classes is None:
            self._classes = set()
            add_classes = True
        else:
            add_classes = False

        if self._intensity_values is None:
            self._intensity_values = Counter()
            add_intensity_values = True
        else:
            add_intensity_values = False

        # no values to add, return early
        if not any((add_spacings, add_sizes, add_classes, add_intensity_values)):
            return

        iterable = self

        if self._verbose:
            iterable = tqdm(iterable, desc="Computing Dataset Statistics")

        for sub in iterable:
            assert isinstance(sub, tio.Subject)
            new_sub = deepcopy(sub)
            if add_spacings:
                if get_stats_key is not None and get_stats_key in new_sub:
                    self._spacings.append(new_sub[get_stats_key].spacing)
                else:
                    self._spacings.append(new_sub.spacing)

            if add_sizes:
                if get_stats_key is not None and get_stats_key in new_sub:
                    self._sizes.append(new_sub[get_stats_key].spatial_shape)
                else:
                    self._sizes.append(new_sub.spatial_shape)

            if (
                add_classes
                and get_classes_key is not None
                and get_classes_key in new_sub
            ):
                if isinstance(new_sub[get_classes_key], tio.data.Image):
                    self._classes.update(
                        new_sub[get_classes_key].tensor.unqiue().tolist()
                    )
                elif isinstance(new_sub[get_classes_key], torch.Tensor):
                    self._classes.update(new_sub[get_classes_key].unique().tolist())
                else:
                    self._classes.update(list(set(new_sub[get_classes_key])))

            if (
                add_intensity_values
                and get_stats_key is not None
                and get_stats_key in new_sub
            ):
                img = new_sub[get_stats_key].tensor
                # only use foreground pixels
                uniques, counts = img[img > 0].unique(return_counts=True)
                uniques, counts = uniques.tolist(), counts.tolist()
                self._intensity_values.update(
                    {uniques[i]: counts[i] for i in range(len(uniques))}
                )

        self.load_getitem = prev_load_getitem
        self.set_transform(prev_transforms)

    @property
    def target_spacing(self):
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
            spacings_of_that_axis = self.all_spacings[:, worst_spacing_axis]
            target_spacing_of_that_axis = torch.quantile(spacings_of_that_axis, 0.1)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = (
                    max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
                )
            target_spacing[worst_spacing_axis] = target_spacing_of_that_axis

        return target_spacing

    @property
    def median_spacing(self):
        if self._median_spacing is None:
            self._median_spacing = torch.median(self.all_spacings, 0).values

        return self._median_spacing

    @median_spacing.setter
    def median_spacing(self, value):
        self._median_spacing = value

    @property
    def all_spacings(self):
        if self._spacings is None:
            self._parse_stats(self._get_stats_key, self._get_classes_key)

        spacings = self._spacings

        if not isinstance(spacings, torch.Tensor):
            spacings = torch.tensor(spacings)
        return spacings

    @property
    def all_sizes(self):
        if self._sizes is None:
            self._parse_stats(self._get_stats_key, self._get_classes_key)
        return torch.tensor(self._sizes)

    @property
    def _sizes_after_resampling(self):
        sizes_after_resampling = (
            self.all_sizes / self.all_spacings * self.median_spacing
        )
        return sizes_after_resampling

    @property
    def median_size_after_resampling(self):
        if self._median_size_after_resampling is None:
            self._median_size_after_resampling = torch.median(
                self._sizes_after_resampling, 0
            ).values

        return self._median_size_after_resampling

    @median_size_after_resampling.setter
    def median_size_after_resampling(self, value):
        self._median_size_after_resampling = value

    @property
    def median_size(self):
        if self._median_size is None:

            self._median_size = torch.median(self.all_sizes, 0).values

        return self._median_size

    @median_size.setter
    def median_size(self, value):
        self._median_size = value

    @property
    def class_mapping(self):
        if self._class_mapping is None:
            self._parse_stats(self._get_stats_key, self._get_classes_key)
            self._class_mapping = {x: i for i, x in enumerate(sorted(self._classes))}
        return self._class_mapping

    @class_mapping.setter
    def class_mapping(self, value):
        self._class_mapping = value

    @property
    def inverse_class_mapping(self):
        return {v: k for k, v in self.class_mapping.items()}

    @property
    def num_classes(self):
        return len(self.class_mapping)

    @property
    def default_preprocessing(self):
        return DefaultPreprocessing(
            num_classes=self.num_classes,
            target_spacing=self.target_spacing.tolist(),
            target_size=self.median_size.tolist(),
            modality=self.image_modality,
            dataset_intensity_mean=self.mean_intensity_value,
            dataset_intensity_std=self.std_intensity_value,
        )

    @property
    def default_augmentation(self):
        return DefaultAugmentation(self.image_modality, include_deformation=False)

    @property
    def mean_intensity_value(self) -> torch.Tensor:
        if self._intensity_mean is None:
            self._parse_stats(self._get_stats_key, self._get_classes_key)
            # normal mean calculation without allocating all the occurences to save memory
            self._intensity_mean = torch.tensor(
                sum([float(k) * float(v) for k, v in self._intensity_values.items()])
                / sum(self._intensity_values.values())
            )
        return self._intensity_mean

    @mean_intensity_value.setter
    def mean_intensity_value(self, value):
        self._intensity_mean = value

    @property
    def std_intensity_value(self) -> torch.Tensor:
        if self._intensity_std is None:
            self._parse_stats(self._get_stats_key, self._get_classes_key)
            n = sum(self._intensity_values.values())
            # normal intensity calculation without allocating all the occurences to save memory
            self._intensity_std = (
                (
                    sum(
                        [
                            float(k) * float(v) ** 2
                            for k, v in self._intensity_values.items()
                        ]
                    )
                    - n * self.mean_intensity_value ** 2
                )
                / (n - 1)
            ).sqrt()

        return self._intensity_std

    @std_intensity_value.setter
    def std_intensity_value(self, value):
        self._intensity_std = value

    @property
    def max_size_after_resampling(self):
        if self._max_size_after_resampling is None:
            self._max_size_after_resampling = torch.max(
                self._sizes_after_resampling, 0
            ).values

        return self._max_size_after_resampling

    @max_size_after_resampling.setter
    def max_size_after_resampling(self, value):
        self._max_size_after_resampling = value

    @property
    def min_size_after_resampling(self):
        if self._min_size_after_resampling is None:
            self._min_size_after_resampling = torch.min(
                self._sizes_after_resampling, 0
            ).values

        return self._min_size_after_resampling

    @min_size_after_resampling.setter
    def min_size_after_resampling(self, value):
        self._min_size_after_resampling = value

    def _save_preprocessed_single_subject(
        self, args: Tuple[int, tio.data.Subject], path: str
    ):
        # idx, sub = args
        idx = args
        sub = self[idx]

        assert isinstance(sub, tio.data.Subject)
        os.makedirs(os.path.join(path, str(idx)), exist_ok=True)
        sub_dict = {}
        for k, v in sub.get_images_dict(intensity_only=False).items():
            assert isinstance(v, tio.data.Image)

            v.save(os.path.join(path, str(idx), k + ".nii.gz"))

            if isinstance(v, tio.data.ScalarImage):
                sub_dict[k] = "ScalarImage"
            elif isinstance(v, tio.data.LabelMap):
                sub_dict[k] = "LabelMap"
            elif isinstance(v, tio.data.Image):
                sub_dict[k] = "Image"

        for k, v in sub.items():
            if not isinstance(v, tio.data.Image):
                
                sub_dict[k] = v

        with open(os.path.join(path, str(idx), "subject.json"), "w") as f:
            json.dump(sub_dict, f, indent=4, sort_keys=True, cls=PyTorchJsonEncoder)

    def _save_preprocessed(self, path: str):
        os.makedirs(path, exist_ok=True)

        func = partial(self._save_preprocessed_single_subject, path=path)

        iterable = list(range(len(self)))

        # if saving_procs is 1, there is no need for a separate process; this would only increase the overhead
        if 0 <= self.num_saving_procs <= 1:
            if self._verbose:
                iterable = tqdm(iterable, desc="Saving Preprocessed Dataset")
            for i in iterable:
                func(i)

        else:

            with torch.multiprocessing.Pool(self.num_saving_procs) as p:
                iter = p.imap(func, iterable)
                if self._verbose:
                    iter = tqdm(
                        iter, total=len(self), desc="Saving Preprocessed Dataset"
                    )

                _ = list(iter)

        with open(os.path.join(path, "dataset.json"), "w") as f:
            json.dump(
                {
                    "median_spacing": self.median_spacing.tolist(),
                    "median_size": self.median_size.tolist(),
                    "class_mapping": self.class_mapping,
                    "mean_intensity_value": self.mean_intensity_value.tolist(),
                    "std_intensity_value": self.std_intensity_value.tolist(),
                    "median_size_after_resampling": self.median_size_after_resampling.tolist(),
                    "max_size_after_resampling": self.max_size_after_resampling.tolist(),
                    "min_size_after_resampling": self.min_size_after_resampling.tolist(),
                    "all_spacings": self.all_spacings.tolist(),
                    "all_sizes": self.all_sizes.tolist(),
                    "all_classes": list(self._classes),
                    "all_intensity_values": {
                        k: v for k, v in self._intensity_values.items()
                    },
                },
                f,
                indent=4,
                sort_keys=True,
                cls=PyTorchJsonEncoder
            )

    @staticmethod
    def _restore_from_preprocessed(path):
        with open(os.path.join(path, "dataset.json"), "r") as f:
            dset_meta = json.load(f, cls=PyTorchJsonDecoder)

        subjects = []
        subs = [
            os.path.join(path, x)
            for x in os.listdir(path)
            if os.path.isdir(os.path.join(path, x))
        ]
        for sub in subs:
            # load information about subject
            with open(os.path.join(sub, "subject.json"), "r") as f:
                subject_meta = json.load(f, cls=PyTorchJsonDecoder)

            # load images and other vals
            subject = {}
            for k, v in subject_meta.items():
                if v == "ScalarImage":
                    value = tio.data.ScalarImage(os.path.join(sub, k + ".nii.gz"))
                elif v == "LabelMap":
                    value = tio.data.LabelMap(os.path.join(sub, k + ".nii.gz"))
                elif v == "Image":
                    value = tio.data.Image(os.path.join(sub, k + ".nii.gz"))
                else:
                    value = v

                if isinstance(value, tio.data.Image):
                    subject[k] = value
                else:
                    subject[k] = value
            subjects.append(tio.data.Subject(subject))

        return subjects, dset_meta

    def _restore_meta(self, loaded_meta: Mapping[str, Any]):
        self.median_spacing = torch.tensor(loaded_meta["median_spacing"])
        self.median_size = torch.tensor(loaded_meta["median_size"])
        self.class_mapping = loaded_meta["class_mapping"]
        self.mean_intensity_value = torch.tensor(loaded_meta["mean_intensity_value"])
        self.std_intensity_value = torch.tensor(loaded_meta["std_intensity_value"])
        self.median_size_after_resampling = torch.tensor(
            loaded_meta["median_size_after_resampling"]
        )
        self.max_size_after_resampling = torch.tensor(
            loaded_meta["max_size_after_resampling"]
        )
        self.min_size_after_resampling = torch.tensor(
            loaded_meta["min_size_after_resampling"]
        )
        self._spacings = torch.tensor(loaded_meta["all_spacings"])
        self._sizes = torch.tensor(loaded_meta["all_sizes"])
        self._classes = set(loaded_meta["all_classes"])
        self._intensity_values = Counter().update(loaded_meta["all_intensity_values"])

    @property
    def num_saving_procs(self) -> int:
        if self._num_saving_procs is None:
            # this is only a very rough approximation. It assumes the main memory occupation comes from images
            # and that they will all be resampled to the same size.
            # it omits the overhead of the current dataset (without the images) which may be pickled (partially) as well.
            # it also assumes the additional memory during processing to be a maximum of 1.5 times the image memory
            num_voxels = torch.prod(self.median_size_after_resampling)
            num_mem_per_image = num_voxels * 32  # images are likely to be 32-bit
            dummy_sub = self._subjects[0]
            assert isinstance(dummy_sub, tio.data.Subject)
            num_images = len(dummy_sub.get_images_names())

            total_mem_per_sub = num_mem_per_image * num_images

            available_mem = (
                psutil.virtual_memory().available + psutil.swap_memory().free
            )
            # use factor of 1.5 to factor in some memory reserves for processing. Otherwise it was crashing...
            saving_procs = min(
                max(1, int(available_mem / (total_mem_per_sub * 1.5))), len(self)
            )

            return saving_procs

        return self._num_saving_procs
