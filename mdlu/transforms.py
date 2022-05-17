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

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import nibabel as nib
import torch
import torchio as tio
from torch.nn import functional as F
from torchio.transforms.preprocessing.spatial.resample import Resample

from mdlu.data.modality import ImageModality

__all__ = [
    "DefaultPreprocessing",
    "ResampleOnehot",
    "ResampleAndCropOrPad",
    "CropToNonZero",
    "NNUnetNormalization",
    "RescaleIntensityPercentiles",
    "DefaultSpatialAugmentation",
    "CropOrPadPerImage",
]


class DefaultSpatialPreIntensityPreprocessingTransforms(tio.transforms.Compose):
    """The default spatial preprocessing to be running BEFORE the intensity preprocessing.

    Can optionally (by default) include a Cropping to nonzero values, a resampling to the target spacing (for masks
    either in one-hot format or using the nearest neighbor resampling).
    """

    def __init__(
        self,
        resample_onehot: bool,
        target_spacing,
        interpolation: str = "linear",
        crop_to_nonzero: str | bool | None = False,
        num_classes: int | None = None,
    ) -> None:
        """
        Args:
            resample_onehot: Whether to resample masks to the target spacing in onme-hot using the same
                interpolation scheme as for images combined with an argmax or using the nearest neighbor approach.
            target_spacing: The target spacing to resample to.
            interpolation: The interpolation scheme to use when resampling.
            crop_to_nonzero: Whether to crop to nonzero values.
            num_classes: The number of classes to use when resampling masks.

        """

        trafos = []

        if crop_to_nonzero or crop_to_nonzero is None:
            if crop_to_nonzero is not None and not isinstance(crop_to_nonzero, str):
                raise TypeError(
                    "crop_to_nonzero must be a string or None or a value evaluating to False to avoid cropping at all"
                )
            trafos.append(CropToNonZero(crop_to_nonzero))

        if resample_onehot:
            assert num_classes is not None
            trafos.append(
                ResampleOnehot(
                    num_classes=num_classes,
                    target=target_spacing,
                    image_interpolation=interpolation,
                )
            )
        else:
            trafos.append(Resample(target=target_spacing, image_interpolation=interpolation))

        super().__init__(trafos)


class DefaultSpatialPostIntensityPreprocessingTransforms(tio.transforms.Compose):
    """The default spatial preprocessing to be running AFTER the intensity preprocessing.

    Currently only consists of cropping or padding to a given target size
    """

    def __init__(self, target_size: torch.Tensor) -> None:
        """
        Args:
            target_size: The target size to crop or pad to.
        """
        trafos = [CropOrPadPerImage(target_shape=target_size)]
        super().__init__(trafos)


class DefaultIntensityPreprocessing(tio.transforms.Compose):
    """The default intensity preprocessing.

    Includes a rescaling of intensity values to certain percentiles as well as a normalization transform.
    """

    def __init__(
        self,
        target_size,
        modality: int | ImageModality | None = None,
        dataset_intensity_mean: torch.Tensor | None = None,
        dataset_intensity_std: torch.Tensor | None = None,
        norm_trafo: Callable[[tio.data.Subject], tio.data.Subject] | None = None,
    ) -> None:
        """
        Args:
            target_size: The target size to crop or pad to.
            modality: The modality to use.
            dataset_intensity_mean: The mean to use for the intensity normalization.
            dataset_intensity_std: The standard deviation to use for the intensity normalization.
            norm_trafo: A function to apply to the subject after normalization.
        """

        if modality == ImageModality.CT:
            percentiles = (0.5, 99.5)
        else:
            percentiles = (0, 100)

        if norm_trafo is None:
            norm_trafo = NNUnetNormalization(
                target_size=target_size,
                image_modality=modality,
                dataset_mean=dataset_intensity_mean,
                dataset_std=dataset_intensity_std,
            )

        trafos = [RescaleIntensityPercentiles(percentiles=percentiles), norm_trafo]
        super().__init__(trafos)


class DefaultPreprocessing(tio.transforms.Compose):
    """The default preprocessing. Consists of the following tranforms (in Order):

    - Transform all images and label images to a canonical RAS+ orientation
    - Copy Affine from image to others (like labels). There can be inaccuracies using different loaders or annotation tools
        which could cause errors downstream if not unifying now.
    - Cropping to nonzero values (optional)
    - Resampling to the target spacing (for masks: either one-hot or nearest neighbor)
    - Rescaling of intensity values to certain percentiles
    - Normalization of intensity values
    - Cropping or padding to target size
    """

    def __init__(
        self,
        target_spacing: Sequence[float],
        target_size: Sequence[int],
        modality: int | ImageModality | None = None,
        dataset_intensity_mean: torch.Tensor | None = None,
        dataset_intensity_std: torch.Tensor | None = None,
        interpolation: str = "linear",
        affine_key: None | str = "data",
        num_classes: int | None = None,
    ):
        """
        Args:
            target_spacing: The target spacing to resample to.
            target_size: The target size to crop or pad to.
            modality: The modality to use.
            dataset_intensity_mean: The mean to use for the intensity normalization.
            dataset_intensity_std: The standard deviation to use for the intensity normalization.
            interpolation: The interpolation scheme to use when resampling.
            affine_key: The key to use for copying the affine from.
            num_classes: The number of classes to use when resampling masks.

        """
        trafos = [
            tio.transforms.ToCanonical(),
        ]

        if affine_key is not None:
            trafos.append(tio.transforms.preprocessing.CopyAffine(target=affine_key))

        trafos += [
            DefaultSpatialPreIntensityPreprocessingTransforms(
                resample_onehot=True,
                num_classes=num_classes,
                target_spacing=target_spacing,
                interpolation=interpolation,
            ),
            DefaultIntensityPreprocessing(
                modality=modality,
                target_size=target_size,
                dataset_intensity_mean=dataset_intensity_mean,
                dataset_intensity_std=dataset_intensity_std,
            ),
            DefaultSpatialPostIntensityPreprocessingTransforms(target_size=target_size),
        ]

        return super().__init__(trafos)


class ResampleOnehot(Resample):
    """Resample a mask to the target spacing using one-hot encoding and same interpolation scheme as for images.

    .. note::
        For additional arguments have a look at :class:`torchio.transforms.preprocessing.spatial.resample.Resample`.
    """

    def __init__(self, num_classes: int, *args: Any, **kwargs: Any) -> None:
        """
        Args:
            num_classes: The number of classes to use when resampling masks.
        """
        super().__init__(*args, **kwargs)
        self.to_onehot_trafo = tio.transforms.OneHot(num_classes=num_classes)
        self.to_categorical = LabelToCategorical()

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        """Applies the resampling to the whole subject.

        Args:
            subject: The subject to apply the transform to.

        Returns:
            The transformed subject.
        """
        subject = self.to_onehot_trafo(subject)
        temp_scalar_images = []

        for k, v in subject.get_images_dict(intensity_only=False).items():
            if not isinstance(v, tio.data.LabelMap):
                continue
            subject[k] = tio.data.ScalarImage(tensor=v.data, affine=v.affine)
            temp_scalar_images.append(k)
        subject = super().apply_transform(subject)
        for k in temp_scalar_images:
            subject[k] = tio.data.LabelMap(tensor=subject[k].data, affine=subject[k].affine)

        subject = self.to_categorical(subject)
        return subject


class ResampleAndCropOrPad(tio.transforms.Transform):
    """Resampling to a custom target spacing and cropping or padding to a custom target size.

    Handles each image individually since depending on initial shape and spacing the in-between results may differ.
    """

    def __init__(
        self,
        target_spacing: tuple[float, float, float],
        target_size: tuple[int, int, int],
        num_classes: int,
        interpolation: str = "linear",
        **kwargs,
    ) -> None:
        """
        Args:
            target_spacing: The target spacing to resample to.
            target_size: The target size to crop or pad to.
            num_classes: The number of classes to use when resampling masks.
            interpolation: The interpolation scheme to use when resampling.
        """
        super().__init__(**kwargs)

        self._resample = ResampleOnehot(
            num_classes,
            target_spacing,
            interpolation=interpolation,
            p=self.probability,
            copy=self.copy,
            include=self.include,
            exclude=self.exclude,
            keys=None,
            keep=self.keep,
        )
        self._crop_or_pad = tio.transforms.CropOrPad(
            target_size,
            p=self.probability,
            copy=self.copy,
            include=self.include,
            exclude=self.exclude,
            keys=None,
            keep=self.keep,
        )

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        """Applies the resampling and cropping or padding to the whole subject.

        Args:
            subject: The subject to apply the transform to.

        Returns:
            The transformed subject.
        """

        # handle each image in subject individually to compensate
        # for misalignment of images in checks inbetween the steps!
        for k, v in subject.get_images_dict(intensity_only=False).items():
            curr_sub = tio.data.Subject(k=v)
            curr_sub = self._resample(curr_sub)
            curr_sub = self._crop_or_pad(curr_sub)

            subject[k] = curr_sub[k]
        return subject


class CropToNonZero(tio.transforms.Transform):
    """Crops the image to a bounding box containing all nonzero voxels."""

    def __init__(self, key: str | None = None, **kwargs: Any) -> None:
        """
        Args:
            key: The key specifying the image to use.
        """
        super().__init__(**kwargs)
        self.key = key

    @staticmethod
    def crop_to_nonzero(image: tio.data.Image, *additional_images: tio.data.Image, **padding_kwargs) -> None:
        """crops tensor to non-zero and additional tensor to the same part of the tensor if given.

        Args:
            tensor: the tensor to crop
            additional_tensor: an additional tensor to crop to the same region
                as :attr`tensor`
            **padding_kwargs: keyword arguments to controll the necessary padding
        """

        centers, ranges = extract_nonzero_bounding_box_from_tensor(image.tensor[None])

        low = (centers - ranges / 2)[0].cpu().detach().numpy()

        for img in (image, *additional_images):
            result = crop_tensor_to_bounding_box(
                img.tensor[None],
                centers=centers,
                ranges=ranges,
                **padding_kwargs,
            )

            img.set_data(result[0])

            img_affine = img.affine.copy()
            new_origin = nib.affines.apply_affine(img.affine, low)
            img_affine[:3, 3] = new_origin
            img.affine = img_affine

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        """Applies the transform to the whole subject.

        Args:
            subject: The subject to apply the transform to.

        Returns:
            The transformed subject.
        """
        intensity_images = subject.get_images_dict(intensity_only=True)
        all_images = subject.get_images_dict(intensity_only=False)

        if self.key is None:
            images = intensity_images
        else:
            images = {self.key: subject[self.key]}

        assert (
            len(images) == 1
        ), "Multiple images found, this may lead to inconsistent cropping behavior. Pleas specify a reference image key"
        # TODO: Add reference image key to default trafos and dataset

        seg_images = {k: v for k, v in all_images.items() if k not in intensity_images}
        for v in images.values():
            self.crop_to_nonzero(v, *seg_images.values())

        return subject


class NNUnetNormalization(tio.transforms.ZNormalization):
    """Normalization similar to the one prposed in the nnUNet paper.

    - If the image is not a CT and not more than 25% of the image are cropped away,
        use the usual Z-Normalization across the whole image.
    - If the image is not a CT and more than 25% of the iamge are cropped away,
        extract the mean and standard deviation of from the non-cropped parts of the image only and
        perform Z-Normalization using these values.
    - If the image is a CT, use the passed mean and standard deviation values computed over the whole dataset
        and use those for Z-Normalization.
    """

    def __init__(
        self,
        target_size: Sequence[int],
        image_modality: int | ImageModality | None = None,
        dataset_mean: torch.Tensor | None = None,
        dataset_std: torch.Tensor | None = None,
    ):
        """
        Args:
            target_size: The target size to crop or pad to.
            image_modality: The image modality to use.
            dataset_mean: The mean to use for the dataset.
            dataset_std: The standard deviation to use for the dataset.
        """
        super().__init__()

        if isinstance(image_modality, int):
            image_modality = ImageModality(image_modality)
        self.image_modality = image_modality
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.num_target_elems = torch.tensor(target_size).prod()
        self.crop_or_pad_trafo = CropOrPadPerImage(target_shape=target_size)

    def apply_normalization(self, subject: tio.data.Subject, image_name: str, mask: torch.Tensor) -> None:
        """Applies the normalization to the image.

        Args:
            subject: The subject to apply the normalization to.
            image_name: The name of the image to apply the normalization to.
            mask: The mask to use for the normalization.
        """

        # no image modality is given, use default (non-CT) behavior
        if self.image_modality is None or self.image_modality != ImageModality.CT:
            # not more than 25% of the voxels are cropped away
            if torch.tensor(subject.spatial_shape).prod() * 0.75 < self.num_target_elems and subject[image_name].tensor[mask].float().std() != 0:
                return super().apply_normalization(subject, image_name, mask)

            cropped_tensor = self.crop_or_pad_trafo(
                tio.data.Subject({image_name: tio.data.ScalarImage(tensor=subject[image_name].data)})
            )[image_name].data
            # masking for foreground
            mask = cropped_tensor > 0
            values = cropped_tensor[mask].float()
            mean = values.mean()
            std = values.std()

        else:
            if self.dataset_mean is None or self.dataset_std is None:
                raise ValueError("dataset mean and std can only be None if the dataset modality is not CT.")

            mean = self.dataset_mean
            std = self.dataset_std

        if std == 0:
            mean = self.dataset_mean
            std = self.dataset_std

        image = subject[image_name]
        tensor = image.data.to(mean)
        tensor -= mean
        tensor /= std
        image.set_data(tensor)


class LabelToCategorical(tio.transforms.preprocessing.label.label_transform.LabelTransform):
    """Converts a One-Hot encoded label to a categorical label."""

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        """Applies the transform to the whole subject.

        Args:
            subject: The subject to apply the transform to.

        Returns:
            The transformed subject.
        """

        for k, v in self.get_images_dict(subject).items():
            if isinstance(v, tio.data.LabelMap):
                v.set_data(v.data.argmax(0).unsqueeze(0))
        return subject


class RescaleIntensityPercentiles(tio.transforms.RescaleIntensity):
    """Rescales the intensity of the image to the given percentiles."""

    def rescale(self, tensor: torch.Tensor, mask: torch.Tensor, image_name: str) -> torch.Tensor:
        """Rescales the intensity of the image to the given percentiles.

        Args:
            tensor: The tensor to rescale.
            mask: The mask to use for the rescaling.
            image_name: The name of the image to rescale.
        """
        if self.percentiles == (0, 100):
            return tensor
        else:
            self.out_max = tensor.max().item()
            self.out_min = tensor.min().item()
            return super().rescale(tensor, mask, image_name)


class DefaultSpatialAugmentation(tio.transforms.Compose):
    """Default spatial augmentation.

    - Approximates a random anisotropy
    - Randomly flips the image
    - Applies a random affine transformation
    - Optionally includes a random elastic deformation
    """

    def __init__(self, image_modality: ImageModality | int, include_deformation: bool = False, p: float = 0.25) -> None:
        """
        Args:
            image_modality: The image modality to use.
            include_deformation: Whether to include elastic deformation.
            p: The probability of applying every single transformation.
        """
        trafos = []
        if image_modality is not None:
            trafos += [
                tio.transforms.RandomAnisotropy(
                    axes=tuple(range(ImageModality.get_dimensionality(image_modality))),
                    p=p,
                ),
                tio.transforms.RandomFlip(tuple(range(ImageModality.get_dimensionality(image_modality))), p=p),
            ]

        trafos.append(tio.transforms.RandomAffine(p=p))

        if include_deformation:
            trafos.append(tio.transforms.RandomElasticDeformation(p=p))

        super().__init__(trafos)


class DefaultIntensityAugmentation(tio.transforms.Compose):
    """Default intensity augmentation.

    - Random Ghosting Artifacts (for MRI only)
    - Random Spiking Artifacts (for MRI only)
    - Random Motion Artifacts
    - Random Bias Fields
    - Random Gamma Transform
    - Random Blurring
    - Random Noise
    - Random Swapping of Patches
    """

    def __init__(self, image_modality: ImageModality | int, p: float = 0.25) -> None:
        """
        Args:
            image_modality: The image modality to use.
            p: The probability of applying every single transformation.

        """
        trafos = []

        if image_modality is not None and image_modality == ImageModality.MR:
            trafos.append(tio.transforms.RandomGhosting(p=p))
            trafos.append(tio.transforms.RandomSpike(p=p))

        trafos += [
            tio.transforms.RandomMotion(p=p),
            tio.transforms.RandomBiasField(p=p),
            tio.transforms.RandomGamma(p=p),
            tio.transforms.RandomBlur(p=p),
            tio.transforms.RandomNoise(p=p),
            tio.transforms.RandomSwap(p=p),
        ]
        super().__init__(trafos)


class DefaultAugmentation(tio.transforms.Compose):
    """Default augmentation.

    - Approximates a random anisotropy
    - Randomly flips the image
    - Applies a random affine transformation
    - Optionally includes a random elastic deformatio
    - Random Ghosting Artifacts (for MRI only)
    - Random Spiking Artifacts (for MRI only)
    - Random Motion Artifacts
    - Random Bias Fields
    - Random Gamma Transform
    - Random Blurring
    - Random Noise
    - Random Swapping of Patches
    """

    def __init__(
        self,
        image_modality: ImageModality | int,
        include_deformation: bool = False,
        spatial_prob: float = 0.25,
        intensity_prob: float = 0.5,
    ) -> None:
        """
        Args:
            image_modality: The image modality to use.
            include_deformation: Whether to include elastic deformation.
            spatial_prob: The probability of applying every single spatial transformation.
            intensity_prob: The probability of applying every single intensity transformation.
        """
        trafos = [
            DefaultSpatialAugmentation(
                image_modality=image_modality,
                include_deformation=include_deformation,
                p=spatial_prob,
            ),
            DefaultIntensityAugmentation(image_modality=image_modality, p=intensity_prob),
        ]
        super().__init__(trafos)


class Transform3DTo2D(tio.transforms.Transform):
    """Transforms a 3D image to a 2D image with the number of channels equal to the depth beforehand."""

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        """Applies the transform to the whole subject.

        Args:
            subject: The subject to apply the transform to.

        Returns:
            The transformed subject.
        """
        for k, v in subject.get_images_dict.items():
            tensor = v.data
            orig_shape = v.data.shape
            tensor = tensor.view(orig_shape[3] * orig_shape[0], orig_shape[1], orig_shape[2], 1)
            v.set_data(tensor)
            subject[f"_{k}_orig_shape"] = orig_shape
        return subject


class Transform2DBackTo3D(tio.transforms.Transform):
    """Transforms a 2D image back to a 3D image with the original shape restored."""

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        """Applies the transform to the whole subject.

        Args:
            subject: The subject to apply the transform to.

        Returns:
            The transformed subject.
        """
        for k, v in subject.get_images_dict.items():
            tensor = v.data
            orig_shape = subject[f"_{k}_orig_shape"]
            tensor = tensor.view(orig_shape)
            v.set_data(tensor)
            subject.pop(f"_{k}_orig_shape")
        return subject


def extract_nonzero_bounding_box_from_tensor(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts the bounding box of nonzero elements from a tensor.

    Args:
        tensor: The tensor to extract the bounding box from.

    Returns:
        The bounding box of nonzero elements (as centers and ranges).
    """

    # create empty ranges
    ranges = torch.zeros(
        tensor.size(0),
        tensor.ndim - 2,
        dtype=torch.long,
        device=tensor.device,
    )
    # create empty centers
    centers = torch.zeros(
        tensor.size(0),
        tensor.ndim - 2,
        dtype=torch.long,
        device=tensor.device,
    )
    for idx, _tensor in enumerate(tensor):
        bounds = torch.tensor(
            [(tmp.min(), tmp.max()) for tmp in _tensor.nonzero(as_tuple=True)],
            device=tensor.device,
        )
        # ensure this is an even number
        ranges[idx, :] = (torch.ceil((bounds[:, 1] - bounds[:, 0]).float() / 2) * 2).long()[
            1:
        ]  # extract max range of all channels
        centers[idx, :] = bounds[1:, 0] + ranges[idx] // 2

    return centers, ranges


def crop_tensor_to_bounding_box(
    tensor: torch.Tensor,
    centers: torch.Tensor,
    ranges: torch.Tensor,
    additional_tensor: torch.Tensor | None = None,
    **padding_kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Crops a tensor to a bounding box.

    Args:
        tensor: The tensor to crop.
        centers: The centers of the bounding box.
        ranges: The ranges of the bounding box.
        additional_tensor: An additional tensor to crop with the same bounbding box.
        **padding_kwargs: Additional arguments to pass to the padding function.

    Returns:
        The cropped tensor optionally (if passed) along with the cropped additional tensor.
    """
    max_range = ranges.max(0)[0]
    out = torch.zeros(
        tensor.size(0),
        tensor.size(1),
        *max_range.tolist(),
        device=tensor.device,
        dtype=tensor.dtype,
    )

    if additional_tensor is not None:
        additional_out = torch.zeros(
            additional_tensor.size(0),
            additional_tensor.size(1),
            *max_range.tolist(),
            device=additional_tensor.device,
            dtype=additional_tensor.dtype,
        )

    # calciulate the absolute minimum range
    abs_min_range = (centers - max_range / 2.0).min(0)[0]

    # calculate padding on the first side per dim
    first_paddings = torch.where(abs_min_range < 0, abs_min_range.abs(), torch.zeros_like(abs_min_range))

    # calculate padding on the last side per dim
    last_paddings = torch.clamp(
        (centers + max_range / 2.0).max(0)[0]
        - torch.tensor(tensor.shape[2:], device=centers.device, dtype=centers.dtype),
        0,
    )

    # combine paddings
    total_paddings = []

    for idx in range(len(first_paddings)):
        total_paddings += [first_paddings[idx], last_paddings[idx]]

    total_paddings = torch.stack(total_paddings).long().tolist()

    # add padding offset to centers
    centers = centers + first_paddings[None]

    # pad tensor
    tensor = F.pad(tensor, total_paddings, **padding_kwargs)

    # pad additional tensor if given
    if additional_tensor is not None:
        additional_tensor = F.pad(additional_tensor, total_paddings, **padding_kwargs)

    # re-assign tensors
    for idx in range(out.size(0)):
        mins = (centers[idx] - max_range / 2.0).long()
        maxs = (centers[idx] + max_range / 2.0).long()
        if len(mins) == 2:
            out[idx] = tensor[idx][..., mins[0] : maxs[0], mins[1] : maxs[1]]
            if additional_tensor is not None:
                additional_out[idx] = additional_tensor[idx][..., mins[0] : maxs[0], mins[1] : maxs[1]]
        elif len(mins) == 3:
            out[idx] = tensor[idx][..., mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]
            if additional_tensor is not None:
                additional_out[idx] = additional_tensor[idx][
                    ..., mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]
                ]
    # remove channel dim for mask_out if mask had no channel dim
    if additional_tensor is not None and additional_tensor.ndim == tensor.ndim - 1:
        additional_out = additional_out[:, 0]

    if additional_tensor is None:
        return out
    return out, additional_out


def crop_tensor_to_nonzero(
    tensor: torch.Tensor,
    additional_tensor: torch.Tensor | None = None,
    **padding_kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    crops tensor to non-zero and additional tensor to the same part of the
    tensor if given
    Args:
        tensor: the tensor to crop
        additional_tensor: an additional tensor to crop to the same region
            as :attr`tensor`
        **padding_kwargs: keyword arguments to controll the necessary padding
    Returns:
        torch.Tensor: the cropped tensor
        Optional[torch.Tensor]: the cropped additional tensor,
            only returned if passed
    """
    centers, ranges = extract_nonzero_bounding_box_from_tensor(tensor)

    return crop_tensor_to_bounding_box(
        tensor,
        centers=centers,
        ranges=ranges,
        additional_tensor=additional_tensor,
        **padding_kwargs,
    )


class CropOrPadPerImage(tio.transforms.CropOrPad):
    """Crops or pads every single image individually to the given shape."""

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        """
        Args:
            subject: the subject to apply the transform to

        Returns:
            the subject with the transformed data

        """
        for k, v in subject.get_images_dict(intensity_only=False, include=self.include, exclude=self.exclude).items():
            part_sub = type(subject)({k: v})
            subject[k] = super().apply_transform(part_sub)[k]
        return subject


# TODO: Add Transform to cut only one side of the image and flip it.
