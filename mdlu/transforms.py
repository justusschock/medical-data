from typing import Callable, Optional, Sequence, Tuple, Union
from SimpleITK.SimpleITK import Crop
import torchio as tio
from mdlu.data.modality import ImageModality
import torch
from torchio.transforms.preprocessing.spatial.copy_affine import CopyAffine
from torchio.transforms.preprocessing.spatial.resample import Resample


class DefaultSpatialPreIntensityPreprocessingTransforms(tio.transforms.Compose):
    def __init__(
        self,
        num_classes,
        target_spacing,
        interpolation: str = "linear",
    ):
        trafos = [
            CropToNonZero(),
            tio.transforms.ToCanonical(),
            ResampleOnehot(
                num_classes=num_classes,
                target=target_spacing,
                image_interpolation=interpolation,
            ),
        ]

        super().__init__(trafos)


class DefaultSpatialPostIntensityPreprocessingTransforms(tio.transforms.Compose):
    def __init__(self, target_size: torch.Tensor):
        trafos = [tio.transforms.CropOrPad(target_shape=target_size)]
        super().__init__(trafos)


class DefaultIntensityPreprocessing(tio.transforms.Compose):
    def __init__(
        self,
        target_size,
        modality: Optional[Union[int, ImageModality]] = None,
        dataset_intensity_mean: Optional[torch.Tensor] = None,
        dataset_intensity_std: Optional[torch.Tensor] = None,
        norm_trafo: Optional[Callable[[tio.data.Subject], tio.data.Subject]] = None,
    ):

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
    def __init__(
        self,
        num_classes: int,
        target_spacing: Sequence[float],
        target_size: Sequence[int],
        modality: Optional[Union[int, ImageModality]] = None,
        dataset_intensity_mean: Optional[torch.Tensor] = None,
        dataset_intensity_std: Optional[torch.Tensor] = None,
        interpolation: str = "linear",
    ):
        trafos = [
            tio.transforms.preprocessing.CopyAffine("data"),
            DefaultSpatialPreIntensityPreprocessingTransforms(
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
    def __init__(self, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_onehot_trafo = tio.transforms.OneHot(num_classes=num_classes)
        self.to_categorical = LabelToCategorical()

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        subject = self.to_onehot_trafo(subject)
        temp_scalar_images = []

        for k, v in subject.get_images_dict(intensity_only=False).items():
            if not isinstance(v, tio.data.LabelMap):
                continue
            subject[k] = tio.data.ScalarImage(tensor=v.data, affine=v.affine)
            temp_scalar_images.append(k)
        subject = super().apply_transform(subject)
        for k in temp_scalar_images:
            subject[k] = tio.data.LabelMap(
                tensor=subject[k].data, affine=subject[k].affine
            )

        subject = self.to_categorical(subject)
        return subject


class CropToNonZero(tio.transforms.Transform):
    @staticmethod
    def crop_to_nonzero(
        image: tio.data.Image, *additional_images: tio.data.Image, **padding_kwargs
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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
        import nibabel as nib

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
        intensity_images = subject.get_images_dict(intensity_only=True)
        all_images = subject.get_images_dict(intensity_only=True)

        seg_images = {k: v for k, v in all_images.items() if k not in intensity_images}

        for v in intensity_images.values():
            self.crop_to_nonzero(v, *seg_images.values())

        return subject


class NNUnetNormalization(tio.transforms.ZNormalization):
    def __init__(
        self,
        target_size: Sequence[int],
        image_modality: Optional[Union[int, ImageModality]] = None,
        dataset_mean: Optional[torch.Tensor] = None,
        dataset_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if isinstance(image_modality, int):
            image_modality = ImageModality(image_modality)
        self.image_modality = image_modality
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.num_target_elems = torch.tensor(target_size).prod()
        self.crop_or_pad_trafo = tio.transforms.CropOrPad(target_shape=target_size)

    def apply_normalization(
        self, subject: tio.data.Subject, image_name: str, mask: torch.Tensor
    ) -> None:

        # no image modality is given, use default (non-CT) behavior
        if self.image_modality is None or self.image_modality != ImageModality.CT:
            # not more than 25% of the voxels are cropped away
            if (
                torch.tensor(subject.spatial_shape).prod() * 0.75
                < self.num_target_elems
            ):
                return super().apply_normalization(subject, image_name, mask)

            cropped_tensor = self.crop_or_pad_trafo(
                tio.data.Subject(
                    {image_name: tio.data.ScalarImage(subject[image_name].data)}
                )
            )[image_name].data
            # masking for foreground
            mask = cropped_tensor > 0
            values = cropped_tensor[mask]
            mean = values.mean()
            std = values.std()

        else:
            if self.dataset_mean is None or self.dataset_std is None:
                raise ValueError(
                    "dataset mean and std can only be None if the dataset modality is not CT."
                )

            mean = self.dataset_mean
            std = self.dataset_std

        if std == 0:
            mean = self.dataset_mean
            std = self.dataset_std

        image = subject[image_name]
        tensor = image.data
        tensor -= mean
        tensor /= std
        image.set_data(tensor)


class LabelToCategorical(
    tio.transforms.preprocessing.label.label_transform.LabelTransform
):
    def apply_transform(self, subject):
        for k, v in self.get_images_dict(subject).items():
            if isinstance(v, tio.data.LabelMap):
                v.set_data(v.data.argmax(0).unsqueeze(0))
        return subject


class RescaleIntensityPercentiles(tio.transforms.RescaleIntensity):
    def __init__(self, percentiles):
        super().__init__(percentiles=percentiles)

    def rescale(self, tensor: torch.Tensor, mask: torch.Tensor, image_name: str):
        if self.percentiles == (0, 100):
            return tensor
        else:
            self.out_max = tensor.max().item()
            self.out_min = tensor.min().item()
            return super().rescale(tensor, mask, image_name)


class DefaultSpatialAugmentation(tio.transforms.Compose):
    def __init__(
        self, image_modality: ImageModality, include_deformation: bool = False, p=0.25
    ):
        trafos = []
        if image_modality is not None:
            trafos += [
                tio.transforms.RandomAnisotropy(
                    axes=tuple(range(ImageModality.get_dimensionality(image_modality))),
                    p=p,
                ),
                tio.transforms.RandomFlip(
                    tuple(range(ImageModality.get_dimensionality(image_modality))), p=p
                ),
            ]
        trafos.append(tio.transforms.RandomAffine(p=p)),

        if include_deformation:
            trafos.append(tio.transforms.RandomElasticDeformation(p=p))

        super().__init__(trafos)


class DefaultIntensityAugmentation(tio.transforms.Compose):
    def __init__(self, image_modality: ImageModality, p=0.25):
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
    def __init__(
        self,
        image_modality: Optional[ImageModality],
        include_deformation: bool = False,
        spatial_prob: float = 0.25,
        intensity_prob: float = 0.5,
    ):
        trafos = [
            DefaultSpatialAugmentation(
                image_modality=image_modality,
                include_deformation=include_deformation,
                p=spatial_prob,
            ),
            DefaultIntensityAugmentation(
                image_modality=image_modality, p=intensity_prob
            ),
        ]
        super().__init__(trafos)


class Transform3DTo2D(tio.transforms.Transform):
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        for k, v in subject.get_images_dict.items():
            tensor = v.data
            orig_shape = v.data.shape
            tensor = tensor.view(-1, orig_shape[1], orig_shape[2], 1)
            v.set_data(tensor)
            subject[f"_{k}_orig_shape"] = orig_shape
        return subject


class Transform2DBackTo3D(tio.transforms.Transform):
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        for k, v in subject.get_images_dict.items():
            tensor = v.data
            orig_shape = subject[f"_{k}_orig_shape"]
            tensor = tensor.view(orig_shape)
            v.set_data(tensor)
            subject.pop(f"_{k}_orig_shape")
        return subject


class CopySpacingTransform(tio.transforms.Transform):
    def __init__(self, origin: str = "data", **kwargs):
        super().__init__(**kwargs)
        self.origin = origin

    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        target_spacing = subject[self.origin].spacing
        for v in subject.get_images_dict(intensity_only=False).values():
            v.spacing = target_spacing


import torch
from torch.nn import functional as F


def extract_nonzero_bounding_box_from_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

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
        ranges[idx, :] = (
            torch.ceil((bounds[:, 1] - bounds[:, 0]).float() / 2) * 2
        ).long()[
            1:
        ]  # extract max range of all channels
        centers[idx, :] = bounds[1:, 0] + ranges[idx] // 2

    return centers, ranges


def crop_tensor_to_bounding_box(
    tensor: torch.Tensor,
    centers: torch.Tensor,
    ranges: torch.Tensor,
    additional_tensor: Optional[torch.Tensor] = None,
    **padding_kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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
    first_paddings = torch.where(
        abs_min_range < 0, abs_min_range.abs(), torch.zeros_like(abs_min_range)
    )

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
                additional_out[idx] = additional_tensor[idx][
                    ..., mins[0] : maxs[0], mins[1] : maxs[1]
                ]
        elif len(mins) == 3:
            out[idx] = tensor[idx][
                ..., mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]
            ]
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
    additional_tensor: Optional[torch.Tensor] = None,
    **padding_kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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
