from typing import Optional, Sequence, Union
import torchio as tio
from medical_segmentation.data.modality import ImageModality
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
        norm_trafo: Optional[Callable[[tio.data.Subject], tio.data.Subject]] = None
    ):

        if modality == ImageModality.CT:
            percentiles = (0.5, 99.5)
        else:
            percentiles = (0, 100)

        if norm_trafo is None:
            norm_trafo = tio.transforms.ZNormalization()

        trafos = [
            RescaleIntensityPercentiles(percentiles=percentiles),
            NNUnetNormalization(
                target_size=target_size,
                image_modality=modality,
                dataset_mean=dataset_intensity_mean,
                dataset_std=dataset_intensity_std,
            ),
        ]
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
            tio.transforms.preprocessing.CopyAffine('data'),
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
