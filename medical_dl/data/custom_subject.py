import torchio as tio
import numpy as np
from typing import Optional
import pprint

# backport of https://github.com/fepegar/torchio/pull/592/
class CustomSubject(tio.data.Subject):
    def check_consistent_attribute(
        self,
        attribute: str,
        relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-5,
        message: Optional[str] = None,
    ) -> None:
        """Checks for consistency of an attribute across all images in the current subject.

        Args:
            attribute: The name of the image attribute to check
            relative_tolerance: The relative tolerance for the consistency check (see formula below)
            absolute_tolerance: The absolute tolerance for the consistency check (see formula below)
            message: The error message to be raised if attributes are not consistent.

        Example:
            >>> import numpy as np
            >>> import torch
            >>> import torchio as tio
            >>> img = torch.randn(1, 512, 512, 100)
            >>> mask = torch.tensor(img > 0).type(torch.int16)
            >>> af1 = np.array([[0.8, 0, 0, 0],
            ...                 [0, 0.8, 0, 0],
            ...                 [0, 0, 2.50000000000001, 0],
            ...                 [0, 0, 0, 1]])
            >>> af2 = np.array([[0.8, 0, 0, 0],
            ...                 [0, 0.8, 0, 0],
            ...                 [0, 0, 2.49999999999999, 0], # small difference here (e.g. due to different reader)
            ...                 [0, 0, 0, 1]])
            >>> sub = tio.Subject(
            ...   image = tio.ScalarImage(tensor = img, affine = af1),
            ...   mask = tio.LabelMap(tensor = mask, affine = af2)
            ... )
            >>> sub.check_consistent_attribute('spacing') # passes due to introduced tolerances



        Note:
            As stated in the numpy docs, this computes the following:
                absolute(a - b) <= (absolute_tolerance + relative_tolerance * absolute(b))
            with a beeing the attribute of the first image and b being the attributes of all
            other images respectively.
        """
        iterable = self.get_images_dict(intensity_only=False).items()
        if message is None:
            if self.get_first_image().path is not None:
                infix = str(self.get_first_image().path)
            else:
                infix = ''
            message = (
                f"More than one value for {attribute} found in {infix} subject images:" "\n{}"
            )
        try:
            first_attr = None
            first_image = None

            for image_name, image in iterable:
                if first_attr is None:
                    first_attr = getattr(image, attribute)
                    first_image = image_name

                else:
                    curr_attr = getattr(image, attribute)
                    if not np.allclose(
                        curr_attr,
                        first_attr,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                    ):
                        message = message.format(
                            pprint.pformat(
                                {first_image: first_attr, image_name: curr_attr}
                            )
                        )
                        raise RuntimeError(message)

        except TypeError:
            # fallback for non-numeric values
            values_dict = {}
            for image_name, image in iterable:
                values_dict[image_name] = getattr(image, attribute)
            num_unique_values = len(set(values_dict.values()))
            if num_unique_values > 1:
                message = message.format(pprint.pformat(values_dict))
                raise RuntimeError(message)
