import os
import pytest
import torch
import torchio as tio
import random
import SimpleITK as sitk

@pytest.fixture
def dummy_dataset(tmpdir):
    image_dir = os.path.join(tmpdir, 'images')
    label_dir = os.path.join(tmpdir, 'labels')
    image_writer = sitk.ImageFileWriter()

    for i in range(10):
        patient_id = str(random.randint(1000000000, 9999999999))
        patient_dir_images = os.path.join(image_dir, patient_id)
        patient_dir_labels = os.path.join(label_dir, patient_id)

        study_id = str(random.randint(1000000000, 9999999999))
        study_dir_images = os.path.join(patient_dir_images, study_id)
        study_dir_labels = os.path.join(patient_dir_labels, study_id)

        series_id = str(random.randint(1000000000, 9999999999))
        series_dir_images = os.path.join(study_dir_images, series_id)
        series_file_labels = os.path.join(study_dir_labels, series_id)
        affine=torch.eye(4)@torch.rand(4, 4)

        subject = tio.data.Subject(data=tio.data.ScalarImage(tensor=torch.randint(0, 255, (1, 20, 30, 40), dtype=torch.uint8), affine=affine.numpy()), label=tio.data.LabelMap(tensor=torch.randint(0, 5, (1, 20, 30, 40)), affine=affine.numpy()))
        sitk_image = subject['data'].as_sitk()

        os.makedirs(series_dir_images, exist_ok=True)
        for i in range(sitk_image.GetDepth()):
            image_slice = sitk_image[:,:,i]

            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            image_writer.SetFileName(os.path.join(series_dir_images, f'{i:03d}.dcm'))
            image_writer.Execute(image_slice)

        os.makedirs(study_dir_labels, exist_ok=True)
        subject['label'].save(series_file_labels + '.nii.gz')

    return {'images': image_dir, 'labels': label_dir}