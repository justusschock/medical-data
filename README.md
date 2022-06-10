# Medical Deep Learning Utilities

[![UnitTest](https://github.com/justusschock/medical-dl-utils/actions/workflows/unittests.yaml/badge.svg)](https://github.com/justusschock/medical-dl-utils/actions/workflows/unittests.yaml) [![Docker](https://github.com/justusschock/medical-dl-utils/actions/workflows/docker_build.yaml/badge.svg)](https://github.com/justusschock/medical-shape/actions/workflows/docker_build.yaml) [![Docker Stable](https://github.com/justusschock/medical-dl-utils/actions/workflows/docker_stable.yaml/badge.svg)](https://github.com/justusschock/medical-dl-utils/actions/workflows/docker_stable.yaml) [![Build Package](https://github.com/justusschock/medical-dl-utils/actions/workflows/package_build.yaml/badge.svg)](https://github.com/justusschock/medical-shape/actions/workflows/package_build.yaml) ![PyPI](https://img.shields.io/pypi/v/medical-dl-utils?color=grene) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/justusschock/medical-dl-utils/main.svg)](https://results.pre-commit.ci/latest/github/justusschock/medical-dl-utils/main)

This repository contains utilities for training and evaluating deep learning models in medical image analysis that are not specific to certain tasks.

So far it consists of 2 major parts:
An abstract Dataset API built on top of `torch.utils.data.Dataset` and `torchio.data.Subject` as well as general transforms for this kind of data.

To use the dataset classes, you basically only need to implement the `parse_subjects` method to return a list of samples and everything else will work automatically.
You will automatically get image statistics such as median spacing or median shape. For label statistics, you either need to subclass the `AbstractDiscreteLabelDataset` or implement the `get_single_label_stats` and `aggregate_label_stats` methods.

All transforms work on `torchio.data.Subjects` and can be passed to the datasets as optional parameters. You can also pass `"default"` as a parameter to use the default transforms.

Pull requests for other common utilities are highly welcomed.

## Installation

This project can be installed either from PyPI or by cloning the repository from GitHub.

For an install of published packages, use the command

```bash

    pip install medical-dl-utils

```

To install from the (cloned) repository, use the command

```bash

    pip install PATH/TO/medical-dl-utils

```

You can also add `-e` to the command to make an editable install in case you want to modify the code.

You can also install the package directly from GitHub by running

```bash

    pip install git+https://github.com/justusschock/medical-dl-utils.git

```

## Docker Images

We provide a docker image for easy usage of the package and as a base image for other projects.

The file for this image can be found at `dockers/Dockerfile`. We provide both, a CPU-only and a CUDA-enabled image based on the NVIDIA NGC PyTorch image.
These images can be found on [DockerHub](https://hub.docker.com/repository/docker/justusschock/medical-dl-utils).
