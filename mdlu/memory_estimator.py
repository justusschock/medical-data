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
import math
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch
import torchio as tio
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from mdlu.utils import suppress_stdout


def b2mb(x: float) -> float:
    return x / (2**20)


def mb2b(x: float) -> float:
    return x * (2**20)


class MemoryEstimator:
    def __init__(
        self, device_id: int = 0, mixed_precision: bool = False, offset: int = 1  # 1mb
    ) -> None:

        self.device_id = device_id
        self.device_props = torch.cuda.get_device_properties(device_id)
        self.offset = offset
        self.mixed_precision = mixed_precision

    @property
    def device_name(self) -> str:
        return self.device_props.name

    @property
    def total_memory(self) -> float:
        return b2mb(self.device_props.total_memory)

    def create_offset_tensor_on_gpu(self) -> torch.Tensor:
        device = torch.device(f"cuda:{self.device_id}")
        tensor_mem = torch.rand(
            1, dtype=float, requires_grad=False, device=device
        ).element_size()
        return torch.rand(
            math.ceil(self.offset / tensor_mem),
            dtype=float,
            requires_grad=False,
            device=device,
        )

    def __call__(
        self,
        network: LightningModule,
        batch_size: int,
        image_patch_shape: np.ndarray,
        output_shape: np.ndarray,
        whole_image_shape: np.ndarray,
        in_channels: int,
    ) -> tuple[int, bool]:

        try:
            device = torch.device(f"cuda:{self.device_id}")

            with cudnn_deterministic():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                network = network.to(device)

                block_tensor = self.create_offset_tensor_on_gpu().to(device=device)
                import time

                time.sleep(1)
                dset_train = [
                    {
                        "data": {
                            tio.constants.DATA: torch.rand(
                                (in_channels, *image_patch_shape),
                                dtype=torch.float,
                            )
                        },
                        "label": {
                            tio.constants.DATA: torch.zeros(
                                tuple(output_shape.tolist()),
                                dtype=torch.long,
                            ),
                        },
                    }
                    for i in range(batch_size * 10)
                ]

                dset_val = [
                    {
                        "data": {
                            tio.constants.DATA: torch.rand(
                                (in_channels, *whole_image_shape),
                                dtype=torch.float,
                            )
                        },
                        "label": {
                            tio.constants.DATA: torch.zeros(
                                tuple(output_shape.tolist()),
                                dtype=torch.long,
                            ),
                        },
                    }
                    for i in range(10)
                ]
                loader_train = DataLoader(
                    dset_train, batch_size=batch_size, shuffle=False
                )
                loader_val = DataLoader(dset_val, batch_size=1, shuffle=False)

                with suppress_stdout():
                    trainer = Trainer(
                        devices=[self.device_id],
                        accelerator="gpu",
                        precision=16 + int(not self.mixed_precision) * 16,
                        enable_model_summary=False,
                        enable_progress_bar=False,
                        max_epochs=1,
                        limit_train_batches=len(loader_train),
                        limit_val_batches=len(loader_val),
                        enable_checkpointing=False,
                        logger=None,
                    )
                    trainer.fit(network, loader_train, loader_val)

                torch.cuda.empty_cache()
                dyn_mem = torch.cuda.max_memory_reserved()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                dyn_mem = float("inf")
            else:
                raise e

        finally:
            for key in ("block_tensor", "trainer", "loader", "dset"):
                if key in locals():
                    del locals()[key]

            network.cpu()
            torch.cuda.empty_cache()
            gc.collect()

        return dyn_mem, torch.isfinite(torch.tensor(dyn_mem))


@contextmanager
def cudnn_deterministic() -> Generator[None, None, None]:
    old_value = torch.backends.cudnn.deterministic
    old_value_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = old_value
        torch.backends.cudnn.benchmark = old_value_benchmark
