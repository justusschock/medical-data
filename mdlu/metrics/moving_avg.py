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

from operator import xor

import torch
from torchmetrics import Metric

__all__ = ["MovingAverage"]


class MovingAverage(Metric):
    sliding_window: list[torch.Tensor]
    current_average: torch.Tensor

    def __init__(
        self,
        momentum: float | None = 0.93,
        sliding_window_size: int | None = None,
    ) -> None:
        super().__init__()

        # need to add states here globally independent of arguments for momentum and sliding_window_size to satisfy mypy
        self.add_state("sliding_window", [], persistent=True)
        self.add_state("current_average", torch.tensor(float("nan")), persistent=True)

        assert xor(
            bool(sliding_window_size is None), bool(momentum is None)
        ), "Either momentum or sliding_window_size must be specified"
        if momentum is not None:
            if not 0.0 < momentum < 1.0:
                raise ValueError("momentum must be between 0.0 and 1.0")

        else:
            if sliding_window_size is None or not sliding_window_size > 0:
                raise ValueError("sliding_window_size must be greater than 0")

        self.momentum = momentum
        self.sliding_window_size = sliding_window_size

    def update(self, value: torch.Tensor) -> None:
        if self.momentum is not None:
            if torch.isnan(self.current_average):
                self.current_average = value.detach()

            else:
                self.current_average = (
                    self.momentum * self.current_average
                    + (1 - self.momentum) * value.detach()
                )
        else:
            self.sliding_window.append(value.detach())
            if self.sliding_window_size is None:
                raise ValueError("sliding_window_size must be specified")

            if len(self.sliding_window) > self.sliding_window_size:
                self.sliding_window.pop(0)

    def compute(self) -> torch.Tensor:
        if self.momentum is not None:
            return self.current_average
        else:
            result = sum(self.sliding_window) / len(self.sliding_window)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, device=self.device, dtype=torch.float)
            return result
