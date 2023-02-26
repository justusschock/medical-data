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

from copy import deepcopy
from typing import Any

import torch
from torchmetrics import Metric

__all__ = ["SingleClassMetric"]


class SingleClassMetric(Metric):
    def __init__(self, class_idx: int, metric_instance: Metric, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = deepcopy(metric_instance)
        self.class_idx = class_idx

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.argmax(1)
        preds = (preds == self.class_idx).long()
        target = (target == self.class_idx).long()

        return self.metric.update(preds, target)

    def compute(self) -> torch.Tensor | Any:
        return self.metric.compute()
