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
import os
import sys
from contextlib import contextmanager
import json
from typing import Any, Mapping, Generator

import torch
from pytorch_lightning.utilities.apply_func import apply_to_collection

__all__ = [
    "PyTorchJsonEncoder",
    "PyTorchJsonDecoder",
]


class PyTorchJsonEncoder(json.JSONEncoder):
    """An encoder for PyTorch Tensors."""

    def default(self, o: Any) -> Any:
        """Encode a PyTorch Tensor."""
        if isinstance(o, torch.Tensor):
            o = {"type": "torch.Tensor", "content": o.tolist()}
        return o


class PyTorchJsonDecoder(json.JSONDecoder):
    """A decoder for serialized PyTorch Tensors."""

    def to_tensor(self, d: Mapping[str, Any]) -> torch.Tensor:
        """convert mappings to tensors if applicable."""
        if isinstance(d, Mapping) and "type" in d and d["type"] == "torch.Tensor" and "content" in d:
            return torch.tensor(d["content"])
        return {k: apply_to_collection(v, Mapping, self.to_tensor) for k, v in d.items()}

    def decode(self, s: str, *args, **kwargs) -> Any:
        decoded = super().decode(s, *args, **kwargs)
        decoded = apply_to_collection(decoded, Mapping, self.to_tensor)
        return decoded


# TODO: Find why this is not suppressing all PL Trainer output
@contextmanager
def suppress_stdout() -> Generator[None, None, None]:
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        old_std_err = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_std_err
