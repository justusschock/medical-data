import json
from typing import Any, Mapping

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
