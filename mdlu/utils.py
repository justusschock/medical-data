import json
from typing import Any, Mapping
import torch
from pytorch_lightning.utilities.apply_func import apply_to_collection


class PyTorchJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, torch.Tensor):
            o = {"type": "torch.Tensor", "content": o.tolist()}
        return o


class PyTorchJsonDecoder(json.JSONDecoder):
    def to_tensor(self, d: Mapping[str, Any]) -> torch.Tensor:
        if (
            isinstance(d, Mapping)
            and "type" in d
            and d["type"] == "torch.Tensor"
            and "content" in d
        ):
            return torch.tensor(d["content"])
        return {k: apply_to_collection(v, Mapping, self.to_tensor) for k, v in d.items()}

    def decode(self, s: str, *args, **kwargs) -> Any:
        decoded = super().decode(s, *args, **kwargs)
        decoded = apply_to_collection(decoded, Mapping, self.to_tensor)
        return decoded
