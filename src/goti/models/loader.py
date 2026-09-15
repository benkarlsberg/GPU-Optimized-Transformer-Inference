from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    name: str
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype


_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in _DTYPES:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[key]


def load_causal_lm(
    model_name: str,
    *,
    dtype: str = "fp16",
    device: str | None = None,
    trust_remote_code: bool = False,
) -> LoadedModel:
    torch_dtype = resolve_dtype(dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype if dev.type == "cuda" else torch.float32,
        trust_remote_code=trust_remote_code,
    )
    model.to(dev)
    model.eval()
    return LoadedModel(name=model_name, model=model, tokenizer=tok, device=dev, dtype=torch_dtype)
