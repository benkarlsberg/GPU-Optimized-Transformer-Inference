from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SyntheticPrompt:
    batch_size: int
    prompt_length: int
    vocab_hint: int = 1000


def make_input_ids(tokenizer, prompt: SyntheticPrompt, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    vocab = min(getattr(tokenizer, "vocab_size", prompt.vocab_hint) - 1, 10000)
    ids = torch.randint(1, max(2, vocab), (prompt.batch_size, prompt.prompt_length), generator=g)
    return ids.to(device)
