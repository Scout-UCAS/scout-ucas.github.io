"""Checkpoint inference protocols shared by the CLI, controlled tests, and public tasks."""
from __future__ import annotations

import copy
import re

import torch

from .comparisons import empty_state, text_prompt
from .curriculum import batched_decode
from .settings import LATENT_RL, PROCESS, TEXT


def completion(answer, mode, steps=""):
    if mode == "cot":
        return (steps + "\nFinal answer: " + answer).strip()
    return answer


def extract(pred, mode):
    if mode != "cot" or pred is None:
        return pred
    parts = re.split(r"Final answer:\s*", pred, flags=re.I)
    return (parts[-1].strip() or None) if len(parts) > 1 else None


class Readout:
    """Encode each prefix once and answer multiple suffixes in their supplied order."""

    def __init__(self, model, cfg, prefixes, mode, *, serial=False):
        self.model, self.prefixes, self.mode = model, prefixes, mode
        self.batch_size = 1 if serial else cfg.forward_batch
        self.text_caches = []
        if mode in TEXT:
            self.state = empty_state(model, len(prefixes))
            if serial:
                for prefix in prefixes:
                    x, mask = model.pack([model.native_text_prefix(prefix)])
                    _, cache = model._hidden(x, mask, "reader", use_cache=True)
                    self.text_caches.append((cache, mask))
            return
        if mode == "native" and cfg.method not in PROCESS + LATENT_RL:
            raise ValueError("Native latent mode requires a literature model")
        if mode not in ("native", "latent"):
            raise ValueError(f"Unknown readout mode: {mode}")

        def encode(batch):
            if mode == "native":
                return model.thoughts(batch, stochastic=False).interface(native=True)
            return model.sample(batch, method=cfg.method)[1]

        self.state = torch.cat([encode([p]) for p in prefixes]) if serial else encode(prefixes)

    def raw(self, owners, suffixes, tokens=None):
        prompts = suffixes
        if self.mode in TEXT:
            prompts = [text_prompt(self.prefixes[i], s, self.mode) for i, s in zip(owners, suffixes)]
            if self.text_caches:
                predictions = []
                for i, (_, suffix, cue) in zip(owners, prompts):
                    cache, prefix_mask = self.text_caches[i]
                    x, tail_mask = self.model.pack([self.model.native_text_tail(suffix, cue)])
                    mask = torch.cat((prefix_mask, tail_mask), 1)
                    predictions.extend(self.model._greedy(x, mask, copy.deepcopy(cache), tokens))
                return predictions
        elif self.mode == "native":
            prompts = [(self.prefixes[i], s) for i, s in zip(owners, suffixes)]
        return batched_decode(self.model, self.state, owners, prompts, self.batch_size, tokens)

    def answers(self, owners, suffixes, tokens=None):
        return [extract(raw, self.mode) for raw in self.raw(owners, suffixes, tokens)]
