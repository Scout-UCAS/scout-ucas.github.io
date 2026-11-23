"""Configuration, data manifests, and reproducible run state."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

from .tasks import SPLITS, Pair, generate_split

METHODS = ("ct-rl", "random-suffix-rl", "single-readout-rl", "hardness-matched-rl", "random-separating-rl",
           "latent-sft", "multi-query-sft", "static-compress-sft", "placeholder-sft")
PROCESS = ("coconut", "codi")
LATENT_RL = ("latent-grpo", "slpo", "ltc")
LOOP = ("loop-grpo", "rltt", "loop-ct-rl")
TEXT = ("direct", "cot")
METHODS += PROCESS + LATENT_RL + LOOP + TEXT
SFT = frozenset(("latent-sft", "multi-query-sft", "static-compress-sft", "placeholder-sft", *PROCESS, *TEXT))


def controlled_cot_tokens(backbone):
    return 128 if backbone == "tiny" else 256


@dataclass
class ModelSettings:
    backbone: str = "tiny"
    revision: str | None = None
    hidden_size: int = 48
    layers: int = 2
    heads: int = 4
    action_dim: int = 8
    latent_steps: int = 3
    adapter_rank: int = 8
    dtype: str = "float32"
    max_answer_tokens: int = 8


@dataclass
class Config:
    model: ModelSettings = field(default_factory=ModelSettings)
    method: str = "ct-rl"
    algorithm: str = "rloo"
    seed: int = 31
    device: str = "cpu"
    threads: int = 2
    pairs_per_batch: int = 3
    k: int = 3
    m: int = 4
    warm_steps: int = 300
    rounds: int = 3
    steps_per_round: int = 12
    lr_warm: float = 0.002
    lr_policy: float = 0.0003
    lr_reader: float = 0.001
    grad_clip: float = 1.0
    beta: float = 0.01
    rho: float = 0.5
    clip_epsilon: float = 0.2
    grpo_epochs: int = 2
    mine_pairs: int = 12
    search_budget: int = 12
    keep_per_length: int = 2
    mine_samples: int = 2
    forward_batch: int = 16
    eval_seed: int = 1729
    top_k: int = 16
    thought_temperature: float = 1.0
    answer_temperature: float = 1.0
    gumbel_delta: float = 0.01
    ltc_answers: int = 4
    matching_weight: float = 1.0
    distill_weight: float = 1.0
    dropout_prob: float = 0.10
    mc_draws: int = 4
    variance_floor: float = 0.0001
    min_thought_steps: int = 3
    gate_weight: float = 1.0
    loop_depth: int = 3
    native_pretrain: bool = False

    @classmethod
    def load(cls, path: Path, overrides=()):
        raw = json.loads(path.read_text())
        cot_limit_given = raw.get("method") == "cot" and "max_answer_tokens" in raw.get("model", {})
        for item in overrides:
            name, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"Expected key=value, got {item}")
            if name == "model.max_answer_tokens":
                cot_limit_given = True
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            target = raw
            if name.startswith("model."):
                target = raw.setdefault("model", {})
                name = name[6:]
            target[name] = parsed
        if raw.get("method") == "cot" and not cot_limit_given:
            model = raw.setdefault("model", {})
            model["max_answer_tokens"] = max(model.get("max_answer_tokens", 0),
                                             controlled_cot_tokens(model.get("backbone", "tiny")))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        model = dict(raw.pop("model", {}))
        if raw.get("method") == "cot":
            model.setdefault("max_answer_tokens", controlled_cot_tokens(model.get("backbone", "tiny")))
        cfg = cls(model=ModelSettings(**model), **raw)
        cfg.validate()
        return cfg

    def validate(self):
        if self.native_pretrain and self.method not in PROCESS + LATENT_RL:
            raise ValueError("Native latent context is supported by the five literature methods")
        if self.method not in METHODS or self.algorithm not in ("rloo", "grpo"):
            raise ValueError("Unknown method or policy algorithm")
        if self.k < 2 or self.beta < 0 or not 0 <= self.rho < 1 or not 0 < self.clip_epsilon < 1:
            raise ValueError("Require K >= 2, beta >= 0, 0 <= rho < 1 and 0 < epsilon < 1")
        if not 0 < self.dropout_prob < 1 or self.mc_draws < 2 or self.ltc_answers < 2:
            raise ValueError("Invalid dropout or independent sample count")
        if min(self.top_k, self.min_thought_steps, self.loop_depth) < 1 or self.top_k < 2:
            raise ValueError("Invalid thinking parameters")
        if min(self.thought_temperature, self.answer_temperature, self.variance_floor) <= 0:
            raise ValueError("Temperature and variance floor must be positive")
        if min(self.gumbel_delta, self.gate_weight, self.distill_weight, self.matching_weight) < 0:
            raise ValueError("Negative comparison loss weight")
        if self.model.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("Unknown model dtype")
        if min(self.model.heads, self.model.hidden_size) < 1 or self.model.hidden_size % self.model.heads or (self.model.hidden_size // self.model.heads) % 2:
            raise ValueError("The tiny attention head dimension must be even")
        positives = (self.m, self.pairs_per_batch, self.threads, self.steps_per_round,
                     self.search_budget, self.keep_per_length, self.mine_samples, self.forward_batch,
                     self.grpo_epochs, self.grad_clip, self.model.adapter_rank, self.model.action_dim,
                     self.model.latent_steps, self.model.layers, self.model.max_answer_tokens,
                     self.lr_warm, self.lr_reader, self.lr_policy)
        if min(positives) <= 0 or min(self.warm_steps, self.rounds, self.mine_pairs) < 0:
            raise ValueError("Invalid training counts, model dimensions or learning rates")


@contextmanager
def atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + "-")
    os.close(fd)
    temp = Path(name)
    try:
        yield temp
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, obj):
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_data(out: Path, train_pairs: int, eval_pairs: int, seed: int, valid_pairs=None):
    valid_pairs = eval_pairs if valid_pairs is None else valid_pairs
    if min(train_pairs, valid_pairs, eval_pairs) < 1:
        raise ValueError("All pair counts must be positive")
    if out.exists() and any(out.iterdir()):
        manifest_path = out/"manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
            matches = existing.get("version") == 2 and existing.get("seed") == seed
            for split in SPLITS:
                count = {"train":train_pairs, "valid":valid_pairs}.get(split, eval_pairs)
                entry = existing.get("splits", {}).get(split, {})
                file = out/(split+".jsonl")
                matches = (matches and entry.get("pairs") == count*(2 if split=="scale" else 3)
                           and file.is_file() and digest(file) == entry.get("sha256"))
            if matches:
                return existing
        raise ValueError(f"Destination contains different data: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 2, "seed": seed, "splits": {}}
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".ct-data-") as scratch:
        staging = Path(scratch)/"splits"
        staging.mkdir()
        for split in SPLITS:
            count = {"train": train_pairs, "valid": valid_pairs}.get(split, eval_pairs)
            path = staging / (split + ".jsonl")
            with path.open("w") as stream:
                total = 0
                for pair in generate_split(split, count, seed):
                    stream.write(json.dumps(pair.record(), separators=(",", ":")) + "\n")
                    total += 1
            manifest["splits"][split] = {"pairs": total, "sha256": digest(path)}
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, out)
    return manifest


def read_split(folder: Path, name: str) -> list[Pair]:
    if name not in SPLITS:
        raise ValueError(name)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["version"] != 2:
        raise ValueError("Regenerate data with the current instance grouping rules")
    path = folder / (name + ".jsonl")
    expected = manifest["splits"][name]
    if digest(path) != expected["sha256"]:
        raise ValueError(f"Dataset digest differs for {path}")
    with path.open() as stream:
        pairs = [Pair(**json.loads(line)) for line in stream]
    if len(pairs) != expected["pairs"] or not pairs or any(p.split != name for p in pairs):
        raise ValueError(f"Invalid split {name}")
    return pairs


def runtime(cfg: Config, device=None):
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = device or cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device)


def versions():
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": transformers.__version__, "platform": platform.platform(),
            "source_sha256": source.hexdigest()}


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
