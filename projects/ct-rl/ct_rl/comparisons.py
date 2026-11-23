"""Local comparison architectures. Strict readers get (Z, suffix), native readers also get text."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from .model import CTModel, LowRank
from .settings import LATENT_RL, LOOP, PROCESS
from .tasks import prefix_state


def prefix_trace(pair, side):
    """Execution of the prefix only. No continuation or future answer enters this function."""
    if pair.family == "rules":
        facts = set(pair.starts[side])
        steps = [",".join(map(str, sorted(facts)))]
        while True:
            extra = {b for a, b in pair.world["edges"] if a in facts} - facts
            if not extra:
                return steps
            facts.update(extra)
            steps.append(",".join(map(str, sorted(facts))))
    result = []
    for end in range(1, len(pair.operations) + 1):
        state = prefix_state(replace(pair, operations=pair.operations[:end]), side)
        result.append(",".join(map(str, state)) if isinstance(state, tuple) else str(state))
    return result


def texts_for(pairs):
    return [prefix_trace(pair, side) for pair in pairs for side in (0, 1)]


def normalized(x, axis=-1):
    return (x-x.mean(axis, keepdim=True))/(x.std(axis, unbiased=False, keepdim=True)+1e-6)


def stop_distribution(logits, minimum):
    """First-stop mass, with forced survival before Tmin and forced stop at Tmax."""
    prob = logits.sigmoid()
    t = torch.arange(prob.shape[-1], device=prob.device)
    prob = torch.where(t < minimum-1, 0., prob)
    prob = torch.where(t == prob.shape[-1]-1, 1., prob)
    survival = F.pad((1-prob[..., :-1]).cumprod(-1), (1, 0), value=1.)
    return prob * survival


@dataclass
class Thoughts:
    raw: torch.Tensor
    z: torch.Tensor
    ids: torch.Tensor | None = None
    perturbed: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    gates: torch.Tensor | None = None
    length: torch.Tensor | None = None

    def interface(self, native=False):
        state = self.raw if native else self.z
        if self.length is None:
            return state
        t = torch.arange(state.shape[1], device=state.device)
        return state * (t[None, :] < self.length[:, None]).unsqueeze(-1)


class LoopModel(CTModel):
    """Depth recurrence over the same causal stack, including while decoding each token."""
    def __init__(self, cfg):
        super().__init__(cfg.model)
        self.depth = cfg.loop_depth
        self.ref_read = copy.deepcopy(self.read_head).requires_grad_(False)
        for layer in self.routed:
            layer.adapters["loop_reference_reader"] = copy.deepcopy(layer.adapters["reader"]).requires_grad_(False)

    @torch.no_grad()
    def snapshot_reference(self):
        super().snapshot_reference()
        self.ref_read.load_state_dict(self.read_head.state_dict())
        for layer in self.routed:
            layer.adapters["loop_reference_reader"].load_state_dict(layer.adapters["reader"].state_dict())

    def ref_logits(self, h):
        return (self.ref_read(h.float()) if self.cfg.backbone == "tiny"
                else self.lm.lm_head(h).float()+self.ref_read(h).float())

    def loop_states(self, x, mask, route, cache=None, use_cache=False):
        outputs, caches = [], []
        for depth in range(self.depth):
            prior = cache[depth] if cache is not None else None
            x, next_cache = super()._hidden(x, mask, route, prior, use_cache)
            outputs.append(x)
            caches.append(next_cache)
        if use_cache:
            size = sum(t.numel()*t.element_size() for c in caches for layer in c.layers
                       for t in (layer.keys, layer.values) if t is not None)
            self.peak_cache_bytes = max(self.peak_cache_bytes, size)
        return outputs, tuple(caches) if use_cache else None

    def _hidden(self, x, mask, route, cache=None, use_cache=False):
        outputs, caches = self.loop_states(x, mask, route, cache, use_cache)
        return outputs[-1], caches


class ThoughtModel(CTModel):
    def __init__(self, cfg):
        super().__init__(cfg.model)
        self.recipe = cfg
        d = self.lm.config.hidden_size
        self.down = nn.Linear(d, cfg.model.action_dim)
        if cfg.method == "codi":
            self.project = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d), nn.LayerNorm(d))
        if cfg.method == "slpo":
            self.gate = nn.Linear(d, 1)
        if cfg.method in ("latent-grpo", "ltc"):
            self.word_adapter = LowRank(d, self.lm.config.vocab_size, cfg.model.adapter_rank)

    def parameters_for(self, stage):
        params = super().parameters_for(stage)
        if stage == "policy":
            for name in ("down", "project", "gate", "word_adapter"):
                if hasattr(self, name):
                    params.extend(getattr(self, name).parameters())
        return params

    @contextmanager
    def mc_dropout(self, enabled):
        handles = []
        if enabled:
            def perturb(module, args, out):
                return F.dropout(out, self.recipe.dropout_prob, training=True)
            handles = [layer.register_forward_hook(perturb) for layer in self.routed]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def policy_words(self, h):
        return self.lm.lm_head(h).float() + self.word_adapter(h).float()

    def thoughts(self, prefixes, *, stochastic, steps=None, full_gate=False):
        cfg = self.recipe
        x, mask = self.prefix_inputs(prefixes)
        total = self.cfg.latent_steps if steps is None else steps
        raw, indices, perturbed, weights, gates = [], [], [], [], []
        if total == 0:
            empty = x[:, :0]
            return Thoughts(empty, empty)
        length = torch.full((len(prefixes),), total, device=self.device)
        active = torch.ones_like(length, dtype=torch.bool)
        use_cache = not stochastic and not torch.is_grad_enabled() and not self.training
        cache = None
        for t in range(total):
            with self.mc_dropout(cfg.method == "slpo" and stochastic):
                hs, cache = self._hidden(x, mask, "policy", cache=cache, use_cache=use_cache)
            h = hs[:, -1]
            if cfg.method in ("ltc", "latent-grpo"):
                values, ids = self.policy_words(h).log_softmax(-1).topk(
                    min(cfg.top_k, self.lm.config.vocab_size), dim=-1)
                score = values
                if stochastic:
                    noise = -torch.empty_like(values).exponential_().log()
                    if cfg.method == "latent-grpo":
                        noise = noise.clamp(-1.5, 3.) + 1.5 + cfg.gumbel_delta
                    score = values + noise
                w = (score/cfg.thought_temperature).softmax(-1)
                coeff = F.normalize(w, p=2, dim=-1) if cfg.method == "ltc" else w
                token = (coeff.unsqueeze(-1)*self.embed(ids)).sum(1).to(x.dtype)
                indices.append(ids)
                perturbed.append(score)
                weights.append(w)
            elif cfg.method == "slpo":
                token = h
                gate = self.gate(token.float()).squeeze(-1)
                gates.append(gate)
                eligible = t+1 >= min(total, cfg.min_thought_steps)
                stop = (torch.rand_like(gate) < gate.sigmoid()) if stochastic else (gate >= 0)
                ended = active & stop & eligible
                length = torch.where(ended, t+1, length)
                active = active & ~ended
            elif cfg.method == "codi":
                token = self.project(h.float()).to(x.dtype)
            else:
                token = h
            raw.append(token)
            if cfg.method == "slpo" and not full_gate and not active.any():
                raw.extend([torch.zeros_like(token)] * (total-t-1))
                break
            if t + 1 < total:
                x = token[:, None] if use_cache else torch.cat((x, token[:, None]), 1)
                mask = F.pad(mask, (0, 1), value=1)
        raw = torch.stack(raw, 1)
        z = self.mapper(self.down(raw.float())).to(x.dtype)
        stack = lambda seq: torch.stack(seq, 1) if seq else None
        return Thoughts(raw, z, stack(indices), stack(perturbed), stack(weights), stack(gates),
                        length if cfg.method == "slpo" else None)

    def sample(self, prefixes, *, method=None, mean_only=False):
        # Evaluation of literature baselines uses their deterministic latent readout.
        state = self.thoughts(prefixes, stochastic=False)
        return self.down(state.raw.float()), state.interface()

    def history_states(self, prefixes, thoughts):
        x, mask = self.prefix_inputs(prefixes)
        first = x.shape[1]-1
        prev = thoughts.raw[:, :-1].detach()
        h, _ = self._hidden(torch.cat((x, prev), 1), F.pad(mask, (0, prev.shape[1]), value=1), "policy")
        return h[:, first:]

    def reader_inputs(self, z, suffixes):
        if suffixes and isinstance(suffixes[0], tuple) and len(suffixes[0]) == 2:
            if self.recipe.method == "slpo":
                z = [row[row.detach().abs().sum(-1) > 0] for row in z]
            return self.pack([torch.cat((self.embed(self.tokens(p+"\n", bos=True)), state,
                                         self.embed(self.tokens("\n"+s+"\n="))), 0)
                              for state, (p, s) in zip(z, suffixes)])
        return super().reader_inputs(z, suffixes)


def build_model(cfg):
    if cfg.method in LOOP:
        return LoopModel(cfg)
    if cfg.method in PROCESS + LATENT_RL:
        return ThoughtModel(cfg)
    return CTModel(cfg.model)


def text_prompt(prefix, suffix, method):
    cue = "Show work. End with Final answer: " if method == "cot" else "Return the final answer."
    return prefix, suffix, cue


def empty_state(model, count):
    w = model.lm.get_input_embeddings().weight
    return w.new_empty((count, 0, w.shape[-1]))
