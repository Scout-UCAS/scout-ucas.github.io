"""Suffix-blind Gaussian actor and a separate conditional answer decoder.

One frozen Qwen backbone carries independently trainable policy/reader LoRA
weights. The reference route snapshots the warm actor. Only Z crosses from a
policy call to a reader call. All caches are local to an individual call.
"""
from __future__ import annotations

import copy
import math
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from .settings import ModelSettings


class ByteTokenizer:
    """The offline model uses bytes plus PAD, BOS and EOS."""
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    vocab_size = 259

    def encode(self, text, add_special_tokens=False):
        return [x + 3 for x in text.encode("utf-8")]

    def decode(self, ids, **kwargs):
        if any(i < 3 for i in ids):
            return "\ufffd"
        return bytes(i-3 for i in ids).decode("utf-8", errors="replace")


class LowRank(nn.Module):
    def __init__(self, n_in, n_out, rank):
        super().__init__()
        self.a = nn.Linear(n_in, rank, bias=False)
        self.b = nn.Linear(rank, n_out, bias=False)
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.b(self.a(x.float())).to(x.dtype)


class RoutedLinear(nn.Module):
    def __init__(self, original, rank):
        super().__init__()
        self.original = original
        self.adapters = nn.ModuleDict({name: LowRank(original.in_features, original.out_features, rank)
                                      for name in ("policy", "reader", "reference")})
        self.adapters["reference"].requires_grad_(False)
        self.route = "policy"

    def forward(self, x):
        return self.original(x) + self.adapters[self.route](x)


class CTModel(nn.Module):
    def __init__(self, cfg: ModelSettings):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "tiny":
            self.tokenizer = ByteTokenizer()
            architecture = Qwen2Config(
                vocab_size=self.tokenizer.vocab_size, hidden_size=cfg.hidden_size,
                intermediate_size=cfg.hidden_size*3, num_hidden_layers=cfg.layers,
                num_attention_heads=cfg.heads, num_key_value_heads=cfg.heads,
                max_position_embeddings=4096, attention_dropout=0.0,
                bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False)
            self.lm = Qwen2ForCausalLM(architecture).to(getattr(torch, cfg.dtype))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, revision=cfg.revision)
            self.lm = AutoModelForCausalLM.from_pretrained(
                cfg.backbone, revision=cfg.revision, torch_dtype=getattr(torch, cfg.dtype))
        generation_eos = self.lm.generation_config.eos_token_id
        if isinstance(generation_eos, int):
            generation_eos = [generation_eos]
        self._answer_eos = self.tokenizer.eos_token_id
        if self._answer_eos is None:
            if not generation_eos:
                raise ValueError("The model needs an EOS token to supervise and terminate answers")
            self._answer_eos = generation_eos[0]
        self.stop_tokens = frozenset([self.eos, *(generation_eos or [])])
        self.register_buffer("_stop_token_ids", torch.tensor(sorted(self.stop_tokens)), persistent=False)
        self.lm.requires_grad_(False)
        self.resolved_revision = getattr(self.lm.config, "_commit_hash", None)
        hidden = self.lm.config.hidden_size
        # Plain Python list keeps a single state_dict path for each wrapped layer.
        self.routed = []
        for block in self.lm.model.layers:
            for name in ("q_proj", "v_proj"):
                layer = RoutedLinear(getattr(block.self_attn, name), cfg.adapter_rank)
                setattr(block.self_attn, name, layer)
                self.routed.append(layer)
        self.policy_head = nn.Linear(hidden, 2*cfg.action_dim)
        self.ref_head = copy.deepcopy(self.policy_head).requires_grad_(False)
        self.mapper = nn.Linear(cfg.action_dim, hidden)
        nn.init.normal_(self.mapper.weight, std=0.04 / math.sqrt(cfg.action_dim))
        nn.init.zeros_(self.mapper.bias)
        self.static_head = nn.Linear(hidden, cfg.latent_steps*cfg.action_dim)
        if cfg.backbone == "tiny":
            self.read_head = copy.deepcopy(self.lm.lm_head).float().requires_grad_(True)
        else:
            self.read_head = LowRank(hidden, self.lm.config.vocab_size, cfg.adapter_rank)
        self.work = Counter()
        self.peak_cache_bytes = 0
        self.eval()  # dropout would break the likelihood/replay and reference checks

    @property
    def device(self):
        return self.policy_head.weight.device

    @property
    def eos(self):
        return self._answer_eos

    def ends_answer(self, ids):
        if len(self.stop_tokens) == 1:
            return ids.eq(self.eos)
        return torch.isin(ids, self._stop_token_ids)

    def tokens(self, text, bos=False):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        start = self.tokenizer.bos_token_id
        return ([start] if bos and start is not None else []) + ids

    def embed(self, ids):
        return self.lm.get_input_embeddings()(torch.as_tensor(ids, dtype=torch.long, device=self.device))

    def pack(self, sequences):
        lengths = torch.tensor([seq.shape[0] for seq in sequences], device=self.device)
        hidden = pad_sequence([seq.flip(0) for seq in sequences], batch_first=True).flip(1)
        offsets = hidden.shape[1] - lengths[:, None]
        mask = torch.arange(hidden.shape[1], device=self.device)[None, :] >= offsets
        return hidden, mask.long()

    def prefix_inputs(self, prefixes):
        return self.pack([self.embed(self.tokens(p + "\n", bos=True)) for p in prefixes])

    def _hidden(self, x, mask, route, cache=None, use_cache=False, all_layers=False):
        if mask.shape[1] > self.lm.config.max_position_embeddings:
            raise ValueError(f"Context of {mask.shape[1]} exceeds model limit")
        for layer in self.routed:
            layer.route = route
        pos = (mask.cumsum(-1)-1).clamp_min(0)[:, -x.shape[1]:]
        result = self.lm.model(inputs_embeds=x, attention_mask=mask, position_ids=pos,
                               use_cache=use_cache, past_key_values=cache, return_dict=True,
                               output_hidden_states=all_layers)
        if use_cache:
            cache_bytes = sum(t.numel() * t.element_size()
                              for layer in result.past_key_values.layers
                              for t in (layer.keys, layer.values) if t is not None)
            self.peak_cache_bytes = max(self.peak_cache_bytes, cache_bytes)
        self.work[f"{route}_forwards"] += 1
        self.work[f"{route}_padded_tokens"] += x.shape[0]*x.shape[1]
        states = result.hidden_states[1:] if all_layers else result.last_hidden_state
        return states, result.past_key_values

    def stats(self, h, route):
        head = self.ref_head if route == "reference" else self.policy_head
        mean, raw_scale = head(h.float()).chunk(2, dim=-1)
        # A bounded, strictly positive standard deviation keeps joint densities finite.
        return mean, 0.05 + 1.45*raw_scale.sigmoid()

    def sample(self, prefixes, *, method="ct-rl", mean_only=False):
        """Returns U [B,T,r] and Z [B,T,d]. No suffix argument exists here."""
        x, mask = self.prefix_inputs(prefixes)
        dtype = x.dtype
        count = self.cfg.latent_steps
        if method == "placeholder-sft":
            actions = x.new_zeros((len(prefixes), count, self.cfg.action_dim))
            return actions, x.new_zeros((len(prefixes), count, x.shape[-1]))
        if method == "static-compress-sft":
            h, _ = self._hidden(x, mask, "policy")
            actions = self.static_head(h[:, -1].float()).reshape(len(prefixes), count, -1)
            return actions, self.mapper(actions).to(dtype)
        actions, embeddings = [], []
        cache = None
        cache_ok = not torch.is_grad_enabled()
        for step in range(count):
            h, cache = self._hidden(x, mask, "policy", cache=cache, use_cache=cache_ok)
            mu, sigma = self.stats(h[:, -1], "policy")
            u = mu if mean_only else Normal(mu, sigma).rsample()
            z = self.mapper(u).to(dtype)
            actions.append(u)
            embeddings.append(z)
            if step + 1 < count:
                x = z[:, None] if cache_ok else torch.cat((x, z[:, None]), 1)
                mask = F.pad(mask, (0, 1), value=1)
        return torch.stack(actions, 1), torch.stack(embeddings, 1)

    def replay(self, prefixes, actions, route="policy"):
        """Teacher-force fixed sampled actions under a chosen parameter route."""
        x, mask = self.prefix_inputs(prefixes)
        first = x.shape[1]-1
        u = actions.detach()
        history = self.mapper(u[:, :-1]).to(x.dtype)
        h, _ = self._hidden(torch.cat((x, history), 1),
                            F.pad(mask, (0, history.shape[1]), value=1), route)
        mu, sigma = self.stats(h[:, first:], route)
        normal = Normal(mu, sigma)
        return normal.log_prob(u).sum(-1), normal

    @torch.no_grad()
    def snapshot_reference(self):
        self.ref_head.load_state_dict(self.policy_head.state_dict())
        for layer in self.routed:
            layer.adapters["reference"].load_state_dict(layer.adapters["policy"].state_dict())

    def parameters_for(self, stage):
        if stage == "policy":
            heads = [self.policy_head]
        elif stage == "reader":
            heads = [self.read_head]
        else:
            raise ValueError(stage)
        modules = heads + [layer.adapters[stage] for layer in self.routed]
        return [parameter for module in modules for parameter in module.parameters()]

    def native_text_prefix(self, prefix):
        prompt = self.embed(self.tokens("Answer only.\n", bos=True))
        return torch.cat((prompt, self.embed(self.tokens("\n"+prefix+"\n"))))

    def native_text_tail(self, suffix, cue):
        return self.embed(self.tokens(suffix+"\n"+cue+"\n="))

    def reader_inputs(self, z, suffixes):
        if suffixes and isinstance(suffixes[0], tuple) and len(suffixes[0]) == 3:
            return self.pack([torch.cat((self.native_text_prefix(p),self.native_text_tail(s,cue)))
                              for p,s,cue in suffixes])
        if len(suffixes) != len(z):
            raise ValueError("A suffix is required for each latent state")
        prompt = self.embed(self.tokens("Answer only.\n", bos=True))
        return self.pack([torch.cat((prompt, latent, self.embed(self.tokens("\n" + suffix + "\n="))), 0)
                          for latent, suffix in zip(z, suffixes)])

    def logits(self, hidden):
        self.work["logit_positions"] += hidden.numel() // hidden.shape[-1]
        if self.cfg.backbone == "tiny":
            return self.read_head(hidden.float())
        return self.lm.lm_head(hidden).float() + self.read_head(hidden).float()

    def answer_loss(self, z, suffixes, answers):
        """Sum gold token NLL over each answer (including EOS), average by answer."""
        x, mask = self.reader_inputs(z, suffixes)
        first = x.shape[1] - 1
        targets = [self.tokens(a) + [self.eos] for a in answers]
        if len(targets) != len(z):
            raise ValueError("A gold answer is required for every reader prompt")
        labels = pad_sequence([torch.tensor(t, device=self.device) for t in targets],
                              batch_first=True, padding_value=-100)
        previous = labels[:, :-1]
        x = torch.cat((x, self.embed(previous.clamp_min(0))), 1)
        mask = torch.cat((mask, (previous != -100).long()), 1)
        h, _ = self._hidden(x, mask, "reader")
        scores = self.logits(h[:, first:]).reshape(-1, self.lm.config.vocab_size)
        return F.cross_entropy(scores.float(), labels.reshape(-1), ignore_index=-100,
                               reduction="sum") / len(answers)

    @torch.no_grad()
    def decode(self, z, suffixes, max_tokens=None):
        """Single greedy completion per row. None denotes exhaustion before EOS."""
        x, mask = self.reader_inputs(z, suffixes)
        return self._greedy(x, mask, None, max_tokens)

    @torch.no_grad()
    def diagnostic_with_prefix_cache(self, prefixes, z, suffixes):
        """Deliberate cache-access intervention, excluded from the strict interface."""
        raw, raw_mask = self.prefix_inputs(prefixes)
        _, cache = self._hidden(raw, raw_mask, "policy", use_cache=True)
        x, mask = self.reader_inputs(z, suffixes)
        return self._greedy(x, torch.cat((raw_mask, mask), 1), cache, None)

    def _greedy(self, x, mask, cache, max_tokens):
        generated = [[] for _ in range(x.shape[0])]
        done = [False] * x.shape[0]
        for _ in range(max_tokens or self.cfg.max_answer_tokens):
            h, cache = self._hidden(x, mask, "reader", cache, use_cache=True)
            ids = self.logits(h[:, -1]).argmax(-1)
            for i, token in enumerate(ids.tolist()):
                if not done[i]:
                    if token in self.stop_tokens:
                        done[i] = True
                    else:
                        generated[i].append(token)
                        self.work["generated_answer_tokens"] += 1
            if all(done):
                break
            x = self.embed(ids)[:, None]
            mask = F.pad(mask, (0, 1), value=1)
        return [self.tokenizer.decode(tokens, skip_special_tokens=False).strip() if finished else None
                for tokens, finished in zip(generated, done)]

    def portable_state(self):
        state = self.state_dict()
        if self.cfg.backbone == "tiny":
            return state
        return {key: value for key, value in state.items()
                if not key.startswith("lm.") or ".adapters." in key}

    def restore(self, state):
        expected = set(self.portable_state())
        if set(state) != expected:
            raise ValueError(f"Checkpoint architecture mismatch ({len(expected ^ set(state))} keys)")
        self.load_state_dict(state, strict=self.cfg.backbone == "tiny")
