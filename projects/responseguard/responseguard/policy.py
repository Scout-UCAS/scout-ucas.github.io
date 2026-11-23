"""Trainable flow policies. Only adapter parameters are optimized."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw
from torch import nn

from .data import COLORS, FACTORS, RELATIONS, Scene, VALUES


@dataclass
class Frame:
    image: Image.Image
    # Simulator truth, available only in the offline integration backend.
    scene: Scene | None = None
    quality: float | None = None


class TinyPolicy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.counts = Counter()
        self.widths = [len(VALUES[k]) for k in FACTORS]
        self.dim = sum(self.widths) + 4
        self.latent_shape = (self.dim,)
        self.module = nn.Sequential(
            nn.Linear(2 * self.dim, cfg.lora_rank, bias=False),
            nn.Linear(cfg.lora_rank, self.dim, bias=False),
        ).to(self.device)
        nn.init.zeros_(self.module[-1].weight)
        self._ref = False

    @contextmanager
    def reference(self):
        old, self._ref = self._ref, True
        try:
            yield
        finally:
            self._ref = old

    def schedule(self, steps):
        return torch.linspace(1.0, 0.0, steps + 1).tolist()

    def encode(self, scene):
        if not isinstance(scene, Scene):
            raise ValueError("The tiny simulator accepts structured Scene inputs")
        code = torch.zeros(1, self.dim, device=self.device)
        offset = 0
        for factor in FACTORS:
            choices = VALUES[factor]
            code[0, offset + choices.index(getattr(scene, factor))] = 1.0
            offset += len(choices)
        return code

    def velocity(self, x, sigma, cond):
        cond = cond.expand_as(x)
        drift = 0.6 * x - 3.2 * cond
        if not self._ref:
            drift = drift + self.module(torch.cat((x, cond), dim=-1)) * (
                self.cfg.lora_alpha / self.cfg.lora_rank
            )
        return drift

    def decode(self, latents):
        colors = (
            "#e55555",
            "#3972c7",
            "#469655",
            "#e4c738",
            "#e3873e",
            "#8859af",
            "#db8ab5",
            "#986749",
        )
        frames = []
        for values in latents.detach().float().cpu():
            atoms, style = values[:-4].split(self.widths), values[-4:].tanh().tolist()
            scene = Scene(**{k: VALUES[k][int(a.argmax())] for k, a in zip(FACTORS, atoms)})
            size = self.cfg.resolution
            image = Image.new("RGB", (size, size), tuple(int(188 + 36 * s) for s in style[:3]))
            pen = ImageDraw.Draw(image)
            side = RELATIONS.index(scene.relation)
            dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][side]
            x = size * (0.5 + 0.21 * dx + 0.025 * style[2])
            y = size * (0.48 + 0.21 * dy + 0.025 * style[3])
            ax, ay = size * (0.5 - 0.22 * dx), size * (0.48 - 0.22 * dy)
            pen.rectangle(
                (ax - size * 0.07, ay - size * 0.07, ax + size * 0.07, ay + size * 0.07),
                fill="#484848",
            )
            paint = colors[COLORS.index(scene.color)]
            for j in range(scene.count):
                shift = (j - (scene.count - 1) / 2) * size * 0.12
                cx, cy = x + shift * abs(dy), y + shift * abs(dx)
                r = size * (0.05 + 0.012 * style[0])
                if VALUES["object"].index(scene.object) % 2:
                    pen.ellipse((cx - r, cy - r, cx + r, cy + r), fill=paint)
                else:
                    pen.rectangle((cx - r, cy - r, cx + r, cy + r), fill=paint)
            pen.text((1, size - 12), f"{scene.object}/{scene.landmark}", fill="#323232")
            quality = 0.7 + 0.26 * style[3]
            frames.append(Frame(image, scene, quality))
        return frames


class SD3Policy:
    def __init__(self, cfg, pipeline=None):
        from diffusers import StableDiffusion3Pipeline
        from peft import LoraConfig

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.counts = Counter()
        self.dtype = getattr(torch, cfg.dtype)
        if pipeline is None:
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                cfg.model, revision=cfg.revision, torch_dtype=self.dtype
            )
        self.pipe = pipeline.to(self.device)
        for part in (
            pipeline.transformer,
            pipeline.vae,
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.text_encoder_3,
        ):
            if part is not None:
                part.eval().requires_grad_(False)
        self.module = pipeline.transformer
        self.module.add_adapter(
            LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )
        )
        self.module.enable_gradient_checkpointing()
        for param in self.module.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        spatial = cfg.resolution // pipeline.vae_scale_factor
        self.latent_shape = (self.module.config.in_channels, spatial, spatial)

    @contextmanager
    def reference(self):
        self.module.disable_adapters()
        try:
            yield
        finally:
            self.module.enable_adapters()

    def schedule(self, steps):
        scheduler = self.pipe.scheduler
        kwargs = {}
        cfg = scheduler.config
        if cfg.get("use_dynamic_shifting", False):
            token_count = (self.latent_shape[-1] // self.module.config.patch_size) ** 2
            base_len = cfg.get("base_image_seq_len", 256)
            base_shift = cfg.get("base_shift", 0.5)
            slope = (cfg.get("max_shift", 1.16) - base_shift) / (
                cfg.get("max_image_seq_len", 4096) - base_len
            )
            kwargs["mu"] = base_shift + slope * (token_count - base_len)
        scheduler.set_timesteps(steps, device="cpu", **kwargs)
        return scheduler.sigmas.float().cpu().tolist()

    @torch.no_grad()
    def encode(self, scene):
        text = scene.text if isinstance(scene, Scene) else scene
        embed, neg, pool, neg_pool = self.pipe.encode_prompt(
            prompt=text,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            do_classifier_free_guidance=self.cfg.cfg_scale > 1,
            max_sequence_length=self.cfg.max_text_length,
        )
        return embed, neg, pool, neg_pool

    def velocity(self, x, sigma, cond):
        embed, neg, pool, neg_pool = cond
        batch = len(x)
        seq = embed.expand(batch, -1, -1)
        pooled = pool.expand(batch, -1)
        use_cfg = self.cfg.cfg_scale > 1
        if use_cfg:
            seq = torch.cat([neg.expand_as(seq), seq])
            pooled = torch.cat([neg_pool.expand_as(pooled), pooled])
            x = torch.cat([x, x])
        timesteps = torch.full(
            (len(x),),
            sigma * self.pipe.scheduler.config.num_train_timesteps,
            device=x.device,
            dtype=torch.float32,
        )
        velocity = self.module(
            hidden_states=x.to(self.dtype),
            timestep=timesteps,
            encoder_hidden_states=seq.to(self.dtype),
            pooled_projections=pooled.to(self.dtype),
            return_dict=False,
        )[0].float()
        if use_cfg:
            unconditional, conditional = velocity.chunk(2)
            velocity = unconditional + self.cfg.cfg_scale * (conditional - unconditional)
        return velocity

    @torch.no_grad()
    def decode(self, latents):
        vae = self.pipe.vae
        z = latents.to(device=self.device, dtype=vae.dtype) / vae.config.scaling_factor
        z = z + (getattr(vae.config, "shift_factor", 0.0) or 0.0)
        pixels = vae.decode(z, return_dict=False)[0]
        return [
            Frame(im) for im in self.pipe.image_processor.postprocess(pixels, output_type="pil")
        ]


def make_policy(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.backend == "tiny":
        return TinyPolicy(cfg)
    return SD3Policy(cfg)
