import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .io import digest


@dataclass(frozen=True)
class Config:
    backend: str = "sd3"
    device: str = "cuda"
    judge_device: str = "cuda"
    dtype: str = "bfloat16"
    model: str = "stabilityai/stable-diffusion-3.5-medium"
    revision: str = "main"
    train_vqa: str = "Salesforce/blip-vqa-base"
    eval_vqa: str = "Salesforce/blip-vqa-capfilt-large"
    feature_model: str = "facebook/dinov2-small"
    preference_model: str = "yuvalkirstain/PickScore_v1"
    preference_processor: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    eval_feature_model: str = "facebook/dinov2-base"
    eval_preference_model: str = "xswu/HPSv2"
    eval_preference_processor: str = ""
    perceptual_metric: str = "dreamsim"
    pair_sampling: str = "with_replacement"
    supervised_margin_weight: float = 0.2
    diverse_clusters: int = 2
    diverse_weight: float = 0.7
    diverse_early_steps: int = 4
    diverse_wasserstein: float = 0.04
    drift_weight: float = 0.6
    drift_cap: float = 1.0
    drift_prompt_noise: float = 0.05
    drift_noise_start: float = 0.4
    drift_noise_end: float = 1.0
    revision_judges: str = "main"
    judge_revisions: dict = field(default_factory=dict)
    resolution: int = 512
    cfg_scale: float = 4.5
    max_text_length: int = 128
    seed: int = 42
    # Fixed across training seeds and methods.
    reference_seed: int = 2718
    evaluation_seed: int = 31415
    train_steps: int = 10
    eval_steps: int = 28
    sde_noise: float = 0.55
    k_train: int = 4
    k_ref: int = 16
    k_cal: int = 16
    k_eval: int = 16
    projection_rank: int = 4
    sem_threshold: float = 0.65
    quality_threshold: float = 0.6
    anchor_margin: float = 0.25
    margin_slack: float = 0.08
    kappa: float = 0.8
    rho_floor: float = 0.5
    rho_slack: float = 0.1
    rate_weight: float = 1.0
    task_weights: tuple = (0.45, 0.20, 0.35)  # preference, quality, mean atom compliance
    sem_budget: float = 0.025
    div_budget: float = 0.025
    dual_initial: tuple = (0.2, 0.2)
    dual_lr: tuple = (0.05, 0.05)
    history_window: int = 256
    scale_floor: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 16
    lr: float = 1.0e-5
    adam_betas: tuple = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_eps: float = 0.1
    flow_clip_eps: float = 0.001
    flow_kl: float = 0.04
    ppo_epochs: int = 2
    groups_per_batch: int = 2
    microbatch: int = 1
    iterations: int = 1000
    checkpoint_every: int = 10
    vendi_m: int = 4

    def __post_init__(self):
        if self.pair_sampling not in {"with_replacement", "shuffled_epochs"}:
            raise ValueError("Unknown pair sampling protocol")
        if self.perceptual_metric not in {"dreamsim", "rgb"}:
            raise ValueError("Unknown perceptual metric")
        if not 0 <= self.drift_noise_start < self.drift_noise_end <= 1:
            raise ValueError("Invalid DRIFT time window")
        if (
            min(self.diverse_clusters, self.diverse_early_steps) < 1
            or min(
                self.diverse_weight,
                self.diverse_wasserstein,
                self.drift_weight,
                self.drift_cap,
                self.drift_prompt_noise,
            )
            < 0
        ):
            raise ValueError("Invalid baseline parameters")
        if self.backend not in {"sd3", "tiny"} or self.dtype not in {
            "float32",
            "bfloat16",
            "float16",
        }:
            raise ValueError("Unsupported backend or dtype")
        for key in (
            "k_cal",
            "projection_rank",
            "history_window",
            "lora_rank",
            "lora_alpha",
            "microbatch",
            "iterations",
            "checkpoint_every",
            "ppo_epochs",
            "groups_per_batch",
            "max_text_length",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if (
            min(self.train_steps, self.eval_steps, self.k_train, self.k_eval) < 2
            or self.k_ref < self.k_train
        ):
            raise ValueError("Require at least two steps/seeds and k_ref >= k_train")
        if not 2 <= self.vendi_m <= self.k_eval:
            raise ValueError("Require 2 <= vendi_m <= k_eval")
        if not 1 / self.k_train < self.rho_floor <= 1:
            raise ValueError("rho_floor must penalize a group with at most one valid image")
        for key in (
            "sem_threshold",
            "quality_threshold",
            "anchor_margin",
            "margin_slack",
            "rho_slack",
            "kappa",
        ):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must lie in [0,1]")
        if (
            not 0 < self.clip_eps < 1
            or min(self.sde_noise, self.lr, self.max_grad_norm, self.scale_floor) <= 0
        ):
            raise ValueError("Invalid noise, clipping or optimizer setting")
        if (
            len(self.adam_betas) != 2
            or not all(0 <= x < 1 for x in self.adam_betas)
            or self.adam_epsilon <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid AdamW settings")
        if not 0 < self.flow_clip_eps < 1 or self.flow_kl < 0:
            raise ValueError("Invalid Flow-GRPO baseline settings")
        if self.resolution < 32 or self.resolution % 16 or self.cfg_scale < 1:
            raise ValueError("Use resolution divisible by 16 and cfg_scale >= 1")
        if not (len(self.dual_lr) == len(self.dual_initial) == 2 and len(self.task_weights) == 3):
            raise ValueError("Expected two dual weights and three task weights")
        if (
            min(
                *self.dual_lr,
                *self.dual_initial,
                *self.task_weights,
                self.sem_budget,
                self.div_budget,
                self.rate_weight,
            )
            < 0
        ):
            raise ValueError("Penalties and weights must be nonnegative")
        if not math.isclose(sum(self.task_weights), 1.0):
            raise ValueError("task_weights must sum to one")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def signature(self, *keys):
        vals = asdict(self)
        return digest({k: vals[k] for k in keys})
