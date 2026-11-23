"""Offline integration with real Diffusers, PEFT and Transformers classes at reduced size."""

from dataclasses import replace
import json

import pytest
import torch
from PIL import Image

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from diffusers import StableDiffusion3Pipeline
from scripts.smoke_models import make_local_models

from responseguard.benchmarks import benchmark_export, generate_text
from responseguard.cli import export_lora
from responseguard.config import Config
from responseguard.data import Pair, Scene
from responseguard.flow import rollout
from responseguard.policy import make_policy
from responseguard.train import load_checkpoint, train
from responseguard.judges import make_judge, alt_scenes


def test_sd3_training_reference_replay_checkpoint_and_exports(tmp_path):
    torch.manual_seed(1)
    torch.set_num_threads(1)
    make_local_models(tmp_path)
    cfg = replace(
        Config(),
        backend="sd3",
        device="cpu",
        judge_device="cpu",
        dtype="float32",
        model=str(tmp_path / "sd3"),
        train_vqa=str(tmp_path / "vqa"),
        eval_vqa=str(tmp_path / "vqa"),
        feature_model=str(tmp_path / "dino"),
        eval_feature_model=str(tmp_path / "dino"),
        eval_preference_model=str(tmp_path / "pick"),
        perceptual_metric="rgb",
        preference_model=str(tmp_path / "pick"),
        preference_processor=str(tmp_path / "pick"),
        k_train=2,
        k_ref=2,
        k_cal=2,
        k_eval=2,
        vendi_m=2,
        rho_floor=1.0,
        train_steps=2,
        eval_steps=2,
        resolution=32,
        max_text_length=16,
        groups_per_batch=1,
        microbatch=2,
        ppo_epochs=2,
        lora_rank=2,
        lora_alpha=4,
        lr=0.0001,
        iterations=1,
    )
    s = Scene("cup", "box", "blue", 2, "above")
    p = Pair(s, s.neighbor("color"), "color", "train")
    trained = train(cfg, [p], tmp_path / "run", tmp_path / "cache")
    model = make_policy(cfg)
    state = load_checkpoint(model, trained)
    assert any("lora_B" in k and bool(v.abs().sum()) for k, v in state["adapter"].items())
    cur = rollout(model, s, [1, 2], 42, 2, True)
    with model.reference():
        base = rollout(model, s, [1, 2], 42, 2, True)
    assert not torch.equal(cur.trace.means[0], base.trace.means[0])
    assert cur.frames[0].image.size == (32, 32)
    ev = make_judge(cfg, evaluation=True).observe(cur.frames, s, alt_scenes(s, p.factor, p.right))
    assert ev.own.shape == (2, 5) and ev.features.shape[0] == 2 and torch.isfinite(ev.quality).all()
    export_lora(cfg, trained, tmp_path / "lora")
    assert list((tmp_path / "lora").glob("*.safetensors"))
    # A fresh vanilla pipeline must be able to consume the exported Diffusers adapter.
    pipe = StableDiffusion3Pipeline.from_pretrained(cfg.model)
    pipe.load_lora_weights(tmp_path / "lora", weight_name="pytorch_lora_weights.safetensors")
    assert any(
        "lora_B" in n and bool(w.abs().sum()) for n, w in pipe.transformer.named_parameters()
    )
    assert next(iter(pipe.transformer.peft_config.values())).lora_alpha == cfg.lora_alpha
    generate_text(cfg, "A red cup.", [4], tmp_path / "inference", trained)
    with Image.open(tmp_path / "inference" / "000_4.png") as im:
        assert im.size == (32, 32)
    (tmp_path / "ge.jsonl").write_text(json.dumps({"prompt": "A cup."}) + "\n")
    benchmark_export(cfg, tmp_path / "ge.jsonl", "geneval2", tmp_path / "ge", trained)
    assert json.loads((tmp_path / "ge" / "image_paths.json").read_text())["A cup."].endswith(".png")
    (tmp_path / "pc.json").write_text(json.dumps([{"caption1": "A cup.", "caption2": "A mug."}]))
    benchmark_export(cfg, tmp_path / "pc.json", "paircomp", tmp_path / "pc", trained)
    assert sorted(p.name for p in (tmp_path / "pc").glob("*.png")) == [
        "0_0_0.png",
        "0_0_1.png",
        "0_1_0.png",
        "0_1_1.png",
    ]
    delta = {
        "category": "count",
        "caption1": "One cup.",
        "caption2": "Two cups.",
        "questions1": ["Is there exactly one cup?"],
        "questions2": ["Are there exactly two cups?"],
    }
    (tmp_path / "delta.json").write_text(json.dumps([delta]))
    benchmark_export(cfg, tmp_path / "delta.json", "deltabench", tmp_path / "delta", trained)
    manifest = [
        json.loads(s) for s in (tmp_path / "delta" / "images.jsonl").read_text().splitlines()
    ]
    assert len(manifest) == 4 and manifest[-1]["questions"] == delta["questions2"]
