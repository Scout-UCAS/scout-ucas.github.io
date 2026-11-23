"""Frozen atom verification, image quality, preference and visual representations."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import FACTORS

QUALITY_QUESTION = "Is this a clear, visually coherent image without obvious visual artifacts?"


@dataclass
class Evidence:
    own: torch.Tensor  # [images, atoms]
    other: torch.Tensor  # same atom probed under its counterfactual value
    quality: torch.Tensor
    preference: torch.Tensor
    features: torch.Tensor

    def select(self, indices):
        return Evidence(
            self.own[indices],
            self.other[indices],
            self.quality[indices],
            self.preference[indices],
            self.features[indices],
        )

    def semantic(self, cfg):
        return self.own.amin(1) >= cfg.sem_threshold

    def valid(self, cfg):
        return self.semantic(cfg) & (self.quality >= cfg.quality_threshold)

    def margin(self, factor):
        j = FACTORS.index(factor)
        return self.own[:, j] - self.other[:, j]

    def reward(self, cfg):
        pref, qual, sem = cfg.task_weights
        return pref * self.preference + qual * self.quality + sem * self.own.mean(1)


def alt_scenes(scene, factor, counterpart):
    return {k: counterpart if k == factor else scene.neighbor(k) for k in FACTORS}


def image_vectors(frames, size):
    pixels = np.stack(
        [np.asarray(frame.image.resize((size, size)), dtype=np.float32).ravel() for frame in frames]
    )
    return F.normalize(torch.from_numpy(pixels) / 255.0, dim=1)


class TinyJudge:
    def __init__(self, evaluation=False):
        self.evaluation = evaluation
        self.counts = Counter()

    def observe(self, frames, scene, alternatives):
        def score(frame, claim, key):
            return 0.96 if getattr(frame.scene, key) == getattr(claim, key) else 0.04

        own = torch.tensor([[score(f, scene, k) for k in FACTORS] for f in frames])
        other = torch.tensor([[score(f, alternatives[k], k) for k in FACTORS] for f in frames])
        quality = torch.tensor([f.quality for f in frames])
        # An intentionally imperfect aesthetic proxy makes task-only optimization observable.
        pref = torch.tensor(
            [np.asarray(f.image)[:, :, 0].mean() / 255.0 for f in frames], dtype=torch.float32
        )
        if self.evaluation:
            # Independent toy photograph proxy uses contrast, whereas the train reward uses red.
            pref = torch.tensor(
                [np.asarray(f.image).astype(np.float32).std() / 128 for f in frames]
            )
        self.counts.update(
            semantic_questions=2 * len(FACTORS) * len(frames),
            quality_scores=len(frames),
            preference_scores=len(frames),
            feature_images=len(frames),
        )
        return Evidence(own, other, quality, pref, image_vectors(frames, 12))


class ModelJudge:
    def __init__(self, cfg, evaluation=False):
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoProcessor,
            BlipForQuestionAnswering,
        )

        self.device = torch.device(cfg.judge_device)
        # FP32 on CPU, BF16 if requested/supported on the chosen GPU.
        self.dtype = torch.float32 if self.device.type == "cpu" else getattr(torch, cfg.dtype)
        self.micro = cfg.microbatch
        self.counts = Counter()

        def revision(repo):
            return cfg.judge_revisions.get(repo, cfg.revision_judges)

        def frozen(model):
            return model.to(device=self.device, dtype=self.dtype).requires_grad_(False).eval()

        vqa_id = cfg.eval_vqa if evaluation else cfg.train_vqa
        self.vqa = frozen(
            BlipForQuestionAnswering.from_pretrained(vqa_id, revision=revision(vqa_id))
        )
        self.vqa_p = AutoProcessor.from_pretrained(vqa_id, revision=revision(vqa_id))
        self.answer_ids = [
            self.vqa_p.tokenizer.encode(s, add_special_tokens=False) for s in ("no", "yes")
        ]
        if any(len(ids) != 1 for ids in self.answer_ids):
            raise ValueError("BLIP binary VQA requires single-token yes/no answers")
        feature_id = cfg.eval_feature_model if evaluation else cfg.feature_model
        self.feat = frozen(AutoModel.from_pretrained(feature_id, revision=revision(feature_id)))
        self.feat_p = AutoImageProcessor.from_pretrained(feature_id, revision=revision(feature_id))
        pref_id = cfg.eval_preference_model if evaluation else cfg.preference_model
        self.hps = pref_id == "xswu/HPSv2"
        if self.hps:
            import open_clip
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(pref_id, "HPS_v2.1_compressed.pt", revision=revision(pref_id))
            prep = open_clip.get_pretrained_cfg("ViT-H-14", "laion2b_s32b_b79k")
            model, _, self.pick_p = open_clip.create_model_and_transforms(
                "ViT-H-14", image_mean=prep.get("mean"), image_std=prep.get("std")
            )
            open_clip.load_checkpoint(model, path, strict=True)
            self.pick = frozen(model)
            self.pick_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        else:
            processor_id = (
                (cfg.eval_preference_processor or pref_id)
                if evaluation
                else cfg.preference_processor
            )
            self.pick = frozen(AutoModel.from_pretrained(pref_id, revision=revision(pref_id)))
            self.pick_p = AutoProcessor.from_pretrained(
                processor_id, revision=revision(processor_id)
            )

    def binary_answers(self, image, questions):
        """One frozen image embedding reused across all binary questions for that image."""
        proc, model = self.vqa_p, self.vqa
        px = proc.image_processor(images=image, return_tensors="pt").pixel_values.to(
            self.device, self.dtype
        )
        vision = model.vision_model(px).last_hidden_state.expand(len(questions), -1, -1)
        tokens = proc.tokenizer(questions, padding=True, return_tensors="pt").to(self.device)
        mask = torch.ones(vision.shape[:2], dtype=torch.long, device=self.device)
        embeds = model.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            encoder_hidden_states=vision,
            encoder_attention_mask=mask,
        ).last_hidden_state
        bos = torch.full(
            (len(questions), 1), model.decoder_start_token_id, dtype=torch.long, device=self.device
        )
        logits = model.text_decoder(
            input_ids=bos,
            encoder_hidden_states=embeds,
            encoder_attention_mask=tokens.attention_mask,
        ).logits
        return logits[:, -1, [i[0] for i in self.answer_ids]].float().softmax(-1)[:, 1].cpu()

    @torch.no_grad()
    def score_images(self, frames, prompt, atom_questions):
        scores = torch.stack(
            [self.binary_answers(f.image, [*atom_questions, QUALITY_QUESTION]) for f in frames]
        )
        if self.hps:
            txt = self.pick.encode_text(self.pick_tokenizer([prompt]).to(self.device))
        else:
            text = self.pick_p(
                text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)
            txt = self.pick.get_text_features(**text)
        txt = F.normalize(txt.float(), dim=-1)
        features, preferences = [], []
        for start in range(0, len(frames), self.micro):
            imgs = [f.image for f in frames[start : start + self.micro]]
            px = self.feat_p(images=imgs, return_tensors="pt").pixel_values.to(
                self.device, self.dtype
            )
            features.append(
                F.normalize(self.feat(pixel_values=px).last_hidden_state[:, 0].float(), dim=1).cpu()
            )
            if self.hps:
                pp = torch.stack([self.pick_p(im) for im in imgs]).to(self.device, self.dtype)
                img = self.pick.encode_image(pp)
            else:
                pp = self.pick_p(images=imgs, return_tensors="pt").pixel_values.to(
                    self.device, self.dtype
                )
                img = self.pick.get_image_features(pixel_values=pp)
            img = F.normalize(img.float(), dim=-1)
            # Absolute per-image text-image cosine.
            preferences.append((img @ txt.T).flatten().cpu())
        self.counts.update(
            semantic_questions=len(frames) * len(atom_questions),
            quality_scores=len(frames),
            preference_scores=len(frames),
            feature_images=len(frames),
        )
        return scores[:, :-1], scores[:, -1], torch.cat(preferences), torch.cat(features)

    def observe(self, frames, scene, alternatives):
        questions = [scene.atom_question(k) for k in FACTORS] + [
            alternatives[k].atom_question(k) for k in FACTORS
        ]
        scores, quality, preference, features = self.score_images(frames, scene.text, questions)
        n = len(FACTORS)
        return Evidence(scores[:, :n], scores[:, n:], quality, preference, features)


def make_judge(cfg, evaluation=False):
    return TinyJudge(evaluation) if cfg.backend == "tiny" else ModelJudge(cfg, evaluation)


class PerceptualSpace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.label = "tiny_rgb_cosine" if cfg.backend == "tiny" else "dreamsim_ensemble_cosine"
        self.rgb = cfg.backend == "tiny" or cfg.perceptual_metric == "rgb"
        if self.rgb and cfg.backend == "sd3":
            self.label = "diagnostic_rgb_cosine"
        if not self.rgb:
            from dreamsim import dreamsim

            self.model, self.preprocess = dreamsim(
                pretrained=True, device=cfg.judge_device, dreamsim_type="ensemble"
            )
            self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def embed(self, frames):
        if self.rgb:
            return image_vectors(frames, 24)
        vectors = []
        for start in range(0, len(frames), self.cfg.microbatch):
            x = torch.cat(
                [self.preprocess(f.image) for f in frames[start : start + self.cfg.microbatch]]
            )
            feat = self.model.embed(x.to(self.cfg.judge_device)).float().cpu()
            if feat.ndim != 2:
                raise ValueError("Use the DreamSim ensemble with per-image vector embeddings")
            vectors.append(F.normalize(feat, dim=-1))
        return torch.cat(vectors)
