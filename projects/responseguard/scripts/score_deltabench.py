"""Score supplied DeltaBench questions in a separate Transformers 4.57+ environment."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean


def summary(rows):
    pairs = defaultdict(dict)
    for r in rows:
        group = pairs[r["category"], r["pair"]]
        cell = (r["side"], r["index"])
        if cell in group:
            raise ValueError(f"Duplicate comparison cell {cell}")
        group[cell] = r["score"]
    categories = defaultdict(list)
    for (cat, pair), cells in pairs.items():
        if set(cells) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError(f"Expected four images for pair {pair}")
        vals = list(cells.values())
        categories[cat].append((fmean(vals), math.prod(vals) ** 0.25))
    by_cat = {
        c: dict(am=fmean(v[0] for v in vals), gm=fmean(v[1] for v in vals), pairs=len(vals))
        for c, vals in categories.items()
    }
    return {
        "categories": by_cat,
        "macro_am": fmean(s["am"] for s in by_cat.values()),
        "macro_gm": fmean(s["gm"] for s in by_cat.values()),
    }


def score_manifest(rows, directory, target, predict_yes, metadata):
    from PIL import Image

    if not rows or any(not r.get("questions") for r in rows):
        raise ValueError("Both captions must have independently supplied questions")
    for row in rows:
        with Image.open(directory / row["image"]) as img:
            image = img.convert("RGB")
            questions = [
                item if isinstance(item, str) else item["question"] for item in row["questions"]
            ]
            answers = [
                "yes" if isinstance(item, str) else item["answer"].strip().lower()
                for item in row["questions"]
            ]
            probs = list(predict_yes(image, questions))
            if (
                len(probs) != len(questions)
                or any(not 0 <= p <= 1 for p in probs)
                or not set(answers) <= {"yes", "no"}
            ):
                raise ValueError(
                    "Each binary question requires one finite probability and a yes/no target"
                )
            scores = [float(p if answer == "yes" else 1 - p) for p, answer in zip(probs, answers)]
            row["question_scores"] = scores
            row["score"] = fmean(scores)
    target.mkdir(parents=True, exist_ok=True)
    (target / "image_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (target / "summary.json").write_text(json.dumps(dict(**metadata, **summary(rows)), indent=2))
    return summary(rows)


def evaluate(manifest, target, model_id, revision):
    import torch
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    if not rows or any(not r["questions"] for r in rows):
        raise ValueError(
            "Supply the benchmark's global and detail questions for each side (questions1, questions2)"
        )
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype="auto", device_map="auto"
    ).eval()
    answers = []
    for word in ("no", "yes"):
        candidates = (word, word.title(), " " + word, " " + word.title())
        ids = [processor.tokenizer.encode(s, add_special_tokens=False) for s in candidates]
        answers.append(sorted({tokens[0] for tokens in ids if len(tokens) == 1}))
    if not all(answers) or set(answers[0]) & set(answers[1]):
        raise ValueError("Expected two disjoint sets of single-token yes/no answers")

    @torch.inference_mode()
    def predict(image, questions):
        scores = []
        for q in questions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q + " Answer yes or no."},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            logits = model(**tokens, logits_to_keep=1).logits[0, -1].float()
            mass = torch.stack([logits[ix].logsumexp(0) for ix in answers])
            scores.append(float(mass.softmax(0)[1]))
        return scores

    return score_manifest(
        rows, manifest.parent, target, predict, dict(model=model_id, revision=revision)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    evaluate(args.manifest, args.out, args.model, args.revision)
