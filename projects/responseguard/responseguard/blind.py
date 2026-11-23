"""Randomized anonymous A/B image sheets and prompt-level majority outcomes."""

import csv
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from .compare import get_runs
from .io import seed_for, write_json, write_rows

DIMENSIONS = ("semantic", "quality", "diversity")


def blind_study(paths, out, n=200, raters=3, seed=913):
    if len(paths) != 2 or raters < 3:
        raise ValueError("Compare two methods with at least three distinct raters per prompt")
    meta, data = get_runs(paths)
    pool = [(i, b) for i in range(len(data[0])) for b in (0, 1)]
    if not 1 <= n <= len(pool):
        raise ValueError("Sample size exceeds the number of unique prompts")
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose a fresh directory to preserve blind assignments")
    chosen = random.Random(seed).sample(pool, n)
    key, public = [], root / "forms"
    for pos, (idx, side) in enumerate(chosen):
        item = f"P{pos + 1:04d}"
        prompt = data[0][idx]["prompts"][side]["text"]
        groups = [run[idx]["prompts"][side]["samples"] for run in data]
        if [s["seed"] for s in groups[0]] != [s["seed"] for s in groups[1]]:
            raise ValueError("A/B image groups must have identical seeds")
        # Both full groups are shown, including failed images. Cards contain no model identifiers.
        for r in range(raters):
            rater = f"rater-{r + 1:02d}"
            flip = random.Random(seed_for("blind", seed, item, rater)).randrange(2)
            key.append(
                dict(item=item, rater=rater, method_a=flip, pair=data[0][idx]["pair"], side=side)
            )
            dest = public / rater
            dest.mkdir(parents=True, exist_ok=True)
            k = len(groups[0])
            size, cols = 112, min(8, k)
            nh = (k + cols - 1) // cols
            card = Image.new("RGB", (max(600, cols * size), 48 + 2 * (nh * size + 24)), "white")
            draw = ImageDraw.Draw(card)
            draw.text((4, 6), prompt, fill="black")
            for arm, j in enumerate((flip, 1 - flip)):
                top = 40 + arm * (nh * size + 24)
                draw.text((4, top - 10), "AB"[arm], fill="black")
                for i, s in enumerate(groups[j]):
                    with Image.open(Path(paths[j]) / s["image"]) as im:
                        card.paste(
                            im.convert("RGB").resize((size, size)),
                            (size * (i % cols), top + 8 + size * (i // cols)),
                        )
            card.save(dest / (item + ".png"))
    for r in range(raters):
        path = public / f"rater-{r + 1:02d}" / "ratings.csv"
        with path.open("w", newline="", encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(["item", *DIMENSIONS])
            w.writerows([[f"P{i + 1:04d}", "", "", ""] for i in range(n)])
    write_rows(root / "private" / "assignments.jsonl", key)
    write_json(
        root / "private" / "protocol.json",
        dict(methods=[m["model"] for m in meta], raters=raters, n=n, seed=seed),
    )
    return dict(prompts=n, raters=raters, cards=n * raters)


def blind_scores(out):
    root = Path(out)
    protocol = json.loads((root / "private" / "protocol.json").read_text())
    key = {
        (r["item"], r["rater"]): r["method_a"]
        for r in map(json.loads, (root / "private" / "assignments.jsonl").read_text().splitlines())
    }
    votes = {}
    completed = 0
    for path in sorted((root / "forms").glob("*/ratings.csv")):
        rater = path.parent.name
        with path.open(newline="", encoding="utf-8-sig") as h:
            for row in csv.DictReader(h):
                cell = (row["item"], rater)
                if cell not in key or cell in votes:
                    raise ValueError(f"Unexpected or repeated rating row {cell}")
                flip = key[cell]
                parsed = {}
                for dim in DIMENSIONS:
                    choice = row[dim].strip().upper()
                    if choice not in {"", "A", "B", "TIE"}:
                        raise ValueError("Use A, B or TIE in each rating field")
                    if choice:
                        parsed[dim] = (
                            str(flip if choice == "A" else 1 - flip) if choice != "TIE" else "tie"
                        )
                votes[cell] = parsed
                completed += len(parsed) == len(DIMENSIONS)
    scores = {}
    for d in DIMENSIONS:
        counts = Counter()
        for item in {i for i, _ in key}:
            vals = [votes.get((i, r), {}).get(d) for i, r in key if i == item]
            if any(v is None for v in vals):
                continue
            tally = Counter(vals)
            winner, top = tally.most_common(1)[0]
            counts[winner if top > len(vals) / 2 else "tie"] += 1
        scores[d] = dict(
            method_0=counts["0"],
            method_1=counts["1"],
            tie=counts["tie"],
            rated_prompts=sum(counts.values()),
        )
    return dict(**protocol, complete_ratings=completed, expected_ratings=len(key), outcomes=scores)
