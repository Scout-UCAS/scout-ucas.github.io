"""Fetch pinned primary sources and build question-disjoint public splits."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .public import read_public
from .settings import atomic_path, digest, write_json

RAW = "https://raw.githubusercontent.com/"
GSM = RAW+"openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/"
PROS = RAW+"facebookresearch/coconut/27273cb8cca4bb763c041a63b036d0c3b7cbbb48/data/"
PRONTO = RAW+"asaparov/prontoqa/f0145b867b3c106285ec9ea1941a3f6eb7c6162d/"
MATH = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl"
FILES = {
    "gsm_train.jsonl": (GSM+"train.jsonl", "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
    "gsm_test.jsonl": (GSM+"test.jsonl", "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
    "pros_train.json": (PROS+"prosqa_train.json", "99e40ce7e9107fd02e35bcd78a7a4479bd57f415bab0a6efd37cc11a72e594f6"),
    "pros_valid.json": (PROS+"prosqa_valid.json", "c74e0de24f1e90ec48b1a993e6458a7be4caeef3928f5104e3c2e45d689a5249"),
    "pros_test.json": (PROS+"prosqa_test.json", "82ddc0afe6ee30eb31bb16fa208e293e55278a20fa987098d48519cae7e12680"),
    "math.jsonl": (MATH, "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"),
}
GEN = {
    "run_experiment.py":"5f6a60f62abba6c58f16ac8feced5e73aea996194a058550e181be08808eb461",
    "syntax.py":"53b0979f30f34d5f12da4ef8a17ca2d62a8e9def6c8c45e80ff8fb7cff52cc84",
    "theory.py":"7b9d2284948fa336ad5e5fb940fb0e6ce3129c9002d7b9ae9634fb8603058e98",
    "proof.py":"30581dcd872660378d49c80b9645f7091694ab0ae43e989ca86348ea71456790",
    "prompt.py":"40e6e498de96ef08074f15313c3ddbd2d4489160a573624d848bb2ddbefacd51",
    "fol.py":"50fc3cd1c10daab9500d834f5212f126360081427cc331cc3380bf8bea6ea18c",
    "bad_patterns.txt":"a0a05bc48adae8181a51a836ed47179eef2d1a761b033b398ca0523d4ac9f632",
}


def download(file, url, expected):
    if file.exists() and digest(file) == expected:
        return
    with atomic_path(file) as tmp:
        with urllib.request.urlopen(url, timeout=90) as src, tmp.open("wb") as dst:
            while block := src.read(1 << 20):
                dst.write(block)
        if digest(tmp) != expected:
            raise ValueError(f"Upstream checksum mismatch for {file.name}")


def generated(cache, smoke):
    trials = 64 if smoke else 10000
    folder = cache/f"pronto-v1-{trials}"
    for filename, checksum in GEN.items():
        download(folder/filename, PRONTO+filename, checksum)
    result = folder/"5hop_0shot_random_seed120.json"
    if not result.exists() or len(json.loads(result.read_text())) != trials:
        command = [sys.executable, str(folder.resolve()/"run_experiment.py"), "--model-name", "json",
                   "--model-size", "dummy", "--ordering", "random", "--num-trials", str(trials),
                   "--few-shot-examples", "0", "--ontology", "fictional", "--min-hops", "5",
                   "--max-hops", "5", "--hops-skip", "1", "--seed", "120"]
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            (Path(tmp)/"bad_patterns.txt").write_bytes((folder/"bad_patterns.txt").read_bytes())
            subprocess.run(command, cwd=tmp, env={**os.environ, "PYTHONHASHSEED":"120"},
                           stdout=subprocess.DEVNULL, check=True)
            with atomic_path(result) as dst:
                dst.write_bytes((Path(tmp)/result.name).read_bytes())
    return result


def key(item):
    return " ".join(item.question.casefold().split())


def prepared(root, profile):
    m = json.loads((root/"manifest.json").read_text())
    if m["profile"] != profile:
        raise ValueError("Public split profile differs")
    for name, entry in m["splits"].items():
        if digest(root/name) != entry["sha256"]:
            raise ValueError(f"Public data changed: {name}")
    return m


def prepare(root, cache, profile):
    if (root/"manifest.json").exists():
        return prepared(root, profile)
    smoke = profile == "smoke"
    for name, (url, checksum) in FILES.items():
        download(cache/name, url, checksum)
    pronto_file = generated(cache, smoke)
    rng = random.Random(120)
    def shuffled(items):
        by_key = {key(i): i for i in items}
        result = [by_key[k] for k in sorted(by_key)]
        rng.shuffle(result)
        return result
    gsm = shuffled(read_public("gsm8k", cache/"gsm_train.jsonl"))
    pronto = shuffled(read_public("prontoqa", pronto_file))
    pv, pt = (8, 8) if smoke else (500, 500)
    source = {"gsm8k":{"train":gsm[500:], "valid":gsm[:500], "test":read_public("gsm8k", cache/"gsm_test.jsonl")},
              "prontoqa":{"train":pronto[pv+pt:], "valid":pronto[:pv], "test":pronto[pv:pv+pt]},
              "prosqa":{s:read_public("prosqa", cache/("pros_"+s+".json")) for s in ("train","valid","test")},
              "math500":{"test":read_public("math500", cache/"math.jsonl")}}
    manifest = {"profile":profile, "seed":120, "upstream":FILES, "pronto_generator":PRONTO,
                "generator_files":GEN, "generated_questions_sha256":digest(pronto_file), "splits":{}}
    for dataset, splits in source.items():
        seen = set()
        for part in ("test", "valid", "train"):
            if part not in splits:
                continue
            original = splits[part]
            if part == "test" and not smoke:
                rows = list(original)  # Preserve the entire official test, including order.
            else:
                rows = [i for i in shuffled(original) if key(i) not in seen]
            seen.update(key(i) for i in rows)
            if smoke:
                rows = rows[:8 if part == "train" else 3]
            path = root/dataset/(part+".json")
            write_json(path, [asdict(i) for i in rows])
            manifest["splits"][str(path.relative_to(root))] = {"count":len(rows), "source_count":len(original),
                                                                "sha256":digest(path)}
    write_json(root/"manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("smoke","paper"), required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    prepare(args.out, args.cache, args.profile)
