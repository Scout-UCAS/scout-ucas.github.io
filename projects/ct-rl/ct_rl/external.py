"""Run an existing baseline adapter on the same held-out continuation tests.

An adapter implements encode(prefix_text) and answer(latent_state, suffix_text).
The factory receives the user-supplied local weights path. It controls its own
model and token budget. Its states must be reusable across independent queries.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .settings import write_json


def predictions(cases, adapter):
    result = {}
    for case in cases:
        states = [adapter.encode(prefix) for prefix in case["prefixes"]]
        rows = [[adapter.answer(state, suffix) for suffix in case["suffixes"]] for state in states]
        probes = [adapter.answer(state, case["probe"]) for state in states]
        result[case["group"]] = {"answers": rows, "probe_answers": probes}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--factory", required=True, help="Installed adapter module:factory")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    module, name = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    adapter = factory(args.weights)
    write_json(args.out, predictions(json.loads(args.inputs.read_text()), adapter))


if __name__ == "__main__":
    main()
