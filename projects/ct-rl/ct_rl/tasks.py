"""Deterministic CT-Bench program generators and interpreters.

A Pair is the unit of splitting. The executor state is never a model input.
One suffix instruction has unit cost, including its terminal readout.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any, Iterable

FAMILIES = ("arithmetic", "graph", "rules")
SPLITS = ("train", "valid", "id", "comp", "prefix_length", "suffix_length", "scale")
READOUT = {"arithmetic": "sum", "graph": "parity", "rules": "count"}
ARITH_READS = ("sum", "diff", "left", "right", "max")
ARITH_OPS = ("swap", "raise", "mix", "neg")
HELD_OUT = {"arithmetic": ("swap", "left"), "graph": ("R", "node"),
            "rules": ("add", "has")}
Suffix = tuple[str, ...]


def stable_id(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def graph_key(world):
    """Canonical form under node renaming for a pair of labelled permutations."""
    remaining = set(range(len(world["L"])))
    components = []
    while remaining:
        def explore(root):
            nodes = [root]
            numbers = {root: 0}
            edges = []
            for vertex in nodes:
                for op in ("L", "R"):
                    dest = world[op][vertex]
                    if dest not in numbers:
                        numbers[dest] = len(nodes)
                        nodes.append(dest)
                    edges.append(numbers[dest])
            return nodes, tuple(edges)
        component, _ = explore(min(remaining))
        components.append(min(explore(root)[1] for root in component))
        remaining.difference_update(component)
    return sorted(components)


def rule_key(world):
    """Isomorphism-invariant color refinement. Collisions co-locate objects."""
    n = world["n"]
    incoming, outgoing = [[] for _ in range(n)], [[] for _ in range(n)]
    for a, b in world["edges"]:
        incoming[b].append(a)
        outgoing[a].append(b)
    colors = [0]*n
    for _ in range(n):
        signatures = [(colors[i], tuple(sorted(colors[j] for j in incoming[i])),
                       tuple(sorted(colors[j] for j in outgoing[i]))) for i in range(n)]
        mapping = {v:k for k,v in enumerate(sorted(set(signatures)))}
        previous_count = len(set(colors))
        colors = [mapping[v] for v in signatures]
        if len(mapping) == previous_count:
            break
    return sorted(colors), sorted((colors[a], colors[b]) for a,b in world["edges"])


@dataclass(frozen=True)
class Pair:
    family: str
    world: dict
    starts: list
    operations: list
    split: str

    @cached_property
    def group(self) -> str:
        # All queries, start variants and paired sides on one underlying object
        # belong to one partition, even when sampling with a different seed.
        if self.family == "arithmetic":
            renamed = [[{"shift_a":"shift_b","shift_b":"shift_a"}.get(op, op),
                        -k if op == "xfer" else k] for op,k in self.operations]
            underlying = min(self.operations, renamed)
        elif self.family == "graph":
            underlying = graph_key(self.world)
        else:
            underlying = rule_key(self.world)
        return stable_id([self.family, underlying])

    def record(self):
        return asdict(self)

    def prefix(self, side: int) -> str:
        start = self.starts[side]
        if self.family == "arithmetic":
            steps = " ".join(f"{op}({k})" if k else op for op, k in self.operations)
            return f"arithmetic a={start[0]} b={start[1]}; {steps}"
        if self.family == "graph":
            edges = " ".join(f"{i}:{l},{r}" for i, (l, r) in
                             enumerate(zip(self.world["L"], self.world["R"])))
            return f"graph edges(L,R) {edges}; start {start}; walk {' '.join(self.operations)}"
        facts = ",".join(map(str, start))
        rules = " ".join(f"{a}>{b}" for a, b in self.world["edges"])
        return f"rules 0..{self.world['n']-1}; {rules}; facts {facts}; close"


def closure(edges: list, facts: set[int]) -> set[int]:
    result = set(facts)
    while True:
        new = {b for a, b in edges if a in result} - result
        if not new:
            return result
        result.update(new)


def prefix_state(pair: Pair, side: int):
    initial = pair.starts[side]
    if pair.family == "arithmetic":
        a, b = initial
        for op, k in pair.operations:
            if op == "swap":
                a, b = b, a
            elif op == "shift_a":
                a += k
            elif op == "shift_b":
                b += k
            elif op == "xfer":
                a, b = a + k, b - k
            elif op == "flip":
                a, b = -a, -b
            else:
                raise ValueError(op)
        return a, b
    if pair.family == "graph":
        node = initial
        for op in pair.operations:
            node = pair.world[op][node]
        return node
    return closure(pair.world["edges"], set(initial))


def execute(pair: Pair, side: int, suffix: Suffix) -> str:
    state = prefix_state(pair, side)
    *steps, read = suffix
    if pair.family == "arithmetic":
        a, b = state
        for op in steps:
            if op == "swap":
                a, b = b, a
            elif op == "raise":
                a += 2
            elif op == "mix":
                a += b
            elif op == "neg":
                b = -b
            else:
                raise ValueError(op)
        return str({"sum": a+b, "diff": a-b, "left": a, "right": b, "max": max(a, b)}[read])
    if pair.family == "graph":
        for op in steps:
            state = pair.world[op][state]
        if read not in ("node", "parity"):
            raise ValueError(read)
        return str(state if read == "node" else state % 2)
    for op in steps:
        verb, value = op.split()
        if verb != "add":
            raise ValueError(op)
        state = closure(pair.world["edges"], state | {int(value)})
    if read == "count":
        return str(len(state))
    verb, value = read.split()
    if verb != "has":
        raise ValueError(read)
    return str(int(int(value) in state))


def make_pair(family: str, split: str, rng: random.Random) -> Pair:
    long_prefix = split == "prefix_length"
    large = split == "scale"
    if family == "arithmetic":
        length = rng.randint(12, 20) if long_prefix else rng.randint(4, 8)
        a, b = rng.randint(-6, 6), rng.randint(-6, 6)
        delta = rng.choice((-3, -2, -1, 1, 2, 3))
        steps = []
        for _ in range(length):
            op = rng.choice(("shift_a", "shift_b", "xfer", "swap", "flip"))
            k = 0 if op in ("swap", "flip") else rng.choice((-3, -2, -1, 1, 2, 3))
            steps.append([op, k])
        return Pair(family, {}, [[a, b], [a+delta, b-delta]], steps, split)
    if family == "graph":
        n = rng.choice((10, 12)) if large else rng.choice((6, 8))
        graph = {name: rng.sample(range(n), n) for name in ("L", "R")}
        length = rng.randint(12, 20) if long_prefix else rng.randint(4, 8)
        path = rng.choices(("L", "R"), k=length)
        ends = rng.sample(range(rng.randrange(2), n, 2), 2)
        starts = []
        for node in ends:
            for op in reversed(path):
                node = graph[op].index(node)
            starts.append(node)
        return Pair(family, graph, starts, path, split)
    if family != "rules":
        raise ValueError(family)
    # Two equally deep Horn chains and unreachable distractor rules. Vary rule
    # count independently of active chain depth in the two OOD experiments.
    n = 28
    depth = rng.randint(6, 8) if long_prefix else rng.randint(2, 4)
    rule_count = rng.randint(34, 40) if large else rng.randint(20, 24)
    names = rng.sample(range(n), n)
    width = depth + 1
    edges = [(names[j+i], names[j+i+1]) for j in (0, width) for i in range(depth)]
    rest = names[2*width:]
    distractors = [(a, b) for a in rest for b in rest if a != b]
    edges.extend(rng.sample(distractors, rule_count - len(edges)))
    world = {"n": n, "edges": [list(edge) for edge in sorted(edges)]}
    return Pair(family, world, [[names[0]], [names[width]]], [], split)


def composition_ood(family: str, suffix: Suffix) -> bool:
    verbs = [op.split()[0] for op in suffix]
    return HELD_OUT[family] in zip(verbs, verbs[1:])


def random_suffix(pair: Pair, rng: random.Random, *, length: int | None = None,
                  composition: bool | None = None) -> Suffix:
    held = pair.split == "comp" if composition is None else composition
    if length is None:
        lo, hi = (5, 8) if pair.split == "suffix_length" else (1, 4)
        length = rng.randint(max(lo, 2 if held else 1), hi)
    if length < 1 or (held and length < 2):
        raise ValueError("Invalid suffix length for the requested split")
    family = pair.family
    while True:
        if family == "arithmetic":
            ops = rng.choices(ARITH_OPS, k=length-1) + [rng.choice(ARITH_READS)]
        elif family == "graph":
            ops = rng.choices(("L", "R"), k=length-1) + [rng.choice(("node", "parity"))]
        else:
            n = pair.world["n"]
            ops = [f"add {rng.randrange(n)}" for _ in range(length-1)]
            ops.append(rng.choice(("count", f"has {rng.randrange(n)}")))
        if held:
            if family == "rules":
                ops[-1] = f"has {rng.randrange(pair.world['n'])}"
            else:
                ops[-2:] = HELD_OUT[family]
        suffix = tuple(ops)
        if composition_ood(family, suffix) == held:
            return suffix


def separating_read(pair: Pair) -> Suffix:
    if pair.family == "rules":
        facts = prefix_state(pair, 0) ^ prefix_state(pair, 1)
        return (f"has {min(facts)}",)
    return ("left",) if pair.family == "arithmetic" else ("node",)


def searches(pair: Pair, rng: random.Random, budget: int, max_length: int = 4) -> list[Suffix]:
    # Seed with a witness, then test additional programs within the same grammar.
    found = {separating_read(pair): None}
    for i in range(budget - 1):
        s = random_suffix(pair, rng, length=i % max_length + 1, composition=False)
        found[s] = None
    return list(found)


def partition(group: str) -> str:
    bucket = int(group[:8], 16) % 10
    return "train" if bucket < 8 else ("valid" if bucket == 8 else "test")


def generate_split(split: str, per_family: int, seed: int) -> Iterable[Pair]:
    if split not in SPLITS or per_family < 1:
        raise ValueError("Provide a known split and a positive pair count")
    families = FAMILIES[1:] if split == "scale" else FAMILIES
    for family in families:
        rng = random.Random(stable_id([seed, split, family]))
        seen = set()
        while len(seen) < per_family:
            pair = make_pair(family, split, rng)
            key = pair.group
            primary = split if split in ("train", "valid") else "test"
            if key in seen or partition(key) != primary:
                continue
            # Assign each test object to exactly one stress condition across
            # seeds and generation runs, independent of its parity/readout.
            if primary == "test":
                stress = [s for s in SPLITS[2:] if s != "scale" or family != "arithmetic"]
                if stress[int(key[8:16], 16) % len(stress)] != split:
                    continue
            seen.add(key)
            yield pair


class Oracle:
    """Count actual executor queries and CPU time, including negative candidates."""
    def __init__(self):
        self.calls = 0
        self.seconds = 0.0

    def labels(self, pair: Pair, suffixes: list[Suffix]) -> list[list[str]]:
        import time
        t = time.process_time()
        result = [[execute(pair, side, s) for s in suffixes] for side in (0, 1)]
        self.calls += 2 * len(suffixes)
        self.seconds += time.process_time() - t
        return result
