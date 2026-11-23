"""Generate single-factor prompt pairs with disjoint composition splits."""

from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from pathlib import Path
import random

from .io import digest, read_rows, write_json, write_rows

OBJECTS = (
    "cup",
    "ball",
    "bottle",
    "book",
    "plate",
    "spoon",
    "mug",
    "bowl",
    "shoe",
    "bag",
    "hat",
    "sock",
    "candle",
    "vase",
    "apple",
    "pear",
    "chair",
    "stool",
    "pencil",
    "crayon",
)
LANDMARKS = (
    "box",
    "basket",
    "clock",
    "lamp",
    "bench",
    "shelf",
    "ladder",
    "mirror",
    "cabinet",
    "window",
    "stove",
    "door",
)
COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "pink", "brown")
RELATIONS = ("to the left of", "to the right of", "above", "below")
COUNTS = ("one", "two", "three", "four")
VALUES = dict(
    object=OBJECTS, landmark=LANDMARKS, color=COLORS, count=(1, 2, 3, 4), relation=RELATIONS
)
FACTORS = tuple(VALUES)
TARGETS = ("color", "count", "relation")


@dataclass(frozen=True)
class Scene:
    object: str
    landmark: str
    color: str
    count: int
    relation: str

    def __post_init__(self):
        for factor, choices in VALUES.items():
            if getattr(self, factor) not in choices:
                raise ValueError(f"Unknown {factor} in scene {self}")

    @property
    def text(self):
        noun = self.object if self.count == 1 else self.object + "s"
        return f"Exactly {COUNTS[self.count - 1]} {self.color} {noun} {self.relation} one {self.landmark}."

    def atom_question(self, factor):
        plural = self.object + "s"
        if factor == "object":
            return f"Is there a {self.object} in this image?"
        if factor == "landmark":
            return f"Is there exactly one {self.landmark} in this image?"
        if factor == "count":
            noun = self.object if self.count == 1 else plural
            verb = "Is" if self.count == 1 else "Are"
            return f"{verb} there exactly {COUNTS[self.count - 1]} {noun}?"
        if factor == "color":
            return f"Are all the {plural} {self.color}?"
        if factor == "relation":
            return f"Are all the {plural} {self.relation} the {self.landmark}?"
        raise KeyError(factor)

    def neighbor(self, factor):
        # Adjacent alternatives stay within the same lexical block, including projector calibration.
        choices = VALUES[factor]
        alt = choices[choices.index(getattr(self, factor)) ^ 1]
        return replace(self, **{factor: alt})


@dataclass(frozen=True)
class Pair:
    left: Scene
    right: Scene
    factor: str
    split: str

    def __post_init__(self):
        changed = [k for k in FACTORS if getattr(self.left, k) != getattr(self.right, k)]
        if self.factor not in TARGETS or changed != [self.factor]:
            raise ValueError(f"Expected exactly one target intervention, got {changed}")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {self.split}")

    @property
    def key(self):
        return digest(asdict(self))[:20]

    @property
    def scenes(self):
        return (self.left, self.right)


def composition_audit(partitions):
    """Check observed triples and all single-factor calibration interventions."""
    observed, extended = {}, {}
    for split, pairs in partitions.items():
        triples = {(s.object, s.color, s.relation) for p in pairs for s in p.scenes}
        neighbors = set(triples)
        # Count and landmark interventions leave the tested triple unchanged.
        for obj, col, rel in triples:
            neighbors.add((OBJECTS[OBJECTS.index(obj) ^ 1], col, rel))
            neighbors.add((obj, COLORS[COLORS.index(col) ^ 1], rel))
            neighbors.add((obj, col, RELATIONS[RELATIONS.index(rel) ^ 1]))
        observed[split], extended[split] = triples, neighbors

    def overlaps(sets):
        return {f"{a}/{b}": len(sets[a] & sets[b]) for a, b in combinations(sets, 2)}

    direct, with_neighbors = overlaps(observed), overlaps(extended)
    if any(with_neighbors.values()):
        raise ValueError(
            f"Object/color/relation split leakage: observed={direct}, including interventions="
            f"{with_neighbors}. Generate a new dataset in a fresh directory for this study."
        )
    return dict(
        triples={s: len(v) for s, v in observed.items()},
        overlap=direct,
        overlap_with_interventions=with_neighbors,
    )


def make_dataset(directory, sizes=(8000, 1000, 1000), seed=1203):
    rng = random.Random(seed)
    # Group adjacent alternatives in all three held-out coordinates. Rotate the
    # held-out object blocks so every atomic concept is represented in training.
    objects = list(range(len(OBJECTS) // 2))
    contexts = list(product(range(len(COLORS) // 2), range(len(RELATIONS) // 2)))
    rng.shuffle(objects)
    rng.shuffle(contexts)
    parts = dict(train=[], val=[], test=[])
    for i, (cb, rb) in enumerate(contexts):
        for j, ob in enumerate(objects):
            split = "val" if j == i else "test" if j == (i + 1) % len(objects) else "train"
            parts[split].append((ob, cb, rb))
    directory = Path(directory)
    partitions = {}
    for (split, subset), requested in zip(parts.items(), sizes, strict=True):
        pool = []
        for ob, cb, rb in subset:
            for obj, lm, col, ct, rel in product(
                range(2 * ob, 2 * ob + 2),
                range(len(LANDMARKS)),
                range(2 * cb, 2 * cb + 2),
                range(len(COUNTS)),
                range(2 * rb, 2 * rb + 2),
            ):
                # A perfect matching within every 2x2x2 color/count/relation cube.
                factor_idx = (obj + lm + col // 2 + ct // 2 + rel // 2) % 3
                if (col, ct, rel)[factor_idx] % 2:
                    continue
                left = Scene(OBJECTS[obj], LANDMARKS[lm], COLORS[col], ct + 1, RELATIONS[rel])
                factor = TARGETS[factor_idx]
                right = left.neighbor(factor)
                if rng.random() < 0.5:
                    left, right = right, left
                pool.append(Pair(left, right, factor, split))
        if not 0 < requested <= len(pool):
            raise ValueError(f"{split}: requested {requested}, capacity {len(pool)} disjoint pairs")
        partitions[split] = rng.sample(pool, requested)
    audit = composition_audit(partitions)
    split_counts = {split: len(pairs) for split, pairs in partitions.items()}
    for split, pairs in partitions.items():
        write_rows(directory / f"{split}.jsonl", (asdict(pair) for pair in pairs))
    write_json(
        directory / "dataset.json",
        dict(
            seed=seed, pair_counts=split_counts, generator="object-color-relation-v2", audit=audit
        ),
    )
    return split_counts


def load_pairs(path, allowed):
    rows = read_rows(path)
    pairs = [Pair(Scene(**r["left"]), Scene(**r["right"]), r["factor"], r["split"]) for r in rows]
    if not pairs or any(p.split not in allowed for p in pairs):
        raise ValueError(f"Expected nonempty data with splits in {allowed}: {path}")
    if len({p.key for p in pairs}) != len(pairs):
        raise ValueError("Duplicate pair in input data")
    return pairs
