"""Run the controlled comparison, validation selection, ablations and transfer experiment."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .cli import evaluate_run, evaluation_options
from .evaluation import INTERVENTIONS
from .experiment import run_lock, train
from .public_data import prepare, prepared
from .settings import Config, LATENT_RL, LOOP, METHODS, PROCESS, TEXT, build_data, controlled_cot_tokens, digest, versions, write_json
from .tasks import SPLITS

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (31, 41, 51)
UNIFIED = tuple(m for m in METHODS if m not in LOOP + TEXT)
RANKABLE = tuple(m for m in UNIFIED if m != "ct-rl")
TESTS = SPLITS[2:]


def read(path):
    return json.loads(path.read_text())


def history(run):
    return read(run/"history.json")


def total(run):
    return sum(x["operator_flops"] for x in history(run))


def recipe_copy(cfg, **changes):
    obj = asdict(cfg)
    for k, v in changes.items():
        if k.startswith("model."):
            obj["model"][k[6:]] = v
        else:
            obj[k] = v
    return Config.from_dict(obj)


def rank_scores(by_method):
    keys = ("ct_acc", "pair_both", "all_correct")
    means = {m:{k:statistics.mean(r[k] for r in rows) for k in keys} for m, rows in by_method.items()}
    ranked = sorted(means, key=lambda m:tuple(-means[m][k] for k in keys)+(m,))
    return ranked, means


class Study:
    def __init__(self, args):
        self.out, self.smoke = args.out, args.profile == "smoke"
        self.profile = args.profile
        self.base = Config.load(ROOT/"configs"/("paper-smoke.json" if self.smoke else "qwen1.5b.json"))
        self.large = (recipe_copy(self.base, **{"model.hidden_size":48,"model.layers":2})
                      if self.smoke else Config.load(ROOT/"configs/qwen3b.json"))
        self.data = self.out/"data" if self.smoke else ROOT/"data/paper"
        self.public = args.public_data or (ROOT/"data/public-smoke" if self.smoke else self.out/"public-data")
        self.cache = args.cache or self.out/"upstream-cache"
        self.seeds = SEEDS
        self.secondary_seeds = SEEDS[:1] if self.smoke else SEEDS
        self.limit = 1 if self.smoke else None
        self.bootstrap = 20 if self.smoke else 1000

    def setup(self):
        build_data(self.data, 20 if self.smoke else 10000, 2 if self.smoke else 500, 120,
                   valid_pairs=2 if self.smoke else 500)
        if (self.public/"manifest.json").exists():
            prepared(self.public, self.profile)
        else:
            prepare(self.public, self.cache, self.profile)
        plan = {"profile":self.profile, "base":asdict(self.base), "large":asdict(self.large),
                "seeds":self.seeds, "secondary_seeds":self.secondary_seeds,
                "data_sha256":digest(self.data/"manifest.json"),
                "public_sha256":digest(self.public/"manifest.json"), "code":versions()["source_sha256"]}
        file = self.out/"protocol.json"
        if file.exists() and read(file) != json.loads(json.dumps(plan)):
            raise ValueError("The recorded study differs in data, seed, configuration, or source. Use a new run.")
        write_json(file, plan)

    def cfg(self, base, method, seed):
        changes = {"method":method, "seed":seed}
        if method == "ltc":
            changes["k"] = 2
        if method == "cot":
            changes["model.max_answer_tokens"] = controlled_cot_tokens(base.model.backbone)
        return recipe_copy(base, **changes)

    def path(self, stage, tag, seed):
        return self.out/stage/tag/str(seed)

    def fit(self, run, cfg, cutoff=None):
        ckpt = run/"checkpoint.pt"
        if ckpt.exists():
            old_cfg = Config.from_dict(read(run/"config.json"))
            cfg.model.revision = cfg.model.revision or old_cfg.model.revision
            if cutoff is not None:
                cfg.rounds = max(cfg.rounds, old_cfg.rounds)
                old_cfg.rounds = cfg.rounds
            if old_cfg != cfg:
                raise ValueError(f"Different recipe under {run}")
            rows = history(run) if (run/"history.json").exists() else []
            if rows and (rows[-1].get("round", 0) >= cfg.rounds or (cutoff and total(run) >= cutoff)):
                return ckpt
        return train(cfg, self.data, run, resume=ckpt.exists(), count_flops=True, stop_flops=cutoff)

    def evaluate(self, run, name, splits, counts=(4,), modes=("normal",), measure=False, limit=None):
        path = run/(name+".json")
        ckpt = run/"checkpoint.pt"
        checksum = digest(ckpt)
        limit = self.limit if limit is None else limit
        options = evaluation_options(self.base.device, splits, counts, modes, limit, self.bootstrap, measure, measure)
        if path.exists():
            prior = read(path)
            if (prior.get("checkpoint_sha256") == checksum and prior.get("options") == options
                    and prior.get("dataset_sha256") == digest(self.data/"manifest.json")):
                return prior["reports"]
        return evaluate_run(ckpt, self.data, self.base.device, splits, counts, modes,
                            limit, self.bootstrap, measure, measure, path)["reports"]

    def equal_steps(self):
        rows = defaultdict(list)
        for method in METHODS:
            for seed in self.seeds:
                run = self.path("primary", method, seed)
                self.fit(run, self.cfg(self.base, method, seed))
                val = self.evaluate(run, "validation", ["valid"])
                if method in RANKABLE:
                    rows[method].append(val[0]["macro"])
        ranked, means = rank_scores(rows)
        write_json(self.out/"equal-step-validation.json", {"ranking":ranked,"scores":means})

    def selection(self):
        rows = defaultdict(list)
        for method in RANKABLE:
            for seed in self.seeds:
                result = self.evaluate(self.path("matched",method,seed),"validation",["valid"])
                rows[method].append(result[0]["macro"])
        ranked, means = rank_scores(rows)
        decision = {"data":"valid", "training_protocol":"matched operator FLOPs",
                    "rank_keys":["mean CT-Acc","mean Pair-Both","mean All-Correct","name"],
                    "seeds":self.seeds, "ranking":ranked, "scores":means, "promote":ranked[:3]}
        write_json(self.out/"selection.json", decision)
        return decision

    def primary_tests(self, decision):
        for method in METHODS:
            for seed in self.seeds:
                self.evaluate(self.path("primary",method,seed), "test", TESTS)
        for method in UNIFIED + TEXT + LOOP:
            for seed in self.seeds:
                self.evaluate(self.path("matched",method,seed), "test", TESTS)
        promoted = {}
        for method in ("ct-rl", *decision["promote"]):
            for seed in self.seeds:
                run = self.path("scale-equal-steps",method,seed)
                cfg = self.cfg(self.large,method,seed)
                self.fit(run,cfg)
                promoted[method,seed] = (cfg,run)
        write_json(self.out/"scale-budget.json",self.match_training("scale-confirmation",promoted))
        for method,seed in promoted:
            self.evaluate(self.path("scale-confirmation",method,seed),"test",TESTS)

    def diagnostic(self, decision):
        for seed in self.secondary_seeds:
            self.evaluate(self.path("matched","ct-rl",seed), "interventions", ("id",),
                          modes=INTERVENTIONS)
            for method in UNIFIED + TEXT + LOOP:
                self.evaluate(self.path("matched",method,seed), "reuse", ("id",),
                              counts=(1,4,8,16), measure=True)

    def ablations(self):
        rows = []
        cfg = self.base
        variants = {}
        for m in (1,2,4,8):
            variants[f"M{m}-equal-steps"] = {"m":m}
            variants[f"M{m}-equal-reads"] = {"m":m,"steps_per_round":cfg.steps_per_round*cfg.m//m}
        for field, values in (("model.latent_steps",(2,4,8,16)), ("model.action_dim",(32,64,128)),
                              ("rho",(0.,.25,.5,.75)), ("k",(2,4,8))):
            for value in values:
                variants[f"{field}-{value}"] = {field:value}
        variants["grpo"] = {"algorithm":"grpo"}
        variants["beta-zero"] = {"beta":0.}
        variants["no-mining"] = {"rho":0., "mine_pairs":0}
        for seed in self.secondary_seeds:
            executed = {}
            for tag, changes in variants.items():
                config = recipe_copy(cfg, seed=seed, **changes)
                signature = json.dumps(asdict(config), sort_keys=True)
                if signature not in executed:
                    run = (self.path("primary","ct-rl",seed) if config == self.cfg(cfg,"ct-rl",seed)
                           else self.path("ablations",tag,seed))
                    self.fit(run, config)
                    self.evaluate(run, "test", TESTS)
                    executed[signature] = run
                run = executed[signature]
                updates = config.rounds*config.steps_per_round
                rows.append({"axis":tag,"seed":seed,"run":str(run.relative_to(self.out)),
                             "policy_updates":updates,"candidate_suffix_reads":updates*2*config.pairs_per_batch*config.k*config.m,
                             "reader_labels":updates*2*config.pairs_per_batch*config.m,
                             "flops_including_warm_and_search":total(run)})
        write_json(self.out/"ablation-index.json", rows)
        capacity = {}
        for row in rows:
            tag, seed = row["axis"], row["seed"]
            if tag.startswith(("model.", "rho-")) or tag == "M4-equal-steps":
                original = self.out/row["run"]
                capacity[tag,seed] = (Config.from_dict(read(original/"config.json")),original)
        write_json(self.out/"capacity-budget.json", self.match_training("capacity-matched",capacity))
        for tag,seed in capacity:
            self.evaluate(self.path("capacity-matched",tag,seed),"test",TESTS)

    def match_training(self, stage, recipes):
        # A common target for every seed and arm. Calibration uses training work only.
        costs = [history(source) for _,source in recipes.values()]
        target = max(max(sum(r["operator_flops"] for r in h),
                         7*max(r["operator_flops"] for r in h[1:])) for h in costs)
        for attempt in range(6):
            actual = {}
            for (tag,seed),(base,source) in recipes.items():
                unit = min(row["operator_flops"] for row in history(source)[1:])
                cap = max(base.rounds, math.ceil(2**(attempt+1)*target/max(unit,1)))
                cfg = recipe_copy(base, rounds=cap)
                run = self.path(stage,tag,seed)
                self.fit(run, cfg, target*.93)
                actual[tag,seed] = total(run)
            if all(abs(c/target-1) <= .10 for c in actual.values()):
                return [{"method":tag,"seed":seed,"target":target,"actual":c,
                         "relative_error":c/target-1,"within_tolerance":True,
                         "rounds":history(self.path(stage,tag,seed))[-1].get("round",0)}
                        for (tag,seed),c in actual.items()]
            if max(actual.values()) > target*1.10:
                target = max(target*1.2, max(actual.values()))
        raise RuntimeError(f"Training compute budget failed to converge for {stage}")

    def budget(self):
        recipes = {}
        for seed in self.seeds:
            methods = [(m,m,False) for m in UNIFIED+TEXT+LOOP]
            if seed in self.secondary_seeds:
                methods += [(m+"-native",m,True) for m in PROCESS+LATENT_RL]
            for tag, m, native in methods:
                cfg = recipe_copy(self.cfg(self.base,m,seed), native_pretrain=native)
                source = self.path("calibrate-native" if native else "primary",tag,seed)
                self.fit(source, cfg)
                recipes[tag,seed] = (cfg,source)
        write_json(self.out/"budget.json", self.match_training("matched",recipes))

    def run(self):
        self.setup()
        if self.smoke and not (self.out/"core-demo"/"evaluation.json").exists():
            from .cli import main as cli
            cli(["demo","--out",str(self.out/"core-demo")])
        self.equal_steps()
        self.budget()
        decision = self.selection()
        self.primary_tests(decision)
        self.diagnostic(decision)
        from .inference_budget import run_tiers
        run_tiers(self, UNIFIED + TEXT + LOOP)
        self.ablations()
        from .paper_transfer import transfer, report
        transfer(self)
        report(self)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("smoke","paper"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--public-data", type=Path)
    p.add_argument("--cache", type=Path)
    args = p.parse_args()
    args.out = args.out.resolve()
    with run_lock(args.out):
        Study(args).run()


if __name__ == "__main__":
    main()
