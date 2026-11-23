"""Objectives for process-supervised, stochastic-latent and loop-depth comparisons."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .comparisons import empty_state, normalized, stop_distribution, text_prompt
from .curriculum import answer_integer
from .learning import clipped, leave_one_out, take_step
from .readout import completion
from .settings import LOOP, PROCESS, TEXT


def labels_for(model, answers):
    rows = [torch.tensor(model.tokens(a)+[model.eos], device=model.device) for a in answers]
    return pad_sequence(rows, batch_first=True, padding_value=-100)


def teacher_force(model, context, labels, route="reader", layers=False, loops=False):
    x, mask = context
    first = x.shape[1]-1
    prev = labels[:, :-1]
    x = torch.cat((x, model.embed(prev.clamp_min(0))), 1)
    mask = torch.cat((mask, (prev != -100).long()), 1)
    if loops:
        states, _ = model.loop_states(x, mask, route)
    elif layers:
        states, _ = model._hidden(x, mask, route, all_layers=True)
    else:
        h, _ = model._hidden(x, mask, route)
        states = [h]
    return [s[:, first:] for s in states]


def token_scores(logits, labels, temperature=1.):
    logp = (logits.float()/temperature).log_softmax(-1)
    return logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1).masked_fill(labels == -100, 0.)


def prefix_lm_loss(model, prefixes, raw, traces):
    prompt = [torch.cat((model.embed(model.tokens(p+"\n", bos=True)), latent), 0)
              for p, latent in zip(prefixes, raw)]
    y = labels_for(model, traces)
    h = teacher_force(model, model.pack(prompt), y, route="policy")[-1]
    return -token_scores(model.lm.lm_head(h), y).sum(-1).mean()


def combine_prefixes(prefixes, suffixes, native):
    return list(zip(prefixes, suffixes)) if native else suffixes


def process_loss(model, prefixes, suffixes, answers, traces, fraction, native):
    cfg = model.recipe
    if cfg.method == "coconut":
        count = min(cfg.model.latent_steps, int(fraction*(cfg.model.latent_steps+1)))
        state = model.thoughts(prefixes, stochastic=False, steps=count)
        visible = ["\n".join(t[math.ceil(len(t)*count/cfg.model.latent_steps):]) for t in traces]
        loss = model.answer_loss(state.interface(native), combine_prefixes(prefixes, suffixes, native), answers)
        if any(visible):
            loss = loss + prefix_lm_loss(model, prefixes, state.raw, visible)
        return loss
    state = model.thoughts(prefixes, stochastic=False)
    rationale = ["\n".join(t) for t in traces]
    teacher_loss = prefix_lm_loss(model, prefixes, state.raw[:, :0], rationale)
    full = [(p+"\n"+r, s) for p, r, s in zip(prefixes, rationale, suffixes)]
    y = labels_for(model, answers)
    teacher = teacher_force(model, model.reader_inputs(state.z[:, :0], full), y, layers=True)
    student = teacher_force(model, model.reader_inputs(state.interface(native),
                            combine_prefixes(prefixes, suffixes, native)), y, layers=True)
    ce = sum(-token_scores(model.logits(v[-1]), y).sum(-1).mean() for v in (teacher, student))
    hidden_loss = 0.
    for t, s in zip(teacher, student):
        target = t[:, 0].detach().float()
        scale = target.std(unbiased=False).clamp_min(1e-4)
        hidden_loss = hidden_loss + F.smooth_l1_loss(s[:, 0].float(), target)/scale
    return ce + teacher_loss + cfg.distill_weight * hidden_loss/len(teacher)


def special_supervision(model, cfg, prefixes, tests, traces, opt, parameters, fraction, native=False):
    """One optimizer step. Per-prefix microbatches keep trace lengths independent."""
    opt.zero_grad(set_to_none=True)
    n = sum(map(len, tests.answers))
    total = 0.
    for p, suffixes, answers, trace in zip(prefixes, tests.suffixes, tests.answers, traces):
        for i in range(0, len(suffixes), cfg.forward_batch):
            ss, yy = suffixes[i:i+cfg.forward_batch], answers[i:i+cfg.forward_batch]
            pp = [p]*len(ss)
            if cfg.method in TEXT:
                targets = [completion(a, cfg.method, "\n".join(trace)) for a in yy]
                loss = model.answer_loss(empty_state(model, len(ss)),
                                         [text_prompt(p, s, cfg.method) for s in ss], targets)
            elif cfg.method in PROCESS:
                loss = process_loss(model, pp, ss, yy, [trace]*len(ss), fraction, native)
            else:
                state = model.thoughts(pp, stochastic=True, full_gate=True)
                prompts = combine_prefixes(pp, ss, native)
                loss = model.answer_loss(state.interface(native), prompts, yy)
                if cfg.method == "slpo":
                    with torch.no_grad():
                        wins = []
                        all_z = state.raw if native else state.z
                        for j in range(1, cfg.model.latent_steps+1):
                            keep = torch.arange(cfg.model.latent_steps, device=model.device) < j
                            guesses = model.decode(all_z*keep[None, :, None], prompts)
                            wins.append([v is not None and v.strip().casefold() == y.strip().casefold()
                                         for v, y in zip(guesses, yy)])
                        valid = torch.tensor(wins, device=model.device).T
                    mass = stop_distribution(state.gates, min(cfg.model.latent_steps, cfg.min_thought_steps))
                    good = (mass*valid).sum(-1)
                    reachable = good > 0
                    if reachable.any():
                        loss = loss - cfg.gate_weight*good[reachable].log().mean()
            weighted = loss * (len(yy)/n)
            weighted.backward()
            total += weighted.detach().item()
    take_step(parameters, opt, cfg.grad_clip)
    return total


@torch.no_grad()
def sample_answers(model, z, prompts, temperature):
    x, mask = model.reader_inputs(z, prompts)
    cache = None
    done = torch.zeros(len(z), device=model.device, dtype=torch.bool)
    ids = []
    for _ in range(model.cfg.max_answer_tokens):
        h, cache = model._hidden(x, mask, "reader", cache, use_cache=True)
        logits = model.logits(h[:, -1])/temperature
        tok = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
        ids.append(tok.masked_fill(done, -100))
        done = done | model.ends_answer(tok)
        if done.all():
            break
        x = model.embed(tok)[:, None]
        mask = F.pad(mask, (0, 1), value=1)
    labels = torch.stack(ids, 1)
    pred = [model.tokenizer.decode([x for x in row if x >= 0 and x not in model.stop_tokens],
                                  skip_special_tokens=False).strip() if finished else None
            for row, finished in zip(labels.tolist(), done.tolist())]
    return labels, pred, done


def latent_terms(model, prefixes, state):
    cfg = model.recipe
    if cfg.method == "slpo":
        with model.mc_dropout(True):
            samples = torch.stack([model.history_states(prefixes, state).float() for _ in range(cfg.mc_draws)])
        h = state.raw
        mu = samples.mean(0)
        var = (samples-mu).square().mean(dim=(0, 3))+cfg.variance_floor
        sq = (state.raw.detach().float()-mu).square().sum(-1)
        logp = -0.5*(sq/var + h.shape[-1]*(math.log(2*math.pi)+var.log()))
        mask = torch.arange(h.shape[1], device=h.device)[None, :] < state.length[:, None]
        # For a realized prefix ending at t, the gate sees the realized state at t.
        gates = model.gate(state.raw.detach().float()).squeeze(-1)
        mass = stop_distribution(gates, min(cfg.min_thought_steps, h.shape[1]))
        stop = mass.gather(1, state.length[:, None]-1).squeeze(1).clamp_min(1e-12).log()
        return logp*mask, stop
    h = model.history_states(prefixes, state)
    words = model.policy_words(h).log_softmax(-1)
    current = words.gather(-1, state.ids)
    if cfg.method == "latent-grpo":
        diff = state.perturbed.detach()-current
        margin = torch.where(diff < 0, 2*diff.detach()-diff, diff)
        return (-margin-(-margin).exp()).sum(-1), None
    # Concrete log density on the retained simplex. Constant terms independent
    # of the policy cancel in REINFORCE and are omitted.
    log_w = state.weights.detach().clamp_min(1e-12).log()
    v = current-cfg.thought_temperature*log_w
    surrogate = (v-v.logsumexp(-1, keepdim=True)).sum(-1)
    clean, ids = words.topk(state.ids.shape[-1], dim=-1)
    match = (clean.softmax(-1).unsqueeze(-1)*model.embed(ids)).sum(-2)
    distance = (match-state.raw.detach()).square().mean(dim=(-1, -2))
    return surrogate, distance


def masked_advantage(reward, valid):
    weights = valid.float()
    count = weights.sum().clamp_min(1.)
    mean = (reward*weights).sum()/count
    std = (((reward-mean).square()*weights).sum()/count).sqrt()
    return (reward-mean)/(std+1e-6)*weights


def one_group(model, cfg, prefix, suffix, gold, native, grader):
    """Sample K distinct thoughts and A independent answers given each fixed thought."""
    k = cfg.k
    a = cfg.ltc_answers if cfg.method == "ltc" else 1
    pp = [prefix]*k
    with torch.no_grad():
        state = model.thoughts(pp, stochastic=True, full_gate=True)
        z = state.interface(native).repeat_interleave(a, 0)
        prompts = combine_prefixes([prefix]*(k*a), [suffix]*(k*a), native)
        y, predictions, complete = sample_answers(model, z, prompts, cfg.answer_temperature)
        reward = torch.tensor([float(grader(p, gold)) for p in predictions], device=z.device)
    mask = y.ne(-100)
    def scores():
        h = teacher_force(model, model.reader_inputs(z, prompts), y)[-1]
        answer = token_scores(model.logits(h), y, cfg.answer_temperature)
        thought, aux = latent_terms(model, pp, state)
        return thought, answer, aux
    def avg_answer(scores):
        return scores.sum(-1)/mask.sum(-1).clamp_min(1)
    if cfg.method == "latent-grpo":
        with torch.no_grad():
            old_thought, old_answer, _ = scores()
            advantage = masked_advantage(reward, complete)
            keep = complete[:, None].expand_as(old_thought).clone()
            correct = (reward > 0) & complete
            if correct.any():
                quality = (old_thought.sum(-1)+old_answer.sum(-1))/(old_thought.shape[1]+mask.sum(-1))
                best = quality.masked_fill(~correct, -torch.inf).argmax()
                keep[correct, 0] = False
                keep[best, 0] = True
        def loss():
            thought, answer, _ = scores()
            lt = clipped(thought, old_thought, advantage[:, None], cfg.clip_epsilon)
            la = clipped(answer, old_answer, advantage[:, None], cfg.clip_epsilon)
            token_mask = mask & complete[:, None]
            trajectory = (lt*keep).sum(-1)+(la*token_mask).sum(-1)
            length = old_thought.shape[1]+mask.sum(-1)
            return -(trajectory/length).mean()
    elif cfg.method == "ltc":
        thought_adv = normalized(reward.reshape(k, a).mean(-1))
        advantage = normalized(reward)
        def loss():
            thought, answer, distance = scores()
            return (-(thought.mean(-1)*thought_adv).mean()-(avg_answer(answer)*advantage).mean()
                    +cfg.matching_weight*(thought_adv.softmax(0)*distance).sum())
    else:
        advantage = leave_one_out(reward)
        def loss():
            thought, answer, stop = scores()
            return -((thought.sum(-1)+answer.sum(-1)+stop)*advantage).mean()
    return loss, {"task_reward": reward.mean().item(), "adv_abs": advantage.abs().mean().item(),
                  "complete_fraction": complete.float().mean().item(), "sampled_answers": k*a,
                  "mean_thought_steps": state.length.float().mean().item() if state.length is not None
                  else cfg.model.latent_steps}


def loop_group(model, cfg, prefix, suffix, gold, grader):
    pp = [prefix]*cfg.k
    with torch.no_grad():
        u, z = model.sample(pp)
        prompts = [suffix]*cfg.k
        y, preds, _ = sample_answers(model, z, prompts, cfg.answer_temperature)
        reward = torch.tensor([float(grader(p, gold)) for p in preds], device=z.device)
        adv = normalized(reward)
    ctx = model.reader_inputs(z, prompts)
    mask = y.ne(-100)
    length = mask.sum(-1).clamp_min(1)
    with torch.no_grad():
        ref_h = teacher_force(model, ctx, y, route="loop_reference_reader")[-1]
        ref_log = (model.ref_logits(ref_h)/cfg.answer_temperature).log_softmax(-1)
    def loss():
        states = teacher_force(model, ctx, y, loops=True)
        if cfg.method == "loop-grpo":
            states = states[-1:]
        logits = [model.logits(h) for h in states]
        per_loop = [token_scores(v, y, cfg.answer_temperature).sum(-1)/length for v in logits]
        final_log = (logits[-1]/cfg.answer_temperature).log_softmax(-1)
        kl = (final_log.exp()*(final_log-ref_log)).sum(-1)
        return -(torch.stack(per_loop).mean(0)*adv).mean()+cfg.beta*((kl*mask).sum(-1)/length).mean()
    return loss, {"task_reward": reward.mean().item(), "adv_abs": adv.abs().mean().item(),
                  "sampled_answers": cfg.k, "credited_loops": model.depth if cfg.method == "rltt" else 1}


def comparison_pg(model, cfg, prefixes, tests, optimizer, parameters, native=False, grader=None):
    grader = grader or (lambda p, y: answer_integer(p) == y)
    groups, logs = [], []
    for p, ss, yy in zip(prefixes, tests.suffixes, tests.answers):
        for s, y in zip(ss, yy):
            fn, row = (loop_group(model, cfg, p, s, y, grader) if cfg.method in LOOP else
                       one_group(model, cfg, p, s, y, native, grader))
            groups.append(fn)
            logs.append(row)
    epochs = cfg.grpo_epochs if cfg.method == "latent-grpo" else 1
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for fn in groups:
            term = fn()/len(groups)
            term.backward()
            total += term.detach().item()
        norm = take_step(parameters, optimizer, cfg.grad_clip)
    out = {key: sum(r[key] for r in logs)/len(logs) for key in logs[0]}
    out["sampled_answers"] = sum(r["sampled_answers"] for r in logs)
    return dict(out, policy_loss=total, policy_grad_norm=norm)
