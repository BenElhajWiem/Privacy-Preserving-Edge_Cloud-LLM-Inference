from __future__ import annotations

import json
import time
from dataclasses import asdict

from gapdeleg import REPO_ROOT
from gapdeleg.backends import make_cloud, make_edge
from gapdeleg.baselines import make_system
from gapdeleg.data import load
from gapdeleg.metrics import cost as cost_m
from gapdeleg.metrics import leakage as leak_m
from gapdeleg.metrics.attacker import Attacker
from gapdeleg.metrics.attacker import aggregate as attack_aggregate
from gapdeleg.metrics.quality import PairwiseJudge, answer_exact_match, quality_summary
from gapdeleg.pipeline.context import Context, Trace
from gapdeleg.pipeline.egress import GuardedCloud, load_outgoing
from gapdeleg.pipeline.runlog import RunDir
from gapdeleg.privacy.detectors import Detector
from gapdeleg.privacy.guard import EdgeLeakCheck, EgressGuard
from gapdeleg.privacy.labeler import (CategorizedEdgeLabeler, DetectorLabeler, PatternLabeler, PersonalPhraseLabeler,
                                      PromptedEdgeLabeler, UnionLabeler)
from gapdeleg.privacy.policy import Policy


def build_labeler(names: list[str], detector: Detector, edge, policy: Policy | None = None):
    parts = []
    for n in names or []:
        if n == "detector":
            parts.append(DetectorLabeler(detector))
        elif n == "edge":
            parts.append(PromptedEdgeLabeler(edge))
        elif n == "edge_categorized":
            parts.append(CategorizedEdgeLabeler(edge, policy))
        elif n == "edge_phrases":
            parts.append(PersonalPhraseLabeler(edge))
        elif n == "patterns":
            parts.append(PatternLabeler(policy))
        else:
            raise ValueError(f"unknown labeler {n!r}")
    return UnionLabeler(*parts) if parts else None


def cloud_spend(cloud_model) -> float:
    from gapdeleg.build.batch import cost
    return cost(cloud_model.name, asdict(cloud_model.spent), batch=False) if cloud_model else 0.0


def run(cfg: dict, run_root=REPO_ROOT / "runs", log=print) -> RunDir:
    rd = RunDir(run_root, name=cfg.get("name", "run"))
    rd.write_config(cfg)
    policy = Policy.load(cfg["policy"]) if cfg.get("policy") else Policy.load()
    detector = Detector(policy)
    edge = make_edge(cfg["edge"])
    system = make_system(cfg["system"], **cfg.get("system_args", {}))

    gcfg = cfg.get("guard", {})
    leak_check = EdgeLeakCheck(edge, wording=gcfg.get("leak_check_wording", "yes_no")) if gcfg.get("llm_check") else None
    guard = EgressGuard(policy, detector=detector, log_path=rd.guard_log,
                        enabled=not getattr(system, "guard_disabled", False),
                        request_check=gcfg.get("request_check", False), llm_check=leak_check)
    cloud_cache = REPO_ROOT / cfg.get("cloud_cache", "runs/cache")
    cloud_model = make_cloud(cfg["cloud"], cache_dir=cloud_cache) if cfg.get("cloud") else None
    cloud = GuardedCloud(cloud_model, guard, rd.outgoing) if cloud_model else None
    budget = cfg.get("cloud_budget_usd")
    if getattr(system, "needs_cloud", False) and cloud is None:
        raise ValueError(f"system {cfg['system']} needs a cloud model")

    models = {"edge": edge.describe()}
    if cloud_model:
        models["cloud"] = cloud_model.describe()
    judge = None
    if cfg.get("judge"):
        judge_model = make_cloud(cfg["judge"], cache_dir=REPO_ROOT / "runs/cache")
        judge = PairwiseJudge(judge_model, guard)
        models["judge"] = judge_model.describe()
    attacker = None
    if cfg.get("attacker"):
        attack_model = make_cloud(cfg["attacker"], cache_dir=REPO_ROOT / "runs/cache")
        attacker = Attacker(attack_model, guard)
        models["attacker"] = attack_model.describe()
    rd.write_metadata(models)

    gen = cfg.get("gen", {})
    ctx = Context(edge=edge, cloud=cloud, labeler=build_labeler(cfg.get("labeler", []), detector, edge, policy),
                  policy=policy, edge_max_tokens=gen.get("edge_max_tokens", 1024),
                  cloud_max_tokens=gen.get("cloud_max_tokens", 1024))

    data_cfg = dict(cfg["data"])
    ids = data_cfg.pop("ids", None)
    examples = load(data_cfg.pop("dataset"), **data_cfg)
    if ids:
        keep = set(ids)
        examples = [ex for ex in examples if ex.id in keep]
    log(f"run {rd.id}: {cfg['system']} on {len(examples)} examples, edge={edge.name}, "
        f"cloud={cloud_model.name if cloud_model else None}")

    traces, costs, stopped = {}, [], None
    for ex in examples:
        if budget and cloud_spend(cloud_model) > budget:
            stopped = ex.id
            log(f"cloud budget of ${budget} reached: stopping before {ex.id}")
            break
        trace = Trace(ex.id)
        t = time.time()
        try:
            answer, error = system(ex, ctx, trace), None
        except NotImplementedError:
            raise
        except Exception as e:
            answer, error = "", f"{type(e).__name__}: {e}"
        c = cost_m.request_cost(trace, time.time() - t)
        costs.append(c)
        traces[ex.id] = trace
        rec = {"id": ex.id, "answer": answer, "error": error, "notes": trace.notes, "cost": c,
               "guard_denied": trace.guard_denied,
               "edge_steps": [{"step": s, "model": g.model, "latency_s": g.latency_s,
                               "in": g.usage.input_tokens, "out": g.usage.output_tokens,
                               "tok_per_s": g.meta.get("eval_tok_per_s")} for s, g in trace.edge]}
        if judge and ex.family == "A" and ex.reference and answer:
            try:
                rec["judge"] = asdict(judge.compare(ex.query, answer, ex.reference, request_id=ex.id))
            except Exception as e:
                rec["judge_error"] = f"{type(e).__name__}: {e}"[:300]
        if ex.family == "B" and ex.reference:
            rec["exact_match"] = answer_exact_match(answer, ex.reference, ex.meta.get("options"))
        rd.append_result(rec)
        log(f"  {ex.id}: cloud_calls={c['cloud_calls']} denials={c['guard_denials']} "
            f"edge_s={c['edge_latency_s']:.1f} wall_s={c['wall_s']:.1f}{' ERROR ' + error if error else ''}")

    examples = [ex for ex in examples if ex.id in traces]
    outgoing = load_outgoing(rd.outgoing)
    leaks = [leak_m.request_leakage(ex.id, ex.private_spans, outgoing.get(ex.id, []),
                                    policy.guard.get("fuzzy_threshold", 90), policy.guard.get("min_fuzzy_len", 4),
                                    query=ex.query)
             for ex in examples]
    rd.write_json("leakage.json", [lk.to_dict() for lk in leaks])
    attacks = []
    if attacker:
        attacks = [attacker.attack(ex.id, ex.private_spans, outgoing.get(ex.id, [])) for ex in examples]
        rd.write_json("attacker.json", [a.to_dict() for a in attacks])
    summary = {"run_id": rd.id, "system": cfg["system"], "n": len(examples),
               "leakage": leak_m.aggregate(leaks), "cost": cost_m.aggregate(costs)}
    if attacks:
        summary["attacker"] = attack_aggregate(attacks)
    if cloud_model:
        summary["cloud_spend_usd"] = round(cloud_spend(cloud_model), 4)
    if stopped:
        summary["stopped_at_budget"] = stopped
    recs = [json.loads(x) for x in rd.results.read_text().splitlines()]
    if quality := quality_summary(recs):
        summary["quality"] = quality
    if any(ex.family == "B" for ex in examples):
        summary["exact_match"] = sum(r.get("exact_match", False) for r in recs) / len(recs)
    rd.write_json("summary.json", summary)
    return rd
