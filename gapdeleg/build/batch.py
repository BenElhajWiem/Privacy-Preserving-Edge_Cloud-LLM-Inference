from __future__ import annotations

import json
import os
import time
from pathlib import Path

from gapdeleg import REPO_ROOT

STORE = REPO_ROOT / "runs/batches"
LEDGER = REPO_ROOT / "runs/cost_ledger.jsonl"

PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.10, 0.50),
          "gpt-5.5": (5.0, 30.0)}


def client():
    import anthropic
    key = os.environ.get("GAPDELEG_ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("GAPDELEG_ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["custom_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def cost(model: str, usage: dict, batch: bool = True) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    usd = (usage.get("input_tokens", 0) * pin + usage.get("output_tokens", 0) * pout) / 1e6
    return usd * (0.5 if batch else 1.0)


def log_cost(job: str, model: str, usage: dict, usd: float, n: int, batch: bool) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "job": job, "model": model, "n": n,
                            "batch": batch, **usage, "usd": round(usd, 4)}) + "\n")


def run_batch(name: str, requests: dict[str, dict], poll_s: float = 20, log=print) -> dict[str, dict]:
    STORE.mkdir(parents=True, exist_ok=True)
    path = STORE / f"{name}.jsonl"
    done = _load(path)
    todo = {k: v for k, v in requests.items() if k not in done or not done[k]["ok"]}
    if todo:
        c = client()
        model = next(iter(todo.values()))["model"]
        b = c.messages.batches.create(requests=[{"custom_id": k, "params": p} for k, p in todo.items()])
        log(f"[{name}] submitted batch {b.id} with {len(todo)} requests")
        while True:
            b = c.messages.batches.retrieve(b.id)
            if b.processing_status == "ended":
                break
            log(f"[{name}] {b.request_counts.succeeded} done, {b.request_counts.processing} processing")
            time.sleep(poll_s)
        usage = {"input_tokens": 0, "output_tokens": 0}
        with open(path, "a") as f:
            for r in c.messages.batches.results(b.id):
                rec = {"custom_id": r.custom_id, "ok": r.result.type == "succeeded"}
                if rec["ok"]:
                    m = r.result.message
                    rec["text"] = "".join(x.text for x in m.content if x.type == "text")
                    rec["usage"] = {"input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens}
                    rec["stop_reason"] = m.stop_reason
                    for k in usage:
                        usage[k] += rec["usage"][k]
                else:
                    rec["error"] = str(getattr(r.result, "error", r.result.type))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[r.custom_id] = rec
        usd = cost(model, usage)
        log_cost(name, model, usage, usd, len(todo), batch=True)
        log(f"[{name}] ended: {sum(done[k]['ok'] for k in todo if k in done)}/{len(todo)} ok, "
            f"{usage['input_tokens']} in / {usage['output_tokens']} out tokens, ${usd:.2f}")
    return {k: done[k] for k in requests if k in done}


def json_params(model: str, prompt: str, schema: dict, system: str | None = None, max_tokens: int = 4096,
                effort: str = "low") -> dict:
    p = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}],
         "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}}}
    if system:
        p["system"] = system
    return p


def parse_json(result: dict | None) -> dict | None:
    if not result or not result.get("ok"):
        return None
    try:
        return json.loads(result["text"])
    except (json.JSONDecodeError, KeyError):
        return None

