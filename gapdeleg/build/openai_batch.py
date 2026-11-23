from __future__ import annotations

import json
import os
import tempfile
import time

from gapdeleg.build.batch import STORE, _load, cost, log_cost


def client():
    from openai import OpenAI
    key = os.environ.get("GAPDELEG_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("GAPDELEG_OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def chat_json_params(model: str, system: str, prompt: str, schema: dict, name: str = "result",
                     reasoning_effort: str = "low", max_completion_tokens: int = 8000) -> dict:
    return {"model": model, "reasoning_effort": reasoning_effort, "max_completion_tokens": max_completion_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}}


def run_openai_batch(name: str, requests: dict[str, dict], poll_s: float = 30, log=print) -> dict[str, dict]:
    STORE.mkdir(parents=True, exist_ok=True)
    path = STORE / f"{name}.jsonl"
    done = _load(path)
    todo = {k: v for k, v in requests.items() if k not in done or not done[k]["ok"]}
    if todo:
        c = client()
        model = next(iter(todo.values()))["model"]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for k, body in todo.items():
                f.write(json.dumps({"custom_id": k, "method": "POST", "url": "/v1/chat/completions",
                                    "body": body}) + "\n")
            tmp = f.name
        with open(tmp, "rb") as fh:
            up = c.files.create(file=fh, purpose="batch")
        os.unlink(tmp)
        b = c.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
        log(f"[{name}] submitted OpenAI batch {b.id} with {len(todo)} requests")
        while b.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(poll_s)
            b = c.batches.retrieve(b.id)
            rc = b.request_counts
            log(f"[{name}] {b.status}: {rc.completed if rc else 0}/{rc.total if rc else '?'} done")
        usage = {"input_tokens": 0, "output_tokens": 0}
        lines = []
        if b.output_file_id:
            lines += c.files.content(b.output_file_id).text.splitlines()
        if b.error_file_id:
            lines += c.files.content(b.error_file_id).text.splitlines()
        with open(path, "a") as f:
            for line in lines:
                r = json.loads(line)
                resp = r.get("response") or {}
                body = resp.get("body") or {}
                rec = {"custom_id": r["custom_id"], "ok": resp.get("status_code") == 200 and "choices" in body}
                if rec["ok"]:
                    ch = body["choices"][0]
                    rec["text"] = ch["message"].get("content") or ""
                    rec["stop_reason"] = ch.get("finish_reason")
                    rec["usage"] = {"input_tokens": body["usage"]["prompt_tokens"],
                                    "output_tokens": body["usage"]["completion_tokens"]}
                    for k in usage:
                        usage[k] += rec["usage"][k]
                else:
                    rec["error"] = json.dumps(r.get("error") or body.get("error") or resp)[:500]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[r["custom_id"]] = rec
        usd = cost(model, usage)
        log_cost(name, model, usage, usd, len(todo), batch=True)
        log(f"[{name}] {b.status}: {sum(done.get(k, {}).get('ok', False) for k in todo)}/{len(todo)} ok, "
            f"{usage['input_tokens']} in / {usage['output_tokens']} out tokens, ${usd:.2f}")
    return {k: done[k] for k in requests if k in done}
