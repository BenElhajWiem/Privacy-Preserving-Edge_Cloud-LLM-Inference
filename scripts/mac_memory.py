#!/usr/bin/env python
import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HOST = "http://localhost:11434"


def api(path: str, body: dict | None = None, timeout: float = 600):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ollama_rss() -> int:
    total = 0
    for p in psutil.process_iter(["name", "exe"]):
        try:
            if "ollama" in ((p.info["name"] or "") + (p.info["exe"] or "")).lower():
                total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def unload_all():
    for m in api("/api/ps")["models"]:
        api("/api/generate", {"model": m["name"], "keep_alive": 0})
    for _ in range(60):
        if not api("/api/ps")["models"]:
            return
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from gapdeleg.build.io import read_examples
    keep = set((ROOT / "configs/mac_cost_ids.txt").read_text().split())
    longest = max((e for e in read_examples("pupa_tnb") if e.id in keep), key=lambda e: len(e.query))
    out = {"prompt": longest.id, "prompt_chars": len(longest.query), "models": {}}
    for model in args.models:
        unload_all()
        base = ollama_rss()
        peak, stop = [base], threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], ollama_rss())
                stop.wait(0.5)
        t = threading.Thread(target=sample, daemon=True)
        t.start()
        r = api("/api/chat", {"model": model, "stream": False, "think": False,
                              "messages": [{"role": "user", "content": longest.query}],
                              "options": {"num_ctx": 16384, "num_predict": 1024, "temperature": 0}})
        size = next((m["size"] for m in api("/api/ps")["models"] if m["name"] == model or m["model"] == model), None)
        stop.set()
        t.join()
        out["models"][model] = {"peak_ollama_rss_gb": round(peak[0] / 2**30, 3),
                                "idle_ollama_rss_gb": round(base / 2**30, 3),
                                "ollama_reported_model_gb": round(size / 2**30, 3) if size else None,
                                "prompt_tokens": r.get("prompt_eval_count"), "written_tokens": r.get("eval_count")}
        print(model, out["models"][model], flush=True)
    unload_all()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "memory.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
