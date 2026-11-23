import importlib
import logging
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envfile import ENV_FILE, load_env

REPO = Path(__file__).resolve().parent.parent
EXTERNAL = Path(os.environ.get("GAPDELEG_EXTERNAL", REPO / "external"))
MODELS_DIR = Path(os.environ.get("GAPDELEG_MODELS", Path.home() / "gapdeleg-models"))
RESULTS = Path(".")

CLOUD_MODEL = os.environ.get("GAPDELEG_CLOUD_MODEL", "claude-sonnet-5-5")
CHEAP_MODEL = os.environ.get("GAPDELEG_CHEAP_MODEL", "claude-haiku-5-5")
SMOKE_GGUF = ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf")
SMOKE_MLX = "mlx-community/Qwen3-0.6B-4bit"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
OLLAMA_SMOKE = os.environ.get("GAPDELEG_OLLAMA_SMOKE", "llama3.2:1b-instruct-q4_K_M")

rows = []


def record(group, name, status, detail=""):
    rows.append((group, name, status, detail))
    print(f"[{status:4}] {group:9} {name:32} {detail}")


def check(group, name, fn):
    try:
        detail = fn()
        record(group, name, "PASS", detail or "")
    except SkipCheck as e:
        record(group, name, "SKIP", str(e))
    except Exception as e:
        record(group, name, "FAIL", f"{type(e).__name__}: {e}"[:160])


class SkipCheck(Exception):
    pass


def pkg(mod):
    def _f():
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "")
    return _f


def reachable(host):
    def _f():
        ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
        req = urllib.request.Request(f"https://{host}/", method="HEAD")
        try:
            urllib.request.urlopen(req, timeout=15, context=ctx)
        except urllib.error.HTTPError as e:
            return f"HTTP {e.code}"
        except urllib.error.URLError as e:
            if "403" in str(e) or "Tunnel" in str(e):
                raise RuntimeError("blocked by network policy") from None
            raise
        return "HTTP 200"
    return _f


def spacy_ner():
    import spacy
    nlp = spacy.load("en_core_web_lg")
    ents = [(e.text, e.label_) for e in nlp("Rachel Zheng works at H&R Technology in London.").ents]
    assert ("Rachel Zheng", "PERSON") in ents, ents
    return f"{len(ents)} entities"


def presidio():
    logging.getLogger("tldextract").setLevel(logging.CRITICAL)
    from presidio_analyzer import AnalyzerEngine
    eng = AnalyzerEngine()
    text = "My name is Rachel Zheng, call me at 212-555-0199 or rachel.zheng@example.com."
    found = {r.entity_type for r in eng.analyze(text=text, language="en")}
    need = {"PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"}
    assert need <= found, found
    return ", ".join(sorted(found))


def pupa():
    import pandas as pd
    tnb = pd.read_csv(EXTERNAL / "PAPILLON/pupa/PUPA_TNB.csv")
    new = pd.read_csv(EXTERNAL / "PAPILLON/pupa/PUPA_New.csv")
    assert len(tnb) == 237 and len(new) == 664, (len(tnb), len(new))
    return "TNB 237, New 664"


def prism_data():
    import pandas as pd
    sheets = pd.read_excel(EXTERNAL / "PRISM/Dataset/prism_dataset.xlsx", sheet_name=None)
    return ", ".join(f"{k} {len(v)}" for k, v in sheets.items())


def gsm8k():
    n = sum(1 for _ in open(EXTERNAL / "grade-school-math/grade_school_math/data/test.jsonl"))
    return f"test {n}"


def file_exists(path):
    def _f():
        assert path.exists(), f"missing {path}"
        return ""
    return _f


def anthropic_call():
    key = os.environ.get("GAPDELEG_ANTHROPIC_API_KEY")
    if not key:
        raise SkipCheck("GAPDELEG_ANTHROPIC_API_KEY not set")
    import anthropic
    client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com")
    out = []
    for model in (CLOUD_MODEL, CHEAP_MODEL):
        t = time.time()
        r = client.messages.create(
            model=model,
            max_tokens=64,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        out.append(f"{model}: '{text}' ({r.usage.input_tokens}+{r.usage.output_tokens} tok, {time.time()-t:.1f}s)")
    return "; ".join(out)


def openai_call():
    key = os.environ.get("GAPDELEG_OPENAI_API_KEY")
    if not key:
        raise SkipCheck("optional; GAPDELEG_OPENAI_API_KEY not set")
    from openai import OpenAI
    r = OpenAI(api_key=key).chat.completions.create(
        model=os.environ.get("GAPDELEG_OPENAI_MODEL", "gpt-4o-mini"),
        max_tokens=8, messages=[{"role": "user", "content": "Reply with: ready"}])
    return r.choices[0].message.content.strip()


def hf_download():
    from huggingface_hub import hf_hub_download
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(SMOKE_GGUF[0], SMOKE_GGUF[1], local_dir=MODELS_DIR)
    return f"{Path(p).stat().st_size/1e6:.0f} MB -> {p}"


def hf_gated():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    try:
        hf_hub_download("meta-llama/Llama-3.2-1B-Instruct", "config.json",
                        local_dir=MODELS_DIR / "gated-check")
    except GatedRepoError as e:
        raise SkipCheck(f"optional, Llama/Gemma come from Ollama; HF gated access denied ({str(e)[:24]})") from None
    src = "env HF_TOKEN" if os.environ.get("HF_TOKEN") else "proxy-injected token"
    return f"gated access OK via {src}"


def ollama_generate():
    import json
    body = {"model": OLLAMA_SMOKE, "stream": False, "logprobs": True, "top_logprobs": 1,
            "options": {"temperature": 0, "num_predict": 16},
            "messages": [{"role": "user", "content": "What is 17 + 25? Answer with the number only."}]}
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    tps = r["eval_count"] / (r["eval_duration"] / 1e9)
    return (f"{OLLAMA_SMOKE}: '{r['message']['content'].strip()[:20]}', {tps:.1f} tok/s, "
            f"logprobs={'yes' if r.get('logprobs') else 'no'}")


def llama_cpp_generate():
    gguf = MODELS_DIR / SMOKE_GGUF[1]
    if not gguf.exists():
        raise SkipCheck(f"no model at {gguf} (needs huggingface.co)")
    from llama_cpp import Llama
    llm = Llama(model_path=str(gguf), n_ctx=2048, n_threads=os.cpu_count(), n_gpu_layers=-1,
                logits_all=True, verbose=False)
    t = time.time()
    r = llm.create_chat_completion(
        messages=[{"role": "user", "content": "What is 17 + 25? Answer with the number only. /no_think"}],
        max_tokens=32, temperature=0, logprobs=True, top_logprobs=1)
    dt = time.time() - t
    n = r["usage"]["completion_tokens"]
    has_lp = r["choices"][0].get("logprobs") is not None
    answer = r["choices"][0]["message"]["content"].split("</think>")[-1].strip()[:30]
    return f"'{answer}', {n/dt:.1f} tok/s, logprobs={'yes' if has_lp else 'no'}"


def mlx():
    if sys.platform != "darwin":
        raise SkipCheck("Mac only")
    import mlx.core as mx
    return f"mlx {mx.__version__}, device {mx.default_device()}"


def mlx_generate():
    if sys.platform != "darwin":
        raise SkipCheck("Mac only")
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    model, tok = load(SMOKE_MLX)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Write a short paragraph about the ocean. /no_think"}],
        add_generation_prompt=True, tokenize=False)
    for _ in stream_generate(model, tok, prompt, max_tokens=8):
        pass
    last = None
    for last in stream_generate(model, tok, prompt, max_tokens=128):
        pass
    return (f"{last.generation_tokens} tok at {last.generation_tps:.1f} tok/s, "
            f"peak {mx.get_peak_memory()/1e9:.2f} GB")


def main():
    loaded = load_env()
    print(f"python {sys.version.split()[0]} on {sys.platform}, {os.cpu_count()} cpus")
    print(f".env: {ENV_FILE} -> loaded {', '.join(loaded) or 'nothing'}\n")
    for mod in ["anthropic", "openai", "dspy", "presidio_analyzer", "spacy", "rapidfuzz",
                "llama_cpp", "huggingface_hub", "pandas", "sklearn", "scipy", "yaml"]:
        check("package", mod, pkg(mod))
    check("package", "mlx (Mac)", mlx)

    for host in ["github.com", "api.anthropic.com", "huggingface.co", "cas-bridge.xethub.hf.co", "arxiv.org", "export.arxiv.org", "api.openai.com"]:
        check("network", host, reachable(host))

    check("privacy", "spaCy en_core_web_lg NER", spacy_ner)
    check("privacy", "Presidio analyzer", presidio)

    check("data", "PUPA (PAPILLON repo)", pupa)
    check("data", "PRISM dataset", prism_data)
    check("data", "GSM8K", gsm8k)
    check("data", "PAPILLON 1B/3B prompts", file_exists(EXTERNAL / "PAPILLON/papillon/optimized_prompts/llama_32_1b_instruct_prompt.json"))

    check("cloud", f"Anthropic API", anthropic_call)
    check("cloud", "OpenAI API", openai_call)

    check("edge", f"HF download {SMOKE_GGUF[1]}", hf_download)
    check("edge", "HF gated model (Llama-3.2)", hf_gated)
    check("edge", "Ollama generate + logprobs", ollama_generate)
    check("edge", "llama.cpp generate + logprobs", llama_cpp_generate)
    check("edge", f"MLX generate {SMOKE_MLX.split('/')[1]}", mlx_generate)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "env_check.md"
    with open(out, "w") as f:
        f.write(f"# Environment check\n\nRun {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} "
                f"on {sys.platform}, python {sys.version.split()[0]}, {os.cpu_count()} cpus.\n\n")
        f.write("| Group | Check | Status | Detail |\n|---|---|---|---|\n")
        for g, n, s, d in rows:
            f.write(f"| {g} | {n} | {s} | {d.replace('|', '/')} |\n")
    fails = sum(1 for r in rows if r[2] == "FAIL")
    print(f"\n{fails} failing checks; table written to {out}")


if __name__ == "__main__":
    main()
