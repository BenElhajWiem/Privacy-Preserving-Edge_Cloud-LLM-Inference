from __future__ import annotations

import re
import sys
from functools import lru_cache

from gapdeleg.data.loaders import EXTERNAL
from gapdeleg.data.schema import Example
from gapdeleg.pipeline.context import Context, Trace, strip_think, user

CODE = EXTERNAL / "PRISM/Code"

QWEN3_NO_THINK = "<think>\n\n</think>\n\n"


def no_think_suffix(model: str) -> str:
    return QWEN3_NO_THINK if model.startswith("qwen3:") and "2507" not in model else ""


@lru_cache(maxsize=1)
def _modules():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import logging
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    from cloud_sketch_generator import CloudSketchGenerator
    from edge_denoising import EdgeDenoisingReintegrator
    from edge_detection import EdgeEntityDetector
    from soft_gating import RoutingMode, SoftGatingPredictor
    from two_layer_ldp import TwoLayerLDP
    return CloudSketchGenerator, EdgeDenoisingReintegrator, EdgeEntityDetector, RoutingMode, SoftGatingPredictor, \
        TwoLayerLDP


def parse_edge_examples(content: str) -> dict[str, list[dict]]:
    examples: dict[str, list[dict]] = {"Tourism": [], "Medical": [], "Banking": [], "Common": []}
    domain, cur, field = None, None, None
    for raw in content.splitlines():
        line = raw.strip()
        head = re.match(r"^##\s*(\w+)\s+DOMAIN", line, re.I)
        if head:
            domain = {"TOURISM": "Tourism", "MEDICAL": "Medical", "BANKING": "Banking"}.get(head.group(1).upper(), "Common")
            continue
        m = re.match(r"^(Prompt|Sketch|Response)\s*\(.*?\)\s*:\s*(.*)$", line)
        if m and domain:
            key = m.group(1).lower()
            if key == "prompt":
                cur = {}
                examples[domain].append(cur)
            cur[key] = m.group(2).strip()
            field = key
        elif field == "response" and cur is not None and line and not line.startswith(("**", "---", "##")):
            cur["response"] += "\n" + line
        else:
            field = None if not line or line.startswith(("**", "---", "##")) else field
    for exs in examples.values():
        for ex in exs:
            for k in ex:
                ex[k] = ex[k].strip().strip('"').strip()
    return {k: [e for e in v if {"prompt", "sketch", "response"} <= e.keys()] for k, v in examples.items()}


class _Edge:

    def __new__(cls, base, temperature: float, top_p: float, max_tokens: int):
        obj = base.__new__(base)
        import logging
        obj.logger = logging.getLogger("EdgeDenoisingReintegrator")
        obj.temperature, obj.top_p, obj.max_tokens = temperature, top_p, max_tokens
        obj.examples = parse_edge_examples((CODE / "few_shot_examples_edge.txt").read_text())
        obj.placeholder_patterns = [r'\[([A-Z_]+)\]_(\w+)', r'\[([A-Z_]+)\]', r'\[MASKED_([A-Z_]+)\]', r'\[XXX\]',
                                    r'<([A-Z_]+)>']
        return obj


class PrismPort:
    name = "prism"
    needs_cloud = True

    def __init__(self, epsilon_total: float = 2.0, alpha: float = 0.5, risk_threshold: float = 0.5,
                 privacy_threshold: float = 0.5, slm_temperature: float = 0.7, slm_top_p: float = 0.9,
                 slm_max_tokens: int = 512, sketch_max_tokens: int = 200, guarded: bool = False, seed: int = 0):
        import random

        import numpy as np
        Cloud, EdgeBase, Detector, self.RoutingMode, Gate, LDP = _modules()
        random.seed(seed)
        np.random.seed(seed)
        self.detector = Detector(risk_threshold=risk_threshold, privacy_threshold=privacy_threshold)
        self.gate = Gate(str(CODE / "models/soft_gating_pretrained.pth"))
        self.ldp = LDP(alpha=alpha)
        self.sketcher = Cloud(api_key="unused", examples_file=str(CODE / "few_shot_examples_cloud.txt"))
        self.edge = _Edge(EdgeBase, slm_temperature, slm_top_p, slm_max_tokens)
        self.epsilon_total = epsilon_total
        self.sketch_max_tokens = sketch_max_tokens
        self.guard_disabled = not guarded

    def _edge_complete(self, trace: Trace, ctx: Context, step: str, prompt: str) -> str:
        e = self.edge
        g = ctx.edge.complete(prompt + no_think_suffix(getattr(ctx.edge, "name", "")), max_tokens=e.max_tokens,
                              temperature=e.temperature, top_p=e.top_p, stop=["###", "\n\n\n"])
        trace.edge.append((step, g))
        return strip_think(g.text).strip()

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        det = self.detector.detect_and_classify(ex.query)
        mode, probs = self.gate.route(det["risk_score"], det["sensitivity_labels"], return_probs=True)
        trace.notes.update(route=mode.name, risk=det["risk_score"], route_probs=probs,
                           n_entities=len(det["entities"]), n_protected=len(det["protected_entities"]))
        spans = [] if self.guard_disabled else trace.label(ctx, ex.query)

        if mode == self.RoutingMode.EDGE_ONLY:
            category = self.edge._detect_category(ex.query)
            return self._edge_complete(trace, ctx, "edge_only", self.edge._build_direct_prompt(ex.query, category))

        if mode == self.RoutingMode.CLOUD_ONLY:
            g = trace.cloud_call(ctx, "cloud_only", user(self.sketcher.build_few_shot_prompt(
                self.sketcher.detect_category(ex.query), ex.query)), spans, max_tokens=self.sketch_max_tokens)
            return re.sub(r"^Sketch:\s*", "", g.text.strip()) if g else ""

        cloud_input = ex.query
        for etype, _, _, value in det["protected_entities"]:
            cloud_input = cloud_input.replace(value, self.ldp.two_layer_dp(etype, value, self.epsilon_total), 1)
        trace.notes["cloud_input"] = cloud_input
        category = self.sketcher.detect_category(cloud_input)
        g = trace.cloud_call(ctx, "cloud_sketch", user(self.sketcher.build_few_shot_prompt(category, cloud_input)),
                             spans, max_tokens=self.sketch_max_tokens)
        if g is None:
            trace.notes["failed"] = "guard denied the perturbed prompt"
            return ""
        sketch = re.sub(r"^Sketch:\s*", "", g.text.strip())
        denoised = self.edge._replace_obfuscated_entities(sketch, ex.query, det["entities"], det["protected_entities"])
        out = self._edge_complete(trace, ctx, "edge_refine",
                                  self.edge._build_refinement_prompt(denoised, ex.query, category))
        if category == "Medical" and "disclaimer" not in out.lower():
            out += "\n\n*This information is for reference only and does not replace professional medical advice.*"
        return out
