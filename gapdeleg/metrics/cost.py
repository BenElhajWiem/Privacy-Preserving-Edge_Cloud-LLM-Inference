from __future__ import annotations

from statistics import mean, median

from gapdeleg.pipeline.context import Trace


def request_cost(trace: Trace, wall_s: float) -> dict:
    return {
        "cloud_calls": len(trace.cloud),
        "cloud_cached_calls": sum(g.cached for _, g in trace.cloud),
        "cloud_in_tokens": sum(g.usage.input_tokens for _, g in trace.cloud),
        "cloud_out_tokens": sum(g.usage.output_tokens for _, g in trace.cloud),
        "cloud_latency_s": sum(g.latency_s for _, g in trace.cloud if not g.cached),
        "guard_denials": len(trace.guard_denied),
        "edge_calls": len(trace.edge),
        "edge_in_tokens": sum(g.usage.input_tokens for _, g in trace.edge),
        "edge_out_tokens": sum(g.usage.output_tokens for _, g in trace.edge),
        "edge_latency_s": sum(g.latency_s for _, g in trace.edge),
        "wall_s": wall_s,
    }


def aggregate(costs: list[dict]) -> dict:
    if not costs:
        return {}
    out = {}
    for k in costs[0]:
        vals = [c[k] for c in costs]
        out[k] = {"sum": sum(vals), "mean": mean(vals), "median": median(vals)}
    return out
