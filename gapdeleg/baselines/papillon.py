from __future__ import annotations

import json
from pathlib import Path

from gapdeleg.data.loaders import EXTERNAL
from gapdeleg.data.schema import Example
from gapdeleg.pipeline.context import Context, Trace, strip_think, user

PROMPTS = EXTERNAL / "PAPILLON/papillon/optimized_prompts"
NAMED = {"llama_32_1b": "llama_32_1b_instruct_prompt.json", "llama_32_3b": "llama_32_3b_instruct_prompt.json",
         "llama_31_8b": "llama_31_8b_instruct_prompt.json", "llama_3_8b": "llama_3_8b_instruct_prompt.json",
         "mistral_7b": "mistral_7b_instruct_prompt.json", "mistral_small": "mistral_small_prompt.json"}

CREATOR_DEFAULT = ("You are a helpful assistant that is very mindful of user privacy. You have access to a powerful "
                   "large language model that you can query. Given a user request, create a prompt for your large "
                   "language model that preserves user privacy, so that this model can help you complete the user "
                   "request. Provide the prompt directly without any preamble. DO NOT COMPLETE THE USER QUERY, ONLY "
                   "GENERATE A PROMPT.")
AGGREGATOR_DEFAULT = "You are a helpful assistant. Respond to queries from the user."


def _signatures(creator_instr: str, aggregator_instr: str):
    import dspy

    class CreateOnePrompt(dspy.Signature):
        userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
        createdPrompt = dspy.OutputField()

    class InfoAggregator(dspy.Signature):
        userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
        modelExampleResponses = dspy.InputField(
            desc="Information from a more powerful language model responding to related queries. Complete the user "
                 "query by referencing this information. Only you have access to this information.")
        finalOutput = dspy.OutputField()

    creator = dspy.ChainOfThought(CreateOnePrompt.with_instructions(creator_instr))
    aggregator = dspy.Predict(InfoAggregator.with_instructions(aggregator_instr))
    return creator.predict.signature, aggregator.signature


def _json_schema(sig) -> dict:
    fields = list(sig.output_fields)
    return {"type": "object", "properties": {f: {"type": "string"} for f in fields}, "required": fields}


def load_instructions(prompts: str | None) -> tuple[str, str]:
    if not prompts:
        return CREATOR_DEFAULT, AGGREGATOR_DEFAULT
    path = PROMPTS / NAMED[prompts] if prompts in NAMED else Path(prompts)
    d = json.loads(path.read_text())
    pc, ia = d["prompt_creater"], d["info_aggregator"]
    return (pc.get("extended_signature_instructions") or pc["signature_instructions"],
            ia.get("extended_signature_instructions") or ia["signature_instructions"])


class PapillonPort:
    name = "papillon"
    needs_cloud = True

    def __init__(self, prompts: str | None = "llama_32_3b", guarded: bool = False, remote_max_tokens: int = 4000):
        import dspy
        self.prompts = prompts
        self.guard_disabled = not guarded
        self.remote_max_tokens = remote_max_tokens
        self.creator_sig, self.aggregator_sig = _signatures(*load_instructions(prompts))
        self.adapters = (("chat", dspy.ChatAdapter()), ("json", dspy.JSONAdapter()))

    def _run(self, trace: Trace, ctx: Context, step: str, sig, inputs: dict) -> dict:
        for fmt, adapter in self.adapters:
            msgs = adapter.format(sig, demos=[], inputs=inputs)
            kw = {"format": _json_schema(sig)} if fmt == "json" else {}
            g = trace.edge_call(ctx, step if fmt == "chat" else f"{step}_json", msgs, logprobs=False, **kw)
            try:
                out = adapter.parse(sig, strip_think(g.text))
            except Exception:
                continue
            trace.notes[f"{step}_format"] = fmt
            return out
        trace.notes[f"{step}_format"] = "failed"
        return {}

    def __call__(self, ex: Example, ctx: Context, trace: Trace) -> str:
        created = self._run(trace, ctx, "prompt_creator", self.creator_sig, {"userQuery": ex.query})
        prompt = (created.get("createdPrompt") or "").strip()
        trace.notes["created_prompt"] = prompt
        if not prompt:
            trace.notes["failed"] = ("prompt creator output unparseable"
                                     if trace.notes["prompt_creator_format"] == "failed" else "empty created prompt")
            return ""
        spans = [] if self.guard_disabled else trace.label(ctx, ex.query)
        remote = trace.cloud_call(ctx, "remote", user(prompt), spans, max_tokens=self.remote_max_tokens)
        if remote is None:
            trace.notes["failed"] = "guard denied the created prompt"
            return ""
        trace.notes["remote_response"] = remote.text
        out = self._run(trace, ctx, "info_aggregator", self.aggregator_sig,
                        {"userQuery": ex.query, "modelExampleResponses": remote.text})
        return (out.get("finalOutput") or "").strip()
