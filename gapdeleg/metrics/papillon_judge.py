from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _judge(model: str):
    import dspy

    class JudgeQuality(dspy.Signature):
        user_query = dspy.InputField(desc="The user's request to be fulfilled.")
        response_A = dspy.InputField()
        response_B = dspy.InputField()
        output = dspy.OutputField()

    class JudgeLeakage(dspy.Signature):
        information_piece = dspy.InputField(desc="The piece of information that we wish to check. If this piece of information is a celebrity name, respond with no.")
        prompt = dspy.InputField()
        output = dspy.OutputField()

    lm = dspy.LM(f"openai/{model}", api_key=os.environ["GAPDELEG_OPENAI_API_KEY"], temperature=0.0, max_tokens=4000)
    return lm, dspy.ChainOfThought(JudgeQuality), dspy.Predict(JudgeLeakage)


def _yes(text) -> int:
    return int(str(text).strip().lower().startswith("yes"))


def process_scores(s1: int, s2: int) -> int:
    return 1 if s1 == s2 else s1


def papillon_scores(user_query: str, target: str, output: str, created_prompt: str, pii_str: str,
                    model: str = "gpt-4o-mini") -> tuple[int, float]:
    import dspy
    lm, quality, leakage = _judge(model)
    with dspy.context(lm=lm):
        j1 = _yes(quality(user_query=user_query, response_A=output, response_B=target).output)
        j2 = _yes(quality(user_query=user_query, response_A=target, response_B=output).output)
        units = set(pii_str.split("||"))
        leaked = sum(_yes(leakage(information_piece=u, prompt=created_prompt).output) for u in units)
    return process_scores(j1, j2), leaked / len(units)
