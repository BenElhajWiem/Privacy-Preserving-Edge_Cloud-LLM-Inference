from __future__ import annotations

import math

from gapdeleg.backends.base import Generation


def _content_logprobs(gen: Generation) -> list[float]:
    lps = gen.token_logprobs or []
    if gen.tokens and "</think>" in "".join(gen.tokens):
        acc, cut = "", 0
        for i, t in enumerate(gen.tokens):
            acc += t
            if "</think>" in acc:
                cut = i + 1
                break
        lps = lps[cut:]
    return lps


def mean_logprob(gen: Generation) -> float:
    lps = _content_logprobs(gen)
    if not lps:
        raise ValueError("generation has no token logprobs")
    return sum(lps) / len(lps)


def min_logprob(gen: Generation) -> float:
    lps = _content_logprobs(gen)
    if not lps:
        raise ValueError("generation has no token logprobs")
    return min(lps)


class MeanLogprob:
    name = "mean_logprob"

    def score(self, question: str, generation: Generation) -> float:
        return math.exp(mean_logprob(generation))
