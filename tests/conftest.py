import pytest

from gapdeleg.backends.base import CloudModel, Generation, Usage
from gapdeleg.privacy.policy import Policy


class FakeEdge:
    name = "fake-edge"

    def __init__(self, replies=None, logprobs=(-0.1, -0.2)):
        self.replies = list(replies or [])
        self.logprobs = list(logprobs)
        self.calls = []

    def generate(self, messages, max_tokens=512, temperature=0.0, logprobs=True, **kw):
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else "edge answer"
        return Generation(text, list(self.logprobs) if logprobs else None, None, Usage(10, 5), 0.01, self.name)

    def describe(self):
        return {"backend": "fake", "model": self.name}


class FakeCloud(CloudModel):
    name = "fake-cloud"

    def __init__(self, reply="cloud answer", cache_dir=None):
        super().__init__(cache_dir)
        self.reply, self.calls = reply, []

    def _generate(self, messages, **params):
        self.calls.append(messages)
        reply = self.reply(messages) if callable(self.reply) else self.reply
        return Generation(reply, usage=Usage(20, 8), model=self.name)


class FakeDetector:

    def __init__(self, hits=None):
        self.hits = hits or {}

    def detect(self, text):
        from gapdeleg.privacy.types import Label, Span
        out = []
        for s, cat in self.hits.items():
            i = text.find(s)
            if i >= 0:
                out.append(Span(s, Label.LOCAL_ONLY, cat, i, i + len(s), "fake", 0.99))
        return out


@pytest.fixture
def policy():
    return Policy.load()
