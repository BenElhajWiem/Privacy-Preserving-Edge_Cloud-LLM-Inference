from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field


@dataclass
class GapItem:
    text: str
    may_leave: bool = False


@dataclass
class GapReport:
    established: list[GapItem] = field(default_factory=list)
    uncertain: list[GapItem] = field(default_factory=list)
    missing: list[GapItem] = field(default_factory=list)
    parsed: bool = True

    def gap_requests(self, only_may_leave: bool = True) -> list[str]:
        return [i.text for i in self.uncertain + self.missing if i.may_leave or not only_may_leave]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "GapReport":
        def items(key):
            out = []
            for x in d.get(key, []) or []:
                if isinstance(x, str):
                    out.append(GapItem(x))
                elif isinstance(x, dict) and "text" in x:
                    out.append(GapItem(str(x["text"]), bool(x.get("may_leave", False))))
            return out
        return cls(items("established"), items("uncertain"), items("missing"))

    @classmethod
    def parse(cls, text: str) -> "GapReport":
        text = text.split("</think>")[-1]
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return cls(parsed=False)
        try:
            return cls.from_dict(json.loads(m.group(0)))
        except json.JSONDecodeError:
            return cls(parsed=False)
