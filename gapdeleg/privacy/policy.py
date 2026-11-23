from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .types import Label

DEFAULT_POLICY = Path(__file__).with_name("policy.yaml")


@dataclass
class Policy:
    categories: dict[str, Label]
    default_label: Label = Label.LOCAL_ONLY
    guard: dict = field(default_factory=dict)

    def label_for(self, category: str) -> Label:
        return self.categories.get(category.upper(), self.default_label)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY) -> "Policy":
        d = yaml.safe_load(Path(path).read_text())
        return cls(categories={k.upper(): Label(v) for k, v in d["categories"].items()},
                   default_label=Label(d.get("default_label", "local-only")),
                   guard=d.get("guard", {}))
