
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import yaml

CANONICAL_COLUMNS = [
    "country", "year", "sex",
    "prevalence_total", "prevalence_male", "prevalence_female",
    "prevalence_depression", "prevalence_anxiety",
]

@dataclass
class ColumnConfig:
    mapping: Dict[str, str]

    @classmethod
    def from_yaml(cls, path: str) -> "ColumnConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(mapping=raw)

    def canonicalize(self, cols: list[str]) -> Dict[str, str]:
        # return dict of {raw_name: canonical_name} for those that exist
        inv = {v: k for k, v in self.mapping.items() if v}
        # if user mistakenly reversed mapping, try to detect
        m = {}
        for c in cols:
            if c in self.mapping.values():
                # already mapped name in file, treat as raw->canonical identity
                m[c] = c
            elif c in self.mapping:
                m[c] = self.mapping[c]
        # if mapping is empty, try identity for canonical names
        for c in cols:
            if c in CANONICAL_COLUMNS and c not in m:
                m[c] = c
        return m
