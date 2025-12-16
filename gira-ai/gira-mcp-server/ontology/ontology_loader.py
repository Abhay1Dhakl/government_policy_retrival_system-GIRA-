import json, re
from pathlib import Path
from typing import Dict, List, Set

class OntologyStore:
    def __init__(self):
        self.policy_syn: Dict[str, Set[str]] = {}
        self.provision_syn: Dict[str, Set[str]] = {}
        self.normalizer = re.compile(r'[^a-z0-9]+')

    def _norm(self, s: str) -> str:
        return self.normalizer.sub(' ', s.lower()).strip()

    def add_concept(self, base: str, synonyms: List[str], bucket: Dict[str, Set[str]]):
        b = self._norm(base)
        bucket.setdefault(b, set()).add(b)
        for s in synonyms:
            bucket[b].add(self._norm(s))

    def load_minimal(self, path: str):
        data = json.loads(Path(path).read_text())
        for d in data.get("policies", []):
            self.add_concept(d["name"], d.get("synonyms", []), self.policy_syn)
        for a in data.get("provisions", []):
            self.add_concept(a["name"], a.get("synonyms", []), self.provision_syn)

    def expand_terms(self, tokens: List[str]):
        toks = [self._norm(t) for t in tokens]
        policy_hits, provision_hits = set(), set()
        for base, syns in self.policy_syn.items():
            if syns & set(toks):
                policy_hits |= syns
        for base, syns in self.provision_syn.items():
            if syns & set(toks):
                provision_hits |= syns
        return list(policy_hits), list(provision_hits)
