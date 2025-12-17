from typing import Dict, Any, List
from ontology.ontology_loader import OntologyStore

_ontology: OntologyStore | None = None

def get_ontology() -> OntologyStore:
    global _ontology
    if _ontology is None:
        _ontology = OntologyStore()
        _ontology.load_minimal("ontology/min_ontology.json")
    return _ontology

def expand_query(query: str) -> Dict[str, List[str]]:
    onto = get_ontology()
    tokens = query.lower().replace('/', ' ').split()
    policy_terms, provision_terms = onto.expand_terms(tokens)
    return {
        "original": query,
        "policy_terms": policy_terms,
        "provision_terms": provision_terms,
        "all_expansion": list(set(policy_terms + provision_terms))
    }

def build_expanded_queries(query: str, max_extra: int = 6) -> List[str]:
    exp = expand_query(query)
    if not exp["all_expansion"]:
        return [query]
    extras = exp["all_expansion"][:max_extra]
    # Basic permutations: original + appended tokens
    augmented = [query]
    for term in extras:
        if term not in query.lower():
            augmented.append(f"{query} {term}")
    return augmented[:1+max_extra]
