"""
Query Intent Types and Data Classes for GIRA AI (Government Information Retrieval)
"""

from enum import Enum
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


class QueryIntent(Enum):
    """Government policy query intent categories"""
    POLICY_INFORMATION = "policy_information"      # General info about a policy/scheme
    PROCEDURAL_STEPS = "procedural_steps"          # How to apply/register/comply
    ELIGIBILITY_CRITERIA = "eligibility_criteria"  # Who qualifies (citizenship, income)
    DOCUMENT_REQUIREMENTS = "document_requirements" # Required paperwork (ID, Tax, etc.)
    LEGAL_FRAMEWORK = "legal_framework"            # Acts, articles, constitution, amendments
    COMPLIANCE_ISSUE = "compliance_issue"          # Violations, penalties, fines
    BENEFITS_SUBSIDIES = "benefits_subsidies"      # Grants, loans, financial aid
    TAXATION_FISCAL = "taxation_fiscal"            # Tax rates, rebates, budget
    JURISDICTION_AUTHORITY = "jurisdiction_authority" # Which department handles what
    CONTACT_LOCATIONS = "contact_locations"        # Where to go (offices, portals)
    TIMELINES_DEADLINES = "timelines_deadlines"    # Processing time, last dates
    GRIEVANCE_REDRESSAL = "grievance_redressal"    # Complaints, appeals
    INTERNATIONAL_AFFAIRS = "international_affairs" # Trade, visas, treaties
    PUBLIC_SAFETY = "public_safety"                # Emergency, police, disaster management


@dataclass
class IntentClassification:
    """Classification result with confidence and features"""
    intent: QueryIntent
    confidence: float
    features: Dict[str, Any]
    reasoning: str
    alternative_intents: List[Tuple[QueryIntent, float]]
