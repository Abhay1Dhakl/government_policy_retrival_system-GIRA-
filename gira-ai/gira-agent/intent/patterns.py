"""
Intent Patterns and Keywords for Government Query Classification
Contains regex patterns and keyword lists for detecting government policy intents
"""

import re
from typing import Dict, List, Tuple, Any
from intent.types import QueryIntent


def build_intent_patterns() -> Dict[QueryIntent, List[Tuple[re.Pattern, float]]]:
    """Build regex patterns for intent classification"""
    return {
        QueryIntent.PROCEDURAL_STEPS: [
            (re.compile(r'\b(how to (apply|register|get|obtain)|application process|registration steps|procedure|workflow)\b', re.IGNORECASE), 3.0),
            (re.compile(r'\b(steps?|guide|method|way to)\b', re.IGNORECASE), 1.5)
        ],
        QueryIntent.ELIGIBILITY_CRITERIA: [
            (re.compile(r'\b(eligib(le|ility)|qualif(y|ication)|who can|criteria|requirements?|prerequisites?)\b', re.IGNORECASE), 3.0),
            (re.compile(r'\b(age limit|income limit|nationality|residency)\b', re.IGNORECASE), 2.0)
        ],
        QueryIntent.DOCUMENT_REQUIREMENTS: [
            (re.compile(r'\b(documents?|paperwork|forms?|certificates?|proof of)\b', re.IGNORECASE), 2.5),
            (re.compile(r'\b(id card|passport|visa|license|permit|tax return|pan card|aadhar|ssn)\b', re.IGNORECASE), 2.0),
            (re.compile(r'\b(attach|submit|file|upload)\b', re.IGNORECASE), 1.5)
        ],
        QueryIntent.LEGAL_FRAMEWORK: [
            (re.compile(r'\b(act|bill|law|article|section|amendment|constitution|gazette|regulation|rule)\b', re.IGNORECASE), 2.5),
            (re.compile(r'\b(supreme court|high court|ruling|judgment|verdict)\b', re.IGNORECASE), 2.0)
        ],
        QueryIntent.COMPLIANCE_ISSUE: [
            (re.compile(r'\b(penalty|fine|violation|illegal|offence|breach|punishment|ban|prohibited)\b', re.IGNORECASE), 2.5),
            (re.compile(r'\b(mandatory|compulsory|obligatory|enforcement)\b', re.IGNORECASE), 2.0)
        ],
        QueryIntent.BENEFITS_SUBSIDIES: [
            (re.compile(r'\b(subsidy|subsidies|grant|scholarship|pension|welfare|allowance|financial aid|loan|scheme)\b', re.IGNORECASE), 3.0),
            (re.compile(r'\b(free|discount|remission|waiver|benefit)\b', re.IGNORECASE), 2.0)
        ],
        QueryIntent.TAXATION_FISCAL: [
            (re.compile(r'\b(tax|gst|vat|income tax|duty|levy|tariff|fiscal|budget|audit)\b', re.IGNORECASE), 3.0),
            (re.compile(r'\b(deduction|exemption|return|refund)\b', re.IGNORECASE), 2.0)
        ],
        QueryIntent.JURISDICTION_AUTHORITY: [
            (re.compile(r'\b(ministry|department|bureau|council|commission|authority|agency|office)\b', re.IGNORECASE), 2.5),
            (re.compile(r'\b(who is responsible|contact person|minister|secretary)\b', re.IGNORECASE), 2.0)
        ],
        QueryIntent.TIMELINES_DEADLINES: [
            (re.compile(r'\b(deadline|last date|due date|expiry|validity|duration|processing time)\b', re.IGNORECASE), 3.0),
            (re.compile(r'\b(how long|when|schedule)\b', re.IGNORECASE), 1.5)
        ],
        QueryIntent.GRIEVANCE_REDRESSAL: [
            (re.compile(r'\b(complaint|grievance|appeal|dispute|ombudsman|redressal|customer care|helpline)\b', re.IGNORECASE), 3.0)
        ],
        QueryIntent.PUBLIC_SAFETY: [
            (re.compile(r'\b(emergency|disaster|police|fire|ambulance|flood|earthquake|cyude|safety|security)\b', re.IGNORECASE), 2.5)
        ]
    }


def build_intent_keywords() -> Dict[QueryIntent, List[Tuple[str, float]]]:
    """Build keyword lists for government intent classification"""
    return {
        QueryIntent.POLICY_INFORMATION: [
            ("what is", 1.5), ("explain", 1.5), ("about", 1.0), ("overview", 1.0),
            ("policy", 1.0), ("scheme", 1.0), ("program", 1.0)
        ],
        QueryIntent.PROCEDURAL_STEPS: [
            ("how to", 2.0), ("steps", 1.5), ("apply", 1.5), ("online", 1.0),
            ("register", 1.5), ("login", 1.0)
        ],
        QueryIntent.ELIGIBILITY_CRITERIA: [
            ("can i", 2.0), ("am i eligible", 2.5), ("limit", 1.0), ("criteria", 1.5)
        ],
        QueryIntent.TAXATION_FISCAL: [
            ("rate", 1.0), ("slab", 1.0), ("percentage", 1.0), ("calculate", 1.0)
        ]
    }


def build_contextual_rules() -> Dict[str, Any]:
    """Build contextual rules for government intent classification"""
    return {
        'subsidy_with_income': {
            'condition': lambda f: f['has_benefit_terms'] and f['has_financial_terms'],
            'boost_intent': QueryIntent.BENEFITS_SUBSIDIES,
            'boost_amount': 1.5
        },
        'tax_compliance': {
            'condition': lambda f: f['has_tax_terms'] and f['has_compliance_terms'],
            'boost_intent': QueryIntent.TAXATION_FISCAL,
            'boost_amount': 2.0
        },
        'application_documents': {
            'condition': lambda f: f['has_procedure_terms'] and f['has_document_terms'],
            'boost_intent': QueryIntent.DOCUMENT_REQUIREMENTS,
            'boost_amount': 2.0
        },
        'deadline_urgency': {
            'condition': lambda f: f['has_time_terms'] and f['has_procedure_terms'],
            'boost_intent': QueryIntent.TIMELINES_DEADLINES,
            'boost_amount': 1.5
        }
    }
