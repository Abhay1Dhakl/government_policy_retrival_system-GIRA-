"""
Text Processing Module for GIRA AI
Handles comprehensive text normalization and search variations for PDF matching.
"""

import re
import unicodedata
from typing import List, Set

class TextProcessor:
    """Handles text normalization and generation of search variations"""

    @staticmethod
    def normalize_pdf_text(text: str) -> str:
        """
        Normalize text to handle PDF formatting idiosyncrasies.
        """
        if not text:
            return text
            
        # Step 1: Normalize Unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Step 2: Handle hyphenated line breaks
        text = re.sub(r'-\s*\n\s*', '', text)
        text = re.sub(r'-\s*[\r\n]+\s*', '', text)
        
        # Step 3: Standardize dashes
        dash_chars = ['–', '—', '−', '‐', '‑', '‒', '―', '⁻', '﹣', '－']
        for dash in dash_chars:
            text = text.replace(dash, '-')
            
        # Step 4: Standardize quotes
        text = re.sub(r'[""''`´‚„‛‟‹›«»「」『』〝〞〟＂]', '"', text)
        text = re.sub(r'[''‛‚ʻʼʽˈˊˋ`´῾]', "'", text)
        
        # Step 5: Ligatures
        ligatures = {'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬀ': 'ff'}
        for ligature, replacement in ligatures.items():
            text = text.replace(ligature, replacement)
            
        # Step 6: Whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    @staticmethod
    def generate_search_variations(text_to_highlight: str) -> List[str]:
        """Generate comprehensive variations of the text for searching."""
        
        normalized = TextProcessor.normalize_pdf_text(text_to_highlight)
        search_texts = [text_to_highlight, normalized]
        
        base_texts = [text_to_highlight, normalized]
        
        for base in base_texts:
            # Punctuation variations
            search_texts.extend([
                base.replace('-', ''),
                base.replace('-', ' '),
                base.replace('_', ' '),
                re.sub(r'[^\w\s]', '', base), # Remove punctuation
                re.sub(r'\s+', ' ', base)     # Normalize space
            ])
            
            # Case variations
            search_texts.extend([
                base.lower(),
                base.upper(),
                base.title()
            ])
            
            # Government/Legal Term Variations (Swapping medical for government)
            # Replacing old "dose-dependent" etc with relevant policy terms
            
            gov_terms = {
                'sub-section': ['subsection', 'sub section', 'sub_section'],
                'co-applicant': ['coapplicant', 'co applicant', 'joint applicant'],
                'non-refundable': ['nonrefundable', 'non refundable'],
                'pre-requisite': ['prerequisite', 'pre requisite'],
                'post-dated': ['postdated', 'post dated'],
                're-verification': ['reverification', 're verification'],
                'cross-border': ['crossborder', 'cross border'],
                'anti-corruption': ['anticorruption', 'anti corruption'],
                'long-term': ['longterm', 'long term'],
                'short-term': ['shortterm', 'short term'],
                'full-time': ['fulltime', 'full time'],
                'part-time': ['parttime', 'part time'],
                'e-governance': ['egovernance', 'e governance'],
                'self-attested': ['selfattested', 'self attested']
            }
            
            for term, vars in gov_terms.items():
                if term in base.lower():
                    for v in vars:
                        # Case insensitive replacement check would be complex, 
                        # so we just append explicit variations if found
                        search_texts.append(base.lower().replace(term, v))
                        
            # Number/Date formatting
            # "Section 5(a)" -> "Section 5 (a)", "Section 5 - a"
            if re.search(r'\d', base):
                # Basic spacers
                search_texts.append(re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', base)) # 5a -> 5 a
                search_texts.append(re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', base)) # Section5 -> Section 5
                
        # Deduplicate
        unique = []
        seen = set()
        for t in search_texts:
            clean = t.strip()
            if clean and clean not in seen and len(clean) >= 3:
                unique.append(clean)
                seen.add(clean)
                
        return unique
