#!/usr/bin/env python3
"""
Script to evaluate recall and precision for your MIRA AI search results
"""

import json
from evaluation import MedicalRetrievalEvaluator

def evaluate_your_search_results():
    """Evaluate the search results you provided"""

    # Your search results from the MCP server response
    search_response = {
        "matches": [
            {"id": "XCA5ghXP7WME_1_1", "score": 0.9580339844198091, "document_type": "pis", "source": "3_24.pdf"},
            {"id": "XCA5ghXP7WME_15_1", "score": 0.9356154495239974, "document_type": "pis", "source": "3_24.pdf"},
            {"id": "XCA5ghXP7WME_15_0", "score": 0.9355461311531424, "document_type": "pis", "source": "3_24.pdf"},
            {"id": "XCA5ghXP7WME_12_2", "score": 0.8956590145413623, "document_type": "pis", "source": "3_24.pdf"},
            {"id": "XCA5ghXP7WME_6_0", "score": 0.8925627825760919, "document_type": "pis", "source": "3_24.pdf"},
            {"id": "XCA5ghXP7WME_9_0", "score": 0.8740343348160918, "document_type": "pis", "source": "3_24.pdf"},
            {"id": "XCA5ghXP7WME_3_0", "score": 0.8185376205469469, "document_type": "pis", "source": "3_24.pdf"},
            {"id": "XCA5ghXP7WME_12_1", "score": 0.7990993508402802, "document_type": "pis", "source": "3_24.pdf"}
        ],
        "query_processed": "azithromycin use in children"
    }

    # Initialize evaluator
    evaluator = MedicalRetrievalEvaluator()

    # Define ground truth - which documents are actually relevant to "azithromycin use in children"
    # You need to manually determine which of these documents contain information about
    # azithromycin use in children. For now, I'll assume some are relevant.
    # Replace this with your actual ground truth!

    # EXAMPLE GROUND TRUTH - Replace with actual relevant document IDs
    ground_truth_relevant_ids = {
        "XCA5ghXP7WME_6_0",   # Contains pediatric information
        "XCA5ghXP7WME_9_0",   # Contains pediatric use information
        "XCA5ghXP7WME_12_1",  # Contains pediatric pharmacokinetic data
        "XCA5ghXP7WME_12_2",  # Contains pediatric dosing information
    }

    # Add ground truth for this query
    evaluator.add_ground_truth(search_response["query_processed"], ground_truth_relevant_ids)

    # Evaluate the search results
    metrics = evaluator.evaluate_medical_search(
        query=search_response["query_processed"],
        retrieved_docs=search_response["matches"],
        ground_truth_ids=ground_truth_relevant_ids
    )

    # Print results
    print("🔍 SEARCH EVALUATION RESULTS")
    print("=" * 50)
    print(f"Query: {search_response['query_processed']}")
    print(f"Documents Retrieved: {metrics['retrieved_count']}")
    print(f"Relevant Documents (Ground Truth): {metrics['relevant_count']}")
    print()

    print("📊 CORE METRICS:")
    print(".2f")
    print(".2f")
    print(".2f")
    print()

    print("🎯 PRECISION AT K:")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print()

    print("🏆 RANKING METRICS:")
    print(".4f")
    print(".4f")
    print(".4f")
    print()

    print("📋 MEDICAL-SPECIFIC METRICS:")
    print(f"Medical Category: {metrics['medical_category']}")
    print(f"Section Relevance Score: {metrics['section_relevance_score']:.2f}")
    print(f"Document Type Distribution: {metrics['doc_type_distribution']}")

    return metrics

def create_ground_truth_template():
    """Create a template for defining ground truth data"""

    template = {
        "queries": {
            "azithromycin use in children": {
                "description": "Documents containing information about azithromycin use in pediatric patients",
                "relevant_document_ids": [
                    # Add the IDs of documents that actually contain relevant information
                    # about azithromycin use in children
                ],
                "partially_relevant_ids": [
                    # Documents with some relevant information
                ]
            },
            "azithromycin side effects": {
                "description": "Documents containing adverse reactions and side effects",
                "relevant_document_ids": [],
                "partially_relevant_ids": []
            }
        },
        "instructions": """
        To create ground truth:
        1. For each query, manually review the retrieved documents
        2. Determine which documents are relevant to the query
        3. Add their IDs to the relevant_document_ids list
        4. Save this as ground_truth.json
        5. Load it in your evaluator: evaluator.load_ground_truth('ground_truth.json')
        """
    }

    with open('ground_truth_template.json', 'w') as f:
        json.dump(template, f, indent=2)

    print("Created ground_truth_template.json - edit this file to define your ground truth data")

if __name__ == "__main__":
    print("Evaluating your search results...")
    metrics = evaluate_your_search_results()

    print("\n" + "="*50)
    print("GROUND TRUTH SETUP:")
    create_ground_truth_template()

    print("\n💡 NEXT STEPS:")
    print("1. Review each retrieved document manually")
    print("2. Determine which ones are actually relevant to 'azithromycin use in children'")
    print("3. Update the ground_truth_relevant_ids set in this script")
    print("4. Re-run to get accurate recall/precision metrics")
    print("5. Create ground_truth.json for systematic evaluation")