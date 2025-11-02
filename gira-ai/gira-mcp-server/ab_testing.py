"""
A/B Testing infrastructure for MIRA AI retrieval system
Enables systematic comparison of different retrieval strategies
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Callable, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import statistics
import random
from dataclasses import dataclass, asdict
from evaluation import MedicalRetrievalEvaluator


@dataclass
class ExperimentResult:
    """Result of a single experiment run"""
    experiment_name: str
    variant_name: str
    query: str
    metrics: Dict[str, float]
    retrieved_docs: List[Dict]
    execution_time: float
    timestamp: str


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment"""
    experiment_name: str
    variant_a: str
    variant_b: str
    winner: Optional[str]
    confidence_level: float
    sample_size: int
    metrics_comparison: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, bool]


class ABTester:
    """A/B testing framework for retrieval strategies"""

    def __init__(self, evaluator: Optional[MedicalRetrievalEvaluator] = None,
                 results_dir: str = "ab_test_results"):
        self.evaluator = evaluator or MedicalRetrievalEvaluator()
        self.results_dir = results_dir
        self.active_experiments = {}
        self.completed_experiments = {}

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

    async def run_experiment(self, experiment_name: str, variants: Dict[str, Callable],
                           queries: List[str], ground_truth: Dict[str, List[str]],
                           sample_size: int = 30) -> ExperimentSummary:
        """
        Run A/B test comparing different retrieval variants

        Args:
            experiment_name: Name of the experiment
            variants: Dict of variant_name -> search_function
            queries: List of test queries
            ground_truth: Dict of query -> list of relevant doc IDs
            sample_size: Number of queries to test per variant

        Returns:
            ExperimentSummary with results and statistical analysis
        """

        print(f"🚀 Starting A/B experiment: {experiment_name}")
        print(f"📊 Testing {len(variants)} variants on {min(sample_size, len(queries))} queries")

        # Sample queries for this experiment
        test_queries = random.sample(queries, min(sample_size, len(queries)))

        # Store ground truth in evaluator
        for query, relevant_docs in ground_truth.items():
            self.evaluator.add_ground_truth(query, relevant_docs)

        # Run each variant on all test queries
        variant_results = {}
        all_results = []

        for variant_name, search_func in variants.items():
            print(f"🔍 Running variant: {variant_name}")
            variant_results[variant_name] = []

            for query in test_queries:
                try:
                    start_time = asyncio.get_event_loop().time()

                    # Execute search
                    result = await search_func(query)

                    execution_time = asyncio.get_event_loop().time() - start_time

                    # Evaluate results
                    retrieved_docs = result.get('matches', [])
                    metrics = self.evaluator.evaluate_medical_search(query, retrieved_docs)

                    # Store result
                    exp_result = ExperimentResult(
                        experiment_name=experiment_name,
                        variant_name=variant_name,
                        query=query,
                        metrics=metrics,
                        retrieved_docs=retrieved_docs,
                        execution_time=execution_time,
                        timestamp=datetime.now().isoformat()
                    )

                    variant_results[variant_name].append(exp_result)
                    all_results.append(exp_result)

                except Exception as e:
                    print(f"❌ Error in {variant_name} for query '{query}': {e}")
                    continue

        # Analyze results
        summary = self._analyze_experiment(experiment_name, variant_results, list(variants.keys()))

        # Save results
        self._save_experiment_results(experiment_name, all_results, summary)

        print(f"✅ Experiment {experiment_name} completed")
        print(f"🏆 Winner: {summary.winner} (confidence: {summary.confidence_level:.2f})")

        return summary

    def _analyze_experiment(self, experiment_name: str, variant_results: Dict[str, List[ExperimentResult]],
                          variant_names: List[str]) -> ExperimentSummary:
        """Analyze experiment results and determine winner"""

        if len(variant_names) != 2:
            raise ValueError("A/B testing currently supports exactly 2 variants")

        variant_a, variant_b = variant_names
        results_a = variant_results[variant_a]
        results_b = variant_results[variant_b]

        # Calculate average metrics for each variant
        metrics_a = self._calculate_average_metrics(results_a)
        metrics_b = self._calculate_average_metrics(results_b)

        # Perform statistical tests
        statistical_significance = {}
        winner = None
        confidence_level = 0.0

        key_metrics = ['f1', 'ndcg', 'precision', 'recall']

        for metric in key_metrics:
            if metric in metrics_a and metric in metrics_b:
                values_a = [r.metrics[metric] for r in results_a if metric in r.metrics]
                values_b = [r.metrics[metric] for r in results_b if metric in r.metrics]

                if len(values_a) >= 10 and len(values_b) >= 10:  # Minimum sample size for t-test
                    # Simple t-test approximation
                    mean_a, mean_b = statistics.mean(values_a), statistics.mean(values_b)
                    std_a, std_b = statistics.stdev(values_a), statistics.stdev(values_b)
                    n_a, n_b = len(values_a), len(values_b)

                    # Pooled standard error
                    se = ((std_a**2 / n_a) + (std_b**2 / n_b)) ** 0.5

                    if se > 0:
                        t_stat = abs(mean_a - mean_b) / se
                        # Approximate p-value (rough estimate)
                        p_value = 2 * (1 - self._normal_cdf(t_stat))

                        statistical_significance[metric] = p_value < 0.05  # 95% confidence

                        if p_value < 0.05 and mean_a > mean_b:
                            winner = variant_a
                            confidence_level = max(confidence_level, 1 - p_value)
                        elif p_value < 0.05 and mean_b > mean_a:
                            winner = variant_b
                            confidence_level = max(confidence_level, 1 - p_value)

        # Calculate metrics comparison
        metrics_comparison = {}
        for metric in key_metrics:
            if metric in metrics_a and metric in metrics_b:
                metrics_comparison[metric] = {
                    f'{variant_a}_mean': metrics_a[metric],
                    f'{variant_b}_mean': metrics_b[metric],
                    'difference': metrics_a[metric] - metrics_b[metric],
                    'relative_improvement': (
                        (metrics_a[metric] - metrics_b[metric]) / metrics_b[metric] * 100
                        if metrics_b[metric] != 0 else 0
                    )
                }

        return ExperimentSummary(
            experiment_name=experiment_name,
            variant_a=variant_a,
            variant_b=variant_b,
            winner=winner,
            confidence_level=confidence_level,
            sample_size=len(results_a) + len(results_b),
            metrics_comparison=metrics_comparison,
            statistical_significance=statistical_significance
        )

    def _calculate_average_metrics(self, results: List[ExperimentResult]) -> Dict[str, float]:
        """Calculate average metrics across results"""
        if not results:
            return {}

        metrics_sum = defaultdict(float)
        metrics_count = defaultdict(int)

        for result in results:
            for metric, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    metrics_sum[metric] += value
                    metrics_count[metric] += 1

        return {metric: metrics_sum[metric] / metrics_count[metric]
                for metric in metrics_sum}

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function"""
        # Abramowitz & Stegun approximation
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / (2**0.5)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp()

        return 0.5 * (1 + sign * y)

    def _save_experiment_results(self, experiment_name: str, results: List[ExperimentResult],
                               summary: ExperimentSummary):
        """Save experiment results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.results_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)

        # Save detailed results
        results_data = [asdict(result) for result in results]
        with open(os.path.join(experiment_dir, "detailed_results.json"), 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save summary
        with open(os.path.join(experiment_dir, "summary.json"), 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        # Save human-readable report
        self._generate_report(experiment_dir, summary, results)

    def _generate_report(self, experiment_dir: str, summary: ExperimentSummary,
                        results: List[ExperimentResult]):
        """Generate human-readable experiment report"""
        report_path = os.path.join(experiment_dir, "report.txt")

        with open(report_path, 'w') as f:
            f.write(f"A/B Testing Report: {summary.experiment_name}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Variants Compared: {summary.variant_a} vs {summary.variant_b}\n")
            f.write(f"Sample Size: {summary.sample_size} queries\n")
            f.write(f"Winner: {summary.winner or 'No clear winner'}\n")
            f.write(f"Confidence Level: {summary.confidence_level:.3f}\n\n")

            f.write("Metrics Comparison:\n")
            f.write("-" * 30 + "\n")
            for metric, comparison in summary.metrics_comparison.items():
                f.write(f"{metric.upper()}:\n")
                f.write(".3f")
                f.write(".3f")
                f.write(".3f")
                f.write(".1f")
                f.write(f"  Statistically Significant: {summary.statistical_significance.get(metric, False)}\n")
                f.write("\n")

            # Query-level analysis
            f.write("Query-Level Performance:\n")
            f.write("-" * 30 + "\n")

            query_performance = defaultdict(lambda: defaultdict(list))
            for result in results:
                for metric in ['f1', 'precision', 'recall']:
                    if metric in result.metrics:
                        query_performance[result.query][result.variant_name].append(result.metrics[metric])

            for query, variants in query_performance.items():
                f.write(f"Query: {query}\n")
                for variant, scores in variants.items():
                    avg_score = statistics.mean(scores) if scores else 0
                    f.write(".3f")
                f.write("\n")

    async def run_quick_comparison(self, variant_a: Callable, variant_b: Callable,
                                 queries: List[str], ground_truth: Dict[str, List[str]],
                                 experiment_name: str = "quick_test") -> Dict[str, Any]:
        """Run a quick A/B test with minimal configuration"""
        variants = {
            "variant_a": variant_a,
            "variant_b": variant_b
        }

        summary = await self.run_experiment(experiment_name, variants, queries, ground_truth, sample_size=10)

        return {
            "winner": summary.winner,
            "confidence": summary.confidence_level,
            "metrics_comparison": summary.metrics_comparison,
            "summary": asdict(summary)
        }


# Example usage functions
async def example_ab_test():
    """Example of how to use the A/B testing framework"""

    # Initialize tester
    tester = ABTester()

    # Define test queries and ground truth
    test_queries = [
        "azithromycin side effects",
        "amoxicillin dosage",
        "cardiac toxicity warnings",
        "pediatric dosing guidelines"
    ]

    ground_truth = {
        "azithromycin side effects": ["doc1", "doc2"],
        "amoxicillin dosage": ["doc3", "doc4"],
        "cardiac toxicity warnings": ["doc5", "doc6"],
        "pediatric dosing guidelines": ["doc7", "doc8"]
    }

    # Define variants (these would be your actual search functions)
    async def baseline_search(query: str) -> Dict[str, Any]:
        # Simulate baseline search
        await asyncio.sleep(0.1)  # Simulate search time
        return {
            "matches": [
                {"id": "doc1", "score": 0.8},
                {"id": "doc2", "score": 0.7}
            ]
        }

    async def improved_search(query: str) -> Dict[str, Any]:
        # Simulate improved search
        await asyncio.sleep(0.1)
        return {
            "matches": [
                {"id": "doc1", "score": 0.9},
                {"id": "doc2", "score": 0.8},
                {"id": "doc3", "score": 0.6}
            ]
        }

    variants = {
        "baseline": baseline_search,
        "improved": improved_search
    }

    # Run experiment
    results = await tester.run_experiment(
        "search_improvement_test",
        variants,
        test_queries,
        ground_truth,
        sample_size=4
    )

    print(f"Experiment completed. Winner: {results.winner}")
    return results


if __name__ == "__main__":
    # Run example
    asyncio.run(example_ab_test())