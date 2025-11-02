"""
Performance Monitoring and Alerting for MIRA AI
Continuously monitors retrieval performance and alerts on degradation
"""

import asyncio
import json
import os
import smtplib
import time
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import statistics

from evaluation import MedicalRetrievalEvaluator


class PerformanceMonitor:
    """Monitors retrieval performance and alerts on degradation"""

    def __init__(self, evaluator: Optional[MedicalRetrievalEvaluator] = None,
                 monitoring_dir: str = "performance_monitoring",
                 alert_thresholds: Optional[Dict[str, float]] = None):
        self.evaluator = evaluator or MedicalRetrievalEvaluator()
        self.monitoring_dir = monitoring_dir
        self.performance_history = deque(maxlen=1000)  # Keep last 1000 measurements

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'precision_drop': 0.1,  # 10% drop from baseline
            'recall_drop': 0.15,    # 15% drop from baseline
            'f1_drop': 0.12,        # 12% drop from baseline
            'latency_increase': 1.5,  # 50% increase in latency
            'error_rate_threshold': 0.05  # 5% error rate
        }

        # Alerting configuration
        self.alert_cooldown = timedelta(hours=1)  # Don't spam alerts
        self.last_alert_time = defaultdict(lambda: datetime.min)

        # Create monitoring directory
        os.makedirs(monitoring_dir, exist_ok=True)

        # Load baseline if exists
        self.baseline_metrics = self._load_baseline()

    def record_performance(self, query: str, retrieved_docs: List[Dict],
                          ground_truth_ids: Set[str], execution_time: float,
                          error: Optional[str] = None):
        """
        Record a performance measurement

        Args:
            query: The search query
            retrieved_docs: Retrieved documents
            ground_truth_ids: Ground truth relevant document IDs
            execution_time: Time taken for retrieval (seconds)
            error: Error message if retrieval failed
        """
        timestamp = datetime.now()

        if error:
            # Record error
            measurement = {
                'timestamp': timestamp.isoformat(),
                'query': query,
                'error': error,
                'execution_time': execution_time,
                'type': 'error'
            }
        else:
            # Evaluate performance
            metrics = self.evaluator.evaluate_search(query, retrieved_docs, ground_truth_ids)

            measurement = {
                'timestamp': timestamp.isoformat(),
                'query': query,
                'metrics': metrics,
                'execution_time': execution_time,
                'retrieved_count': len(retrieved_docs),
                'relevant_count': len(ground_truth_ids),
                'type': 'success'
            }

        self.performance_history.append(measurement)
        self._save_measurement(measurement)

        # Check for alerts
        self._check_alerts(measurement)

    def _save_measurement(self, measurement: Dict[str, Any]):
        """Save measurement to daily log file"""
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.monitoring_dir, f"performance_{date_str}.jsonl")

        with open(log_file, 'a') as f:
            f.write(json.dumps(measurement) + '\n')

    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline performance metrics"""
        baseline_file = os.path.join(self.monitoring_dir, "baseline.json")
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load baseline: {e}")
        return None

    def set_baseline(self, baseline_metrics: Dict[str, float]):
        """Set baseline performance metrics"""
        self.baseline_metrics = baseline_metrics.copy()
        baseline_file = os.path.join(self.monitoring_dir, "baseline.json")

        with open(baseline_file, 'w') as f:
            json.dump({
                'metrics': baseline_metrics,
                'set_date': datetime.now().isoformat(),
                'description': 'Baseline performance metrics'
            }, f, indent=2)

        print(f"✅ Baseline metrics set: {baseline_metrics}")

    def establish_baseline(self, days: int = 7):
        """Establish baseline from recent performance history"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_measurements = [
            m for m in self.performance_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_date and m['type'] == 'success'
        ]

        if len(recent_measurements) < 10:
            print(f"⚠️ Not enough measurements for baseline ({len(recent_measurements)} < 10)")
            return

        # Calculate average metrics
        metrics_data = defaultdict(list)
        for measurement in recent_measurements:
            for metric, value in measurement['metrics'].items():
                if isinstance(value, (int, float)):
                    metrics_data[metric].append(value)

        baseline_metrics = {}
        for metric, values in metrics_data.items():
            if values:
                baseline_metrics[metric] = statistics.mean(values)

        self.set_baseline(baseline_metrics)
        print(f"✅ Established baseline from {len(recent_measurements)} measurements")

    def _check_alerts(self, measurement: Dict[str, Any]):
        """Check if current measurement triggers any alerts"""
        if not self.baseline_metrics or measurement['type'] == 'error':
            return

        alerts = []

        # Check performance degradation
        current_metrics = measurement['metrics']
        for metric in ['precision', 'recall', 'f1']:
            if metric in current_metrics and metric in self.baseline_metrics:
                current_value = current_metrics[metric]
                baseline_value = self.baseline_metrics[metric]
                threshold_key = f'{metric}_drop'

                if threshold_key in self.alert_thresholds:
                    threshold = self.alert_thresholds[threshold_key]
                    if current_value < baseline_value * (1 - threshold):
                        alerts.append({
                            'type': 'performance_degradation',
                            'metric': metric,
                            'current_value': current_value,
                            'baseline_value': baseline_value,
                            'drop_percentage': ((baseline_value - current_value) / baseline_value) * 100,
                            'threshold': threshold * 100
                        })

        # Check latency increase
        if 'execution_time' in measurement:
            current_latency = measurement['execution_time']
            if 'avg_latency' in self.baseline_metrics:
                baseline_latency = self.baseline_metrics['avg_latency']
                threshold = self.alert_thresholds.get('latency_increase', 1.5)
                if current_latency > baseline_latency * threshold:
                    alerts.append({
                        'type': 'latency_increase',
                        'current_latency': current_latency,
                        'baseline_latency': baseline_latency,
                        'increase_factor': current_latency / baseline_latency,
                        'threshold': threshold
                    })

        # Check error rate
        recent_measurements = list(self.performance_history)[-100:]  # Last 100 measurements
        error_count = sum(1 for m in recent_measurements if m['type'] == 'error')
        error_rate = error_count / len(recent_measurements) if recent_measurements else 0

        if error_rate > self.alert_thresholds.get('error_rate_threshold', 0.05):
            alerts.append({
                'type': 'high_error_rate',
                'error_rate': error_rate,
                'threshold': self.alert_thresholds['error_rate_threshold'],
                'recent_measurements': len(recent_measurements)
            })

        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert, measurement)

    def _trigger_alert(self, alert: Dict[str, Any], measurement: Dict[str, Any]):
        """Trigger an alert (email, log, etc.)"""
        alert_type = alert['type']
        now = datetime.now()

        # Check cooldown
        if now - self.last_alert_time[alert_type] < self.alert_cooldown:
            return

        self.last_alert_time[alert_type] = now

        # Log alert
        alert_message = self._format_alert_message(alert, measurement)
        self._log_alert(alert_message)

        # Send email alert (if configured)
        self._send_email_alert(alert_message)

        print(f"🚨 ALERT: {alert_message}")

    def _format_alert_message(self, alert: Dict[str, Any], measurement: Dict[str, Any]) -> str:
        """Format alert message for notification"""
        alert_type = alert['type']

        if alert_type == 'performance_degradation':
            return (f"Performance degradation detected: {alert['metric']} dropped by "
                   f"{alert['drop_percentage']:.1f}% (current: {alert['current_value']:.3f}, "
                   f"baseline: {alert['baseline_value']:.3f}). Query: {measurement['query']}")

        elif alert_type == 'latency_increase':
            return (f"Latency increase detected: {alert['increase_factor']:.2f}x slower "
                   f"(current: {alert['current_latency']:.2f}s, "
                   f"baseline: {alert['baseline_latency']:.2f}s). Query: {measurement['query']}")

        elif alert_type == 'high_error_rate':
            return (f"High error rate detected: {alert['error_rate']:.1f}% "
                   f"(threshold: {alert['threshold']:.1f}%, "
                   f"based on {alert['recent_measurements']} measurements)")

        return f"Unknown alert type: {alert_type}"

    def _log_alert(self, message: str):
        """Log alert to file"""
        alert_file = os.path.join(self.monitoring_dir, "alerts.log")
        timestamp = datetime.now().isoformat()

        with open(alert_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def _send_email_alert(self, message: str):
        """Send email alert (placeholder - implement based on your email setup)"""
        # This is a placeholder - implement actual email sending based on your infrastructure
        email_config = os.getenv('MIRA_ALERT_EMAIL_CONFIG')
        if not email_config:
            return  # Email not configured

        try:
            # Parse email config (you would set this as environment variable)
            config = json.loads(email_config)
            smtp_server = config.get('smtp_server')
            smtp_port = config.get('smtp_port', 587)
            sender_email = config.get('sender_email')
            sender_password = config.get('sender_password')
            recipient_emails = config.get('recipient_emails', [])

            if not all([smtp_server, sender_email, sender_password, recipient_emails]):
                return

            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipient_emails)
            msg['Subject'] = 'MIRA AI Performance Alert'

            body = f"""
MIRA AI Performance Monitoring Alert

{message}

Time: {datetime.now().isoformat()}
System: MIRA AI Retrieval Engine

Please investigate the issue promptly.
            """
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_emails, text)
            server.quit()

            print("📧 Alert email sent successfully")

        except Exception as e:
            print(f"❌ Failed to send alert email: {e}")

    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_measurements = [
            m for m in self.performance_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]

        if not recent_measurements:
            return {'error': 'No measurements available'}

        # Separate successes and errors
        successes = [m for m in recent_measurements if m['type'] == 'success']
        errors = [m for m in recent_measurements if m['type'] == 'error']

        report = {
            'period': f"{cutoff_date.date()} to {datetime.now().date()}",
            'total_measurements': len(recent_measurements),
            'successful_queries': len(successes),
            'failed_queries': len(errors),
            'error_rate': len(errors) / len(recent_measurements) if recent_measurements else 0
        }

        if successes:
            # Calculate metric statistics
            metrics_data = defaultdict(list)
            latencies = []

            for measurement in successes:
                latencies.append(measurement['execution_time'])
                for metric, value in measurement['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics_data[metric].append(value)

            for metric, values in metrics_data.items():
                report[f'{metric}_mean'] = round(statistics.mean(values), 3)
                report[f'{metric}_std'] = round(statistics.stdev(values), 3) if len(values) > 1 else 0
                report[f'{metric}_min'] = round(min(values), 3)
                report[f'{metric}_max'] = round(max(values), 3)

            report['latency_mean'] = round(statistics.mean(latencies), 3)
            report['latency_p95'] = round(sorted(latencies)[int(len(latencies) * 0.95)], 3)

        # Compare to baseline
        if self.baseline_metrics:
            report['baseline_comparison'] = {}
            for metric in ['precision', 'recall', 'f1']:
                if f'{metric}_mean' in report and metric in self.baseline_metrics:
                    current = report[f'{metric}_mean']
                    baseline = self.baseline_metrics[metric]
                    report['baseline_comparison'][f'{metric}_change'] = round(
                        ((current - baseline) / baseline) * 100, 1
                    )

        return report

    def export_report(self, filepath: str, days: int = 7):
        """Export performance report to file"""
        report = self.get_performance_report(days)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 Performance report exported to {filepath}")

    async def continuous_monitoring(self, check_interval: int = 300):
        """
        Run continuous monitoring (call this in a background task)

        Args:
            check_interval: Seconds between performance checks
        """
        print("🔍 Starting continuous performance monitoring...")

        while True:
            try:
                # Generate and log periodic report
                report = self.get_performance_report(days=1)

                if report.get('total_measurements', 0) > 0:
                    print(f"📈 Performance check: {report['successful_queries']}/{report['total_measurements']} successful "
                          f"(error rate: {report['error_rate']:.1f}%)")

                    # Export daily report
                    date_str = datetime.now().strftime("%Y%m%d")
                    self.export_report(os.path.join(self.monitoring_dir, f"daily_report_{date_str}.json"))

                await asyncio.sleep(check_interval)

            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(check_interval)


# Convenience functions
def monitor_performance(query: str, retrieved_docs: List[Dict],
                       ground_truth_ids: Set[str], execution_time: float,
                       error: Optional[str] = None):
    """Convenience function for recording performance"""
    monitor = PerformanceMonitor()
    monitor.record_performance(query, retrieved_docs, ground_truth_ids, execution_time, error)


def get_performance_report(days: int = 7) -> Dict[str, Any]:
    """Get performance report"""
    monitor = PerformanceMonitor()
    return monitor.get_performance_report(days)


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()

    # Simulate some performance measurements
    monitor.record_performance(
        "azithromycin side effects",
        [{"id": "doc1", "score": 0.8}, {"id": "doc2", "score": 0.7}],
        {"doc1"},
        0.15
    )

    monitor.record_performance(
        "amoxicillin dosage",
        [{"id": "doc3", "score": 0.9}],
        {"doc3", "doc4"},  # doc4 not retrieved
        0.12
    )

    # Set baseline
    monitor.establish_baseline(days=1)

    # Get report
    report = monitor.get_performance_report()
    print("Performance Report:", json.dumps(report, indent=2))