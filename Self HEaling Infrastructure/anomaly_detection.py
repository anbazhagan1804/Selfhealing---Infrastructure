#!/usr/bin/env python3
"""
Anomaly Detection Module for SelfHealing Sentinel

This module implements machine learning based anomaly detection for infrastructure metrics.
It uses isolation forest algorithm to detect outliers in time series data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('anomaly_detector')


class MetricAnomalyDetector:
    """Anomaly detector for infrastructure metrics using Isolation Forest algorithm."""
    
    def __init__(self, config_path: str = 'config/thresholds.yaml'):
        """Initialize the anomaly detector with configuration.
        
        Args:
            config_path: Path to the configuration file containing thresholds
        """
        self.config_path = config_path
        self.models: Dict[str, IsolationForest] = {}
        self.thresholds: Dict[str, float] = {}
        self.training_data: Dict[str, pd.DataFrame] = {}
        
        # Load configuration
        self._load_config()
        
        logger.info("Anomaly detector initialized")
    
    def _load_config(self) -> None:
        """Load configuration from the config file."""
        try:
            # In a real implementation, this would load from a YAML file
            # For this example, we'll use hardcoded values
            self.thresholds = {
                'cpu_usage': 0.95,  # 95% CPU usage threshold
                'memory_usage': 0.90,  # 90% memory usage threshold
                'disk_usage': 0.85,  # 85% disk usage threshold
                'latency': 500,  # 500ms latency threshold
                'error_rate': 0.05  # 5% error rate threshold
            }
            logger.info(f"Loaded thresholds: {self.thresholds}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def train(self, metric_name: str, data: pd.DataFrame) -> None:
        """Train the anomaly detection model for a specific metric.
        
        Args:
            metric_name: Name of the metric (e.g., 'cpu_usage')
            data: DataFrame containing the metric data with timestamp and value columns
        """
        logger.info(f"Training model for metric: {metric_name}")
        
        # Store the training data
        self.training_data[metric_name] = data
        
        # Create and train the model
        model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=0.05,  # Expected proportion of anomalies
            random_state=42
        )
        
        # Fit the model on the values
        model.fit(data[['value']])
        
        # Store the trained model
        self.models[metric_name] = model
        
        logger.info(f"Model for {metric_name} trained successfully")
    
    def detect_anomalies(self, metric_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the given metric data.
        
        Args:
            metric_name: Name of the metric
            data: DataFrame containing the metric data with timestamp and value columns
            
        Returns:
            DataFrame with anomaly scores and binary anomaly flags
        """
        if metric_name not in self.models:
            logger.error(f"No trained model found for metric: {metric_name}")
            raise ValueError(f"No trained model found for metric: {metric_name}")
        
        logger.info(f"Detecting anomalies for metric: {metric_name}")
        
        # Get the model
        model = self.models[metric_name]
        
        # Predict anomaly scores (-1 for anomalies, 1 for normal)
        scores = model.decision_function(data[['value']])
        predictions = model.predict(data[['value']])  # -1 for anomalies, 1 for normal
        
        # Add scores and predictions to the data
        result = data.copy()
        result['anomaly_score'] = scores
        result['is_anomaly'] = predictions == -1  # Convert to boolean
        
        # Apply threshold-based detection as well
        threshold = self.thresholds.get(metric_name)
        if threshold is not None:
            result['threshold_exceeded'] = data['value'] > threshold
            result['is_anomaly'] = result['is_anomaly'] | result['threshold_exceeded']
        
        # Count anomalies
        anomaly_count = result['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies in {len(data)} data points for {metric_name}")
        
        return result
    
    def generate_alert(self, metric_name: str, anomalies: pd.DataFrame) -> Dict:
        """Generate an alert for detected anomalies.
        
        Args:
            metric_name: Name of the metric
            anomalies: DataFrame containing the anomalies
            
        Returns:
            Alert data as a dictionary
        """
        if anomalies.empty or not anomalies['is_anomaly'].any():
            return {}
        
        # Filter only anomalies
        anomaly_data = anomalies[anomalies['is_anomaly']]
        
        # Create alert data
        alert = {
            'metric': metric_name,
            'timestamp': time.time(),
            'anomaly_count': len(anomaly_data),
            'severity': 'critical' if len(anomaly_data) > 5 else 'warning',
            'values': anomaly_data['value'].tolist(),
            'timestamps': anomaly_data['timestamp'].tolist()
        }
        
        logger.info(f"Generated alert for {metric_name}: {alert['severity']} severity with {alert['anomaly_count']} anomalies")
        
        return alert


def simulate_metric_data(metric_name: str, num_points: int = 100) -> pd.DataFrame:
    """Simulate metric data for testing.
    
    Args:
        metric_name: Name of the metric to simulate
        num_points: Number of data points to generate
        
    Returns:
        DataFrame with simulated metric data
    """
    timestamps = pd.date_range(start='2023-01-01', periods=num_points, freq='5min')
    
    # Generate normal values based on the metric type
    if metric_name == 'cpu_usage':
        # CPU usage between 10% and 70% with some noise
        values = np.random.uniform(0.1, 0.7, num_points)
        # Add some anomalies (spikes)
        anomaly_indices = np.random.choice(num_points, size=5, replace=False)
        values[anomaly_indices] = np.random.uniform(0.95, 1.0, 5)
    elif metric_name == 'memory_usage':
        # Memory usage between 20% and 60% with some noise
        values = np.random.uniform(0.2, 0.6, num_points)
        # Add some anomalies (spikes)
        anomaly_indices = np.random.choice(num_points, size=5, replace=False)
        values[anomaly_indices] = np.random.uniform(0.92, 0.98, 5)
    else:
        # Generic metric with values between 0 and 100
        values = np.random.normal(50, 15, num_points)
        # Add some anomalies (outliers)
        anomaly_indices = np.random.choice(num_points, size=5, replace=False)
        values[anomaly_indices] = np.random.uniform(150, 200, 5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    return df


def main():
    """Main function to demonstrate the anomaly detection."""
    # Create detector
    detector = MetricAnomalyDetector()
    
    # Simulate data for CPU usage
    cpu_data = simulate_metric_data('cpu_usage', 200)
    
    # Train the model
    detector.train('cpu_usage', cpu_data.iloc[:100])  # Use first 100 points for training
    
    # Detect anomalies in the test data
    test_data = cpu_data.iloc[100:]  # Use remaining points for testing
    anomalies = detector.detect_anomalies('cpu_usage', test_data)
    
    # Generate alert if anomalies are detected
    alert = detector.generate_alert('cpu_usage', anomalies)
    if alert:
        print(json.dumps(alert, indent=2))
    else:
        print("No anomalies detected")


if __name__ == "__main__":
    main()