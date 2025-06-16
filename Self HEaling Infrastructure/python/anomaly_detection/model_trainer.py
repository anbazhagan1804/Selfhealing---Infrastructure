#!/usr/bin/env python3

import os
import sys
import json
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs', 'model_trainer.log'), mode='a')
    ]
)
logger = logging.getLogger('model_trainer')

# Global variables
config = {}
thresholds = {}

# Load configuration
def load_config() -> Dict:
    """Load the configuration from the settings.yaml file"""
    global config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'settings.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return {}

# Load thresholds
def load_thresholds() -> Dict:
    """Load the thresholds from the thresholds.yaml file"""
    global thresholds
    thresholds_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'thresholds.yaml')
    try:
        with open(thresholds_path, 'r') as f:
            thresholds = yaml.safe_load(f)
        logger.info(f"Thresholds loaded from {thresholds_path}")
        return thresholds
    except Exception as e:
        logger.error(f"Failed to load thresholds: {str(e)}")
        return {}

# Fetch historical data from Prometheus
def fetch_historical_data(days: int = 7) -> Dict[str, pd.DataFrame]:
    """Fetch historical data from Prometheus for model training"""
    logger.info(f"Fetching historical data for the last {days} days")
    
    # In a real implementation, this would query Prometheus for historical data
    # For this example, we'll simulate the data
    
    # Define the metrics we want to fetch
    metrics = [
        # Node metrics
        'node_cpu_seconds_total',
        'node_memory_MemAvailable_bytes',
        'node_filesystem_avail_bytes',
        'node_network_receive_bytes_total',
        'node_network_transmit_bytes_total',
        
        # Pod metrics
        'kube_pod_container_resource_usage_cpu_cores',
        'kube_pod_container_resource_usage_memory_bytes',
        'kube_pod_container_status_restarts_total',
        
        # Service metrics
        'http_request_duration_seconds',
        'http_requests_total',
        'http_requests_errors_total',
        
        # Custom application metrics
        'app_request_latency_seconds',
        'app_error_rate',
        'app_saturation_ratio'
    ]
    
    # Create a dictionary to hold the dataframes for each metric
    dataframes = {}
    
    # Generate simulated data for each metric
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    for metric in metrics:
        # Create a base dataframe with timestamps
        df = pd.DataFrame(index=timestamps)
        
        # Add columns for different instances/labels
        if 'node_' in metric:
            # Simulate data for multiple nodes
            for node_id in range(1, 6):  # 5 nodes
                node_name = f"node-{node_id}"
                
                # Generate normal data with some random noise
                if 'cpu' in metric:
                    # CPU usage follows a daily pattern
                    base = 0.6 + 0.2 * np.sin(np.linspace(0, days * 2 * np.pi, len(df)))
                    noise = np.random.normal(0, 0.05, len(df))
                    values = base + noise
                    # Add some anomalies
                    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
                    values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(1.5, 3, size=len(anomaly_indices))
                    
                elif 'memory' in metric:
                    # Memory usage is more stable but has occasional spikes
                    base = 0.7 + 0.05 * np.sin(np.linspace(0, days * 2 * np.pi, len(df)))
                    noise = np.random.normal(0, 0.03, len(df))
                    values = base + noise
                    # Add some anomalies
                    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
                    values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(1.3, 2, size=len(anomaly_indices))
                    
                elif 'filesystem' in metric:
                    # Filesystem usage gradually increases
                    base = np.linspace(0.5, 0.8, len(df))
                    noise = np.random.normal(0, 0.02, len(df))
                    values = base + noise
                    # Add some anomalies
                    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.005), replace=False)
                    values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(1.2, 1.5, size=len(anomaly_indices))
                    
                elif 'network' in metric:
                    # Network traffic has peaks during working hours
                    hour_of_day = df.index.hour
                    is_working_hour = ((hour_of_day >= 9) & (hour_of_day <= 17)).astype(float)
                    base = 0.3 + 0.5 * is_working_hour
                    noise = np.random.normal(0, 0.1, len(df))
                    values = base + noise
                    # Add some anomalies
                    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.03), replace=False)
                    values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(2, 5, size=len(anomaly_indices))
                    
                else:
                    # Generic pattern for other metrics
                    base = 0.5 + 0.1 * np.sin(np.linspace(0, days * 2 * np.pi, len(df)))
                    noise = np.random.normal(0, 0.05, len(df))
                    values = base + noise
                    
                # Ensure values are positive
                values = np.maximum(values, 0)
                
                # Add to dataframe
                df[node_name] = values
                
        elif 'pod_' in metric or 'kube_pod' in metric:
            # Simulate data for multiple pods
            for ns_id in range(1, 4):  # 3 namespaces
                namespace = f"namespace-{ns_id}"
                for pod_id in range(1, 6):  # 5 pods per namespace
                    pod_name = f"{namespace}/pod-{pod_id}"
                    
                    # Generate normal data with some random noise
                    if 'cpu' in metric:
                        base = 0.4 + 0.15 * np.sin(np.linspace(0, days * 2 * np.pi, len(df)))
                        noise = np.random.normal(0, 0.08, len(df))
                        values = base + noise
                        # Add some anomalies
                        anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
                        values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(1.5, 3, size=len(anomaly_indices))
                        
                    elif 'memory' in metric:
                        # Memory usage gradually increases for some pods (memory leaks)
                        if pod_id % 3 == 0:  # Every 3rd pod has a memory leak
                            base = np.linspace(0.4, 0.9, len(df))
                        else:
                            base = 0.5 + 0.05 * np.sin(np.linspace(0, days * 2 * np.pi, len(df)))
                        noise = np.random.normal(0, 0.05, len(df))
                        values = base + noise
                        # Add some anomalies
                        anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
                        values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(1.3, 2, size=len(anomaly_indices))
                        
                    elif 'restarts' in metric:
                        # Pod restarts are rare events
                        values = np.zeros(len(df))
                        # Add some restarts
                        restart_indices = np.random.choice(len(df), size=int(len(df) * 0.005), replace=False)
                        values[restart_indices] = 1
                        # Cumulative sum for total restarts
                        values = np.cumsum(values)
                        
                    else:
                        # Generic pattern for other metrics
                        base = 0.5 + 0.1 * np.sin(np.linspace(0, days * 2 * np.pi, len(df)))
                        noise = np.random.normal(0, 0.05, len(df))
                        values = base + noise
                        
                    # Ensure values are positive
                    values = np.maximum(values, 0)
                    
                    # Add to dataframe
                    df[pod_name] = values
                    
        elif 'http_' in metric or 'app_' in metric:
            # Simulate data for multiple services/applications
            for service_id in range(1, 4):  # 3 services
                service_name = f"service-{service_id}"
                
                # Generate normal data with some random noise
                if 'latency' in metric or 'duration' in metric:
                    # Latency has peaks during high traffic periods
                    hour_of_day = df.index.hour
                    is_peak_hour = ((hour_of_day >= 10) & (hour_of_day <= 14) | (hour_of_day >= 19) & (hour_of_day <= 22)).astype(float)
                    base = 0.2 + 0.3 * is_peak_hour
                    noise = np.random.normal(0, 0.1, len(df))
                    values = base + noise
                    # Add some anomalies
                    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.03), replace=False)
                    values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(2, 10, size=len(anomaly_indices))
                    
                elif 'error' in metric:
                    # Error rate is normally low
                    base = 0.02 + 0.01 * np.sin(np.linspace(0, days * 2 * np.pi, len(df)))
                    noise = np.random.normal(0, 0.005, len(df))
                    values = base + noise
                    # Add some anomalies (error spikes)
                    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
                    values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(5, 20, size=len(anomaly_indices))
                    
                elif 'requests_total' in metric:
                    # Request rate follows daily patterns
                    hour_of_day = df.index.hour
                    is_working_hour = ((hour_of_day >= 9) & (hour_of_day <= 17)).astype(float)
                    base = 0.3 + 0.6 * is_working_hour
                    noise = np.random.normal(0, 0.1, len(df))
                    values = base + noise
                    # Add some anomalies (traffic spikes)
                    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
                    values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(1.5, 3, size=len(anomaly_indices))
                    
                elif 'saturation' in metric:
                    # Saturation ratio increases with load
                    hour_of_day = df.index.hour
                    is_peak_hour = ((hour_of_day >= 10) & (hour_of_day <= 14) | (hour_of_day >= 19) & (hour_of_day <= 22)).astype(float)
                    base = 0.4 + 0.4 * is_peak_hour
                    noise = np.random.normal(0, 0.05, len(df))
                    values = base + noise
                    # Add some anomalies
                    anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
                    values[anomaly_indices] = values[anomaly_indices] * np.random.uniform(1.3, 1.8, size=len(anomaly_indices))
                    
                else:
                    # Generic pattern for other metrics
                    base = 0.5 + 0.1 * np.sin(np.linspace(0, days * 2 * np.pi, len(df)))
                    noise = np.random.normal(0, 0.05, len(df))
                    values = base + noise
                    
                # Ensure values are positive
                values = np.maximum(values, 0)
                
                # Add to dataframe
                df[service_name] = values
        
        # Store the dataframe
        dataframes[metric] = df
    
    logger.info(f"Generated simulated data for {len(metrics)} metrics")
    return dataframes

# Preprocess data for model training
def preprocess_data(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Preprocess the data for model training"""
    logger.info("Preprocessing data for model training")
    
    # Dictionary to store preprocessed data for each metric and entity
    preprocessed_data = {}
    
    for metric, df in dataframes.items():
        preprocessed_data[metric] = {}
        
        # Process each column (entity) in the dataframe
        for entity in df.columns:
            # Get the time series for this entity
            series = df[entity]
            
            # Create features from the time series
            features = create_features(series)
            
            # Store the preprocessed data
            preprocessed_data[metric][entity] = {
                'features': features,
                'series': series
            }
    
    logger.info(f"Preprocessed data for {len(preprocessed_data)} metrics")
    return preprocessed_data

# Create features from time series
def create_features(series: pd.Series) -> pd.DataFrame:
    """Create features from a time series for anomaly detection"""
    # Create a dataframe with the original values
    df = pd.DataFrame({'value': series})
    
    # Add time-based features
    df['hour'] = series.index.hour
    df['day_of_week'] = series.index.dayofweek
    df['is_weekend'] = (series.index.dayofweek >= 5).astype(int)
    
    # Add rolling statistics
    df['rolling_mean_1h'] = series.rolling(window=12).mean()  # 12 * 5min = 1h
    df['rolling_std_1h'] = series.rolling(window=12).std()
    df['rolling_mean_6h'] = series.rolling(window=72).mean()  # 72 * 5min = 6h
    df['rolling_std_6h'] = series.rolling(window=72).std()
    
    # Add lag features
    df['lag_1'] = series.shift(1)
    df['lag_2'] = series.shift(2)
    df['lag_3'] = series.shift(3)
    df['lag_6'] = series.shift(6)
    df['lag_12'] = series.shift(12)
    
    # Add difference features
    df['diff_1'] = series.diff(1)
    df['diff_2'] = series.diff(2)
    df['diff_3'] = series.diff(3)
    
    # Add rate of change
    df['rate_of_change_1'] = series.pct_change(1)
    df['rate_of_change_3'] = series.pct_change(3)
    df['rate_of_change_6'] = series.pct_change(6)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Train anomaly detection models
def train_models(preprocessed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Train anomaly detection models for each metric and entity"""
    logger.info("Training anomaly detection models")
    
    # Dictionary to store models and related information
    models = {}
    
    for metric, entities in preprocessed_data.items():
        models[metric] = {}
        
        for entity, data in entities.items():
            # Get features
            features_df = data['features']
            
            # Split features into X (feature columns) and timestamps
            X = features_df.drop('value', axis=1)  # Remove the original value column
            timestamps = features_df.index
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train an Isolation Forest model
            contamination = 0.02  # Assuming 2% of data points are anomalies
            model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            model.fit(X_scaled)
            
            # Predict anomalies
            y_pred = model.predict(X_scaled)
            # Convert to binary labels (1: normal, 0: anomaly)
            y_pred_binary = np.where(y_pred == 1, 0, 1)  # Isolation Forest returns 1 for normal, -1 for anomaly
            
            # Calculate anomaly scores
            scores = -model.score_samples(X_scaled)  # Higher score = more anomalous
            
            # Store the model and related information
            models[metric][entity] = {
                'model': model,
                'scaler': scaler,
                'timestamps': timestamps,
                'anomaly_labels': y_pred_binary,
                'anomaly_scores': scores,
                'threshold': np.percentile(scores, 98)  # Set threshold at 98th percentile of scores
            }
    
    logger.info(f"Trained models for {len(models)} metrics")
    return models

# Evaluate models
def evaluate_models(models: Dict[str, Dict[str, Any]], preprocessed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Evaluate the trained models"""
    logger.info("Evaluating trained models")
    
    # Dictionary to store evaluation metrics
    evaluation = {}
    
    for metric, entities in models.items():
        evaluation[metric] = {}
        
        for entity, model_data in entities.items():
            # Get the original time series
            series = preprocessed_data[metric][entity]['series']
            
            # Get anomaly scores and labels
            scores = model_data['anomaly_scores']
            labels = model_data['anomaly_labels']
            
            # Calculate metrics (in a real scenario, we would have ground truth labels)
            # For this example, we'll use the model's own predictions as "ground truth"
            # and evaluate how well the threshold-based detection works
            
            # Apply threshold to scores to get binary predictions
            threshold = model_data['threshold']
            predictions = (scores > threshold).astype(int)
            
            # Calculate precision, recall, and F1 score
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            
            # Store evaluation metrics
            evaluation[metric][entity] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'anomaly_rate': labels.mean(),
                'threshold': threshold
            }
    
    logger.info(f"Evaluated models for {len(evaluation)} metrics")
    return evaluation

# Save models
def save_models(models: Dict[str, Dict[str, Any]]) -> bool:
    """Save trained models to disk"""
    logger.info("Saving trained models")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Save each model
        for metric, entities in models.items():
            metric_dir = os.path.join(models_dir, metric)
            os.makedirs(metric_dir, exist_ok=True)
            
            for entity, model_data in entities.items():
                # Create a safe filename from the entity name
                safe_entity = entity.replace('/', '_').replace('\\', '_')
                model_path = os.path.join(metric_dir, f"{safe_entity}.pkl")
                
                # Save the model data
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
        
        logger.info(f"Saved models to {models_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to save models: {str(e)}")
        return False

# Main function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train anomaly detection models for SelfHealing Sentinel')
    parser.add_argument('--days', type=int, default=7, help='Number of days of historical data to use')
    parser.add_argument('--output-dir', type=str, help='Directory to save models')
    args = parser.parse_args()
    
    # Load configuration
    load_config()
    load_thresholds()
    
    # Fetch historical data
    dataframes = fetch_historical_data(days=args.days)
    
    # Preprocess data
    preprocessed_data = preprocess_data(dataframes)
    
    # Train models
    models = train_models(preprocessed_data)
    
    # Evaluate models
    evaluation = evaluate_models(models, preprocessed_data)
    
    # Print evaluation summary
    print("\nModel Evaluation Summary:")
    print("=========================\n")
    
    for metric, entities in evaluation.items():
        print(f"Metric: {metric}")
        for entity, metrics in entities.items():
            print(f"  Entity: {entity}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1_score']:.4f}")
            print(f"    Anomaly Rate: {metrics['anomaly_rate']:.4f}")
            print(f"    Threshold: {metrics['threshold']:.4f}")
        print()
    
    # Save models
    if save_models(models):
        print("\nModels saved successfully!")
    else:
        print("\nFailed to save models.")

# Entry point
if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    main()