#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentinel API - RESTful API for the SelfHealing Sentinel system

This module provides a FastAPI-based REST API for monitoring and controlling
the SelfHealing Sentinel system. It allows users to view system status,
manage remediation actions, and configure the system.
"""

import os
import sys
import json
import yaml
import logging
import datetime
from typing import Dict, List, Optional, Any, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
    from pydantic import BaseModel, Field
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    from starlette.responses import RedirectResponse
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install required packages: pip install fastapi uvicorn pydantic prometheus-client")
    sys.exit(1)

# Import local modules
try:
    from remediation.orchestrator import RemediationOrchestrator
    from anomaly.detector import AnomalyDetector
    from config.loader import ConfigLoader
except ImportError:
    # For development/testing without the full system
    print("Warning: Running in standalone mode without full system components")
    RemediationOrchestrator = None
    AnomalyDetector = None
    ConfigLoader = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log")
    ]
)
logger = logging.getLogger("sentinel-api")

# Initialize FastAPI app
app = FastAPI(
    title="SelfHealing Sentinel API",
    description="API for monitoring and controlling the SelfHealing Sentinel system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus metrics
REQUEST_COUNT = Counter(
    'sentinel_api_requests_total',
    'Total number of requests received by the API',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'sentinel_api_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)
ACTIVE_REMEDIATIONS = Gauge(
    'sentinel_active_remediations',
    'Number of active remediation actions'
)
ANOMALY_COUNT = Counter(
    'sentinel_anomalies_detected_total',
    'Total number of anomalies detected',
    ['severity', 'category']
)

# Define Pydantic models for API
class HealthStatus(BaseModel):
    status: str = Field(..., description="System health status")
    version: str = Field(..., description="System version")
    components: Dict[str, str] = Field(..., description="Component statuses")
    uptime: int = Field(..., description="System uptime in seconds")

class Anomaly(BaseModel):
    id: str = Field(..., description="Unique identifier for the anomaly")
    timestamp: datetime.datetime = Field(..., description="Time when the anomaly was detected")
    source: str = Field(..., description="Source of the anomaly (component, node, etc.)")
    metric: str = Field(..., description="Metric that triggered the anomaly")
    value: float = Field(..., description="Value that triggered the anomaly")
    threshold: float = Field(..., description="Threshold that was exceeded")
    severity: str = Field(..., description="Severity of the anomaly (low, medium, high, critical)")
    description: str = Field(..., description="Description of the anomaly")
    status: str = Field(..., description="Current status (active, resolved, in_remediation)")

class RemediationAction(BaseModel):
    id: str = Field(..., description="Unique identifier for the remediation action")
    anomaly_id: str = Field(..., description="ID of the anomaly that triggered this action")
    action_type: str = Field(..., description="Type of remediation action")
    target: str = Field(..., description="Target of the remediation action")
    status: str = Field(..., description="Current status (pending, in_progress, completed, failed)")
    start_time: Optional[datetime.datetime] = Field(None, description="Time when the action started")
    end_time: Optional[datetime.datetime] = Field(None, description="Time when the action completed")
    result: Optional[str] = Field(None, description="Result of the action")
    error: Optional[str] = Field(None, description="Error message if the action failed")

class RemediationRequest(BaseModel):
    action_type: str = Field(..., description="Type of remediation action to perform")
    target: str = Field(..., description="Target of the remediation action")
    parameters: Optional[Dict[str, Any]] = Field({}, description="Additional parameters for the action")

class ConfigurationUpdate(BaseModel):
    path: str = Field(..., description="Path to the configuration section to update")
    value: Any = Field(..., description="New value for the configuration")

class SystemStats(BaseModel):
    cpu_usage: float = Field(..., description="Current CPU usage percentage")
    memory_usage: float = Field(..., description="Current memory usage percentage")
    disk_usage: float = Field(..., description="Current disk usage percentage")
    network_in: float = Field(..., description="Current network inbound traffic (bytes/sec)")
    network_out: float = Field(..., description="Current network outbound traffic (bytes/sec)")
    active_anomalies: int = Field(..., description="Number of active anomalies")
    active_remediations: int = Field(..., description="Number of active remediation actions")
    total_anomalies_24h: int = Field(..., description="Total anomalies in the last 24 hours")
    total_remediations_24h: int = Field(..., description="Total remediation actions in the last 24 hours")
    success_rate_24h: float = Field(..., description="Success rate of remediation actions in the last 24 hours")

# Global variables
start_time = datetime.datetime.now()
config = {}
remediation_orchestrator = None
anomaly_detector = None

# Simulated data for development/testing
simulated_anomalies = []
simulated_actions = []
simulated_stats = {
    "cpu_usage": 35.2,
    "memory_usage": 42.7,
    "disk_usage": 68.3,
    "network_in": 1024.5,
    "network_out": 512.8,
    "active_anomalies": 2,
    "active_remediations": 1,
    "total_anomalies_24h": 15,
    "total_remediations_24h": 12,
    "success_rate_24h": 91.7
}

# Helper functions
def load_config():
    """Load configuration from file"""
    global config
    try:
        config_path = os.environ.get("SENTINEL_CONFIG", "config/settings.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Load default config for development/testing
        config = {
            "system": {"name": "SelfHealing Sentinel", "version": "1.0.0"},
            "api": {"port": 8080},
            "monitoring": {"prometheus": {"url": "http://localhost:9090"}},
            "remediation": {"enabled": True}
        }
        return config

def initialize_components():
    """Initialize system components"""
    global remediation_orchestrator, anomaly_detector
    
    if RemediationOrchestrator is not None:
        remediation_orchestrator = RemediationOrchestrator(config)
        logger.info("Initialized remediation orchestrator")
    
    if AnomalyDetector is not None:
        anomaly_detector = AnomalyDetector(config)
        logger.info("Initialized anomaly detector")

# Middleware for request metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = datetime.datetime.now()
    response = await call_next(request)
    process_time = (datetime.datetime.now() - start_time).total_seconds()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

# Routes
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    """Check the health status of the system"""
    uptime = (datetime.datetime.now() - start_time).total_seconds()
    
    # Check component statuses
    components = {
        "api": "healthy",
        "remediation_orchestrator": "healthy" if remediation_orchestrator else "not_initialized",
        "anomaly_detector": "healthy" if anomaly_detector else "not_initialized",
    }
    
    # Add additional component checks here
    try:
        # Check Prometheus connection
        if "monitoring" in config and "prometheus" in config["monitoring"]:
            import requests
            prometheus_url = config["monitoring"]["prometheus"]["url"]
            response = requests.get(f"{prometheus_url}/-/healthy", timeout=2)
            if response.status_code == 200:
                components["prometheus"] = "healthy"
            else:
                components["prometheus"] = "unhealthy"
    except Exception as e:
        logger.warning(f"Failed to check Prometheus health: {e}")
        components["prometheus"] = "unknown"
    
    return {
        "status": "healthy",  # Could be "healthy", "degraded", or "unhealthy"
        "version": config.get("system", {}).get("version", "1.0.0"),
        "components": components,
        "uptime": int(uptime)
    }

@app.get("/metrics", tags=["System"])
async def metrics():
    """Expose Prometheus metrics"""
    return prometheus_client.generate_latest()

@app.get("/stats", response_model=SystemStats, tags=["System"])
async def get_system_stats():
    """Get current system statistics"""
    # In a real implementation, this would collect actual system stats
    # For now, return simulated data
    return simulated_stats

@app.get("/anomalies", response_model=List[Anomaly], tags=["Anomalies"])
async def get_anomalies(
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(100, description="Maximum number of anomalies to return")
):
    """Get list of detected anomalies with optional filtering"""
    if anomaly_detector:
        # In a real implementation, this would get anomalies from the detector
        anomalies = anomaly_detector.get_anomalies(status=status, severity=severity, source=source, limit=limit)
    else:
        # Return simulated data for development/testing
        anomalies = simulated_anomalies
        
        # Apply filters
        if status:
            anomalies = [a for a in anomalies if a.status == status]
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        if source:
            anomalies = [a for a in anomalies if a.source == source]
        
        # Apply limit
        anomalies = anomalies[:limit]
    
    return anomalies

@app.get("/anomalies/{anomaly_id}", response_model=Anomaly, tags=["Anomalies"])
async def get_anomaly(anomaly_id: str = Path(..., description="ID of the anomaly to retrieve")):
    """Get details of a specific anomaly"""
    if anomaly_detector:
        # In a real implementation, this would get the anomaly from the detector
        anomaly = anomaly_detector.get_anomaly(anomaly_id)
        if not anomaly:
            raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")
    else:
        # Return simulated data for development/testing
        anomaly = next((a for a in simulated_anomalies if a.id == anomaly_id), None)
        if not anomaly:
            raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")
    
    return anomaly

@app.get("/remediations", response_model=List[RemediationAction], tags=["Remediation"])
async def get_remediation_actions(
    status: Optional[str] = Query(None, description="Filter by status"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    limit: int = Query(100, description="Maximum number of actions to return")
):
    """Get list of remediation actions with optional filtering"""
    if remediation_orchestrator:
        # In a real implementation, this would get actions from the orchestrator
        actions = remediation_orchestrator.get_actions(status=status, action_type=action_type, limit=limit)
    else:
        # Return simulated data for development/testing
        actions = simulated_actions
        
        # Apply filters
        if status:
            actions = [a for a in actions if a.status == status]
        if action_type:
            actions = [a for a in actions if a.action_type == action_type]
        
        # Apply limit
        actions = actions[:limit]
    
    return actions

@app.get("/remediations/{action_id}", response_model=RemediationAction, tags=["Remediation"])
async def get_remediation_action(action_id: str = Path(..., description="ID of the remediation action to retrieve")):
    """Get details of a specific remediation action"""
    if remediation_orchestrator:
        # In a real implementation, this would get the action from the orchestrator
        action = remediation_orchestrator.get_action(action_id)
        if not action:
            raise HTTPException(status_code=404, detail=f"Remediation action {action_id} not found")
    else:
        # Return simulated data for development/testing
        action = next((a for a in simulated_actions if a.id == action_id), None)
        if not action:
            raise HTTPException(status_code=404, detail=f"Remediation action {action_id} not found")
    
    return action

@app.post("/remediations", response_model=RemediationAction, status_code=status.HTTP_201_CREATED, tags=["Remediation"])
async def create_remediation_action(request: RemediationRequest = Body(...)):
    """Create a new remediation action"""
    if not config.get("remediation", {}).get("enabled", True):
        raise HTTPException(status_code=403, detail="Remediation is disabled in the configuration")
    
    if remediation_orchestrator:
        # In a real implementation, this would create an action in the orchestrator
        try:
            action = remediation_orchestrator.create_action(
                action_type=request.action_type,
                target=request.target,
                parameters=request.parameters
            )
        except Exception as e:
            logger.error(f"Failed to create remediation action: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Create simulated action for development/testing
        import uuid
        action_id = str(uuid.uuid4())
        action = RemediationAction(
            id=action_id,
            anomaly_id="",  # No associated anomaly for manual actions
            action_type=request.action_type,
            target=request.target,
            status="pending",
            start_time=None,
            end_time=None,
            result=None,
            error=None
        )
        simulated_actions.append(action)
        logger.info(f"Created simulated remediation action: {action_id}")
    
    return action

@app.post("/remediations/{action_id}/cancel", response_model=RemediationAction, tags=["Remediation"])
async def cancel_remediation_action(action_id: str = Path(..., description="ID of the remediation action to cancel")):
    """Cancel a pending or in-progress remediation action"""
    if remediation_orchestrator:
        # In a real implementation, this would cancel the action in the orchestrator
        try:
            action = remediation_orchestrator.cancel_action(action_id)
            if not action:
                raise HTTPException(status_code=404, detail=f"Remediation action {action_id} not found")
        except Exception as e:
            logger.error(f"Failed to cancel remediation action: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Update simulated action for development/testing
        action = next((a for a in simulated_actions if a.id == action_id), None)
        if not action:
            raise HTTPException(status_code=404, detail=f"Remediation action {action_id} not found")
        
        if action.status not in ["pending", "in_progress"]:
            raise HTTPException(status_code=400, detail=f"Cannot cancel action with status {action.status}")
        
        action.status = "cancelled"
        action.end_time = datetime.datetime.now()
        logger.info(f"Cancelled simulated remediation action: {action_id}")
    
    return action

@app.get("/config", tags=["Configuration"])
async def get_configuration():
    """Get the current system configuration"""
    # Return a sanitized version of the configuration (remove sensitive fields)
    sanitized_config = json.loads(json.dumps(config))  # Deep copy
    
    # Remove sensitive fields
    sensitive_fields = config.get("security", {}).get("sensitive_fields", [])
    for field in sensitive_fields:
        # Recursively remove sensitive fields
        def remove_sensitive(obj, field):
            if isinstance(obj, dict):
                if field in obj:
                    obj[field] = "*****"
                for key, value in obj.items():
                    remove_sensitive(value, field)
            elif isinstance(obj, list):
                for item in obj:
                    remove_sensitive(item, field)
        
        remove_sensitive(sanitized_config, field)
    
    return sanitized_config

@app.patch("/config", tags=["Configuration"])
async def update_configuration(update: ConfigurationUpdate = Body(...)):
    """Update a specific configuration value"""
    global config
    
    # Parse the path to navigate the config dictionary
    path_parts = update.path.split(".")
    
    # Navigate to the target location
    current = config
    for i, part in enumerate(path_parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Update the value
    current[path_parts[-1]] = update.value
    
    # Save the updated configuration
    try:
        config_path = os.environ.get("SENTINEL_CONFIG", "config/settings.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Updated configuration at {update.path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        # Continue anyway for development/testing
    
    return {"status": "success", "message": f"Updated configuration at {update.path}"}

@app.post("/reset", tags=["System"])
async def reset_system():
    """Reset the system (for development/testing only)"""
    global start_time, simulated_anomalies, simulated_actions
    
    # Reset start time
    start_time = datetime.datetime.now()
    
    # Clear simulated data
    simulated_anomalies = []
    simulated_actions = []
    
    # Reset components
    if remediation_orchestrator:
        remediation_orchestrator.reset()
    
    if anomaly_detector:
        anomaly_detector.reset()
    
    logger.info("Reset system state")
    
    return {"status": "success", "message": "System state has been reset"}

# Startup event
@app.on_event("startup")
def startup_event():
    """Initialize the API on startup"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    load_config()
    
    # Initialize components
    initialize_components()
    
    # Create some simulated data for development/testing
    if not RemediationOrchestrator or not AnomalyDetector:
        create_simulated_data()
    
    logger.info("API started successfully")

def create_simulated_data():
    """Create simulated data for development/testing"""
    global simulated_anomalies, simulated_actions
    
    # Create simulated anomalies
    for i in range(5):
        anomaly_id = f"a-{i+1}"
        severity = "critical" if i == 0 else "high" if i == 1 else "medium" if i == 2 else "low"
        status = "active" if i < 2 else "in_remediation" if i == 2 else "resolved"
        
        anomaly = Anomaly(
            id=anomaly_id,
            timestamp=datetime.datetime.now() - datetime.timedelta(minutes=i*30),
            source=f"node-{i%3+1}",
            metric="cpu_usage" if i%3 == 0 else "memory_usage" if i%3 == 1 else "disk_usage",
            value=95.0 - i*10,
            threshold=80.0,
            severity=severity,
            description=f"High resource usage detected on node-{i%3+1}",
            status=status
        )
        simulated_anomalies.append(anomaly)
    
    # Create simulated remediation actions
    for i in range(3):
        action_id = f"r-{i+1}"
        anomaly_id = f"a-{i+1}" if i < 3 else ""
        status = "in_progress" if i == 0 else "completed" if i == 1 else "failed"
        action_type = "restart_pod" if i == 0 else "scale_deployment" if i == 1 else "drain_node"
        
        action = RemediationAction(
            id=action_id,
            anomaly_id=anomaly_id,
            action_type=action_type,
            target=f"deployment/app-{i+1}" if i < 2 else f"node-{i+1}",
            status=status,
            start_time=datetime.datetime.now() - datetime.timedelta(minutes=i*15),
            end_time=datetime.datetime.now() - datetime.timedelta(minutes=i*10) if i > 0 else None,
            result="Successfully scaled deployment to 3 replicas" if i == 1 else None,
            error="Failed to drain node: Node is unreachable" if i == 2 else None
        )
        simulated_actions.append(action)
    
    logger.info(f"Created {len(simulated_anomalies)} simulated anomalies and {len(simulated_actions)} simulated actions")

# Main function for running the API directly
def main():
    """Run the API server"""
    # Load configuration
    config = load_config()
    
    # Get API port from config or environment
    port = int(os.environ.get("SENTINEL_API_PORT", config.get("api", {}).get("port", 8080)))
    
    # Run the API server
    uvicorn.run(
        "sentinel_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

if __name__ == "__main__":
    main()