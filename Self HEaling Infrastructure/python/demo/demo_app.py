#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo Application for SelfHealing Sentinel

This application simulates a real-world service with configurable failure modes
to demonstrate the self-healing capabilities of the SelfHealing Sentinel system.

Features:
- HTTP API with configurable endpoints
- Simulated failures (CPU spikes, memory leaks, slow responses, errors)
- Prometheus metrics for monitoring
- Kubernetes-friendly design
"""

import os
import sys
import time
import random
import logging
import threading
import argparse
import json
import signal
import datetime
from typing import Dict, List, Optional, Any, Union

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query, Path, Response, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install required packages: pip install fastapi uvicorn pydantic prometheus-client")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/demo_app.log")
    ]
)
logger = logging.getLogger("demo-app")

# Initialize FastAPI app
app = FastAPI(
    title="SelfHealing Sentinel Demo App",
    description="Demo application with configurable failure modes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus metrics
REQUEST_COUNT = Counter(
    'demo_app_requests_total',
    'Total number of requests received',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'demo_app_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)
ACTIVE_REQUESTS = Gauge(
    'demo_app_active_requests',
    'Number of active requests'
)
MEMORY_USAGE = Gauge(
    'demo_app_memory_usage_bytes',
    'Memory usage in bytes'
)
CPU_USAGE = Gauge(
    'demo_app_cpu_usage_percent',
    'CPU usage percentage'
)
ERROR_COUNT = Counter(
    'demo_app_errors_total',
    'Total number of errors',
    ['type']
)

# Define Pydantic models for API
class HealthStatus(BaseModel):
    status: str = Field(..., description="Application health status")
    version: str = Field(..., description="Application version")
    uptime: int = Field(..., description="Application uptime in seconds")

class FailureConfig(BaseModel):
    enabled: bool = Field(..., description="Whether failures are enabled")
    probability: float = Field(..., description="Probability of failure (0-1)")
    types: List[str] = Field(..., description="Types of failures to simulate")

class Item(BaseModel):
    id: int = Field(..., description="Item ID")
    name: str = Field(..., description="Item name")
    description: str = Field(..., description="Item description")
    price: float = Field(..., description="Item price")
    category: str = Field(..., description="Item category")
    in_stock: bool = Field(..., description="Whether the item is in stock")

# Global variables
start_time = datetime.datetime.now()
failure_config = {
    "enabled": os.environ.get("DEMO_FAILURES_ENABLED", "true").lower() == "true",
    "probability": float(os.environ.get("DEMO_FAILURE_PROBABILITY", "0.1")),
    "types": os.environ.get("DEMO_FAILURE_TYPES", "high_cpu,memory_leak,slow_response,error_500").split(",")
}

# Simulated data
items_db = [
    {
        "id": i,
        "name": f"Item {i}",
        "description": f"This is item {i}",
        "price": random.uniform(10.0, 100.0),
        "category": random.choice(["Electronics", "Clothing", "Food", "Books"]),
        "in_stock": random.choice([True, False])
    }
    for i in range(1, 101)
]

# Memory leak simulation
memory_leak_data = []

# Middleware for request metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    # Increment active requests
    ACTIVE_REQUESTS.inc()
    
    # Simulate failures if enabled
    should_fail = False
    failure_type = None
    
    if failure_config["enabled"] and random.random() < failure_config["probability"]:
        failure_type = random.choice(failure_config["types"])
        should_fail = True
        logger.info(f"Simulating failure: {failure_type}")
    
    try:
        if should_fail:
            if failure_type == "high_cpu":
                # Simulate CPU spike
                logger.info("Simulating high CPU usage")
                simulate_high_cpu()
                CPU_USAGE.set(95.0)  # Set a high CPU usage value for monitoring
            elif failure_type == "memory_leak":
                # Simulate memory leak
                logger.info("Simulating memory leak")
                simulate_memory_leak()
                # Update memory usage metric
                MEMORY_USAGE.set(len(memory_leak_data) * 1024)  # Rough estimate
            elif failure_type == "slow_response":
                # Simulate slow response
                logger.info("Simulating slow response")
                time.sleep(random.uniform(5.0, 15.0))
            elif failure_type == "error_500":
                # Simulate server error
                logger.info("Simulating server error")
                ERROR_COUNT.labels(type="server_error").inc()
                return JSONResponse(
                    status_code=500,
                    content={"error": "Internal Server Error", "message": "Simulated server error"}
                )
        
        # Process the request normally
        response = await call_next(request)
        
        # Record metrics
        process_time = time.time() - start_time
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
    except Exception as e:
        # Record error
        ERROR_COUNT.labels(type="unhandled_exception").inc()
        logger.error(f"Unhandled exception: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )
    finally:
        # Decrement active requests
        ACTIVE_REQUESTS.dec()

# Simulation functions
def simulate_high_cpu():
    """Simulate high CPU usage"""
    end_time = time.time() + random.uniform(1.0, 3.0)
    while time.time() < end_time:
        # Perform CPU-intensive calculation
        [i**2 for i in range(10000)]

def simulate_memory_leak():
    """Simulate memory leak"""
    global memory_leak_data
    # Add random data to the global list
    memory_leak_data.extend([random.random() for _ in range(1000000)])

# Routes
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the SelfHealing Sentinel Demo App"}

@app.get("/health", response_model=HealthStatus, tags=["General"])
async def health_check():
    """Check the health status of the application"""
    uptime = (datetime.datetime.now() - start_time).total_seconds()
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": int(uptime)
    }

@app.get("/metrics", tags=["General"])
async def metrics():
    """Expose Prometheus metrics"""
    return Response(content=prometheus_client.generate_latest(), media_type="text/plain")

@app.get("/failures", tags=["Configuration"])
async def get_failure_config():
    """Get the current failure configuration"""
    return failure_config

@app.post("/failures", tags=["Configuration"])
async def update_failure_config(config: FailureConfig):
    """Update the failure configuration"""
    global failure_config
    failure_config = {
        "enabled": config.enabled,
        "probability": config.probability,
        "types": config.types
    }
    logger.info(f"Updated failure configuration: {failure_config}")
    return failure_config

@app.get("/items", tags=["Items"])
async def get_items(
    skip: int = Query(0, description="Number of items to skip"),
    limit: int = Query(10, description="Maximum number of items to return"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get a list of items with optional filtering"""
    filtered_items = items_db
    
    if category:
        filtered_items = [item for item in filtered_items if item["category"] == category]
    
    return filtered_items[skip:skip+limit]

@app.get("/items/{item_id}", tags=["Items"])
async def get_item(item_id: int = Path(..., description="ID of the item to retrieve")):
    """Get a specific item by ID"""
    item = next((item for item in items_db if item["id"] == item_id), None)
    
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    return item

@app.post("/items", status_code=status.HTTP_201_CREATED, tags=["Items"])
async def create_item(item: Item):
    """Create a new item"""
    # Check if item with this ID already exists
    if any(existing["id"] == item.id for existing in items_db):
        raise HTTPException(status_code=400, detail=f"Item with ID {item.id} already exists")
    
    # Convert Pydantic model to dict and add to database
    item_dict = item.dict()
    items_db.append(item_dict)
    
    return item_dict

@app.put("/items/{item_id}", tags=["Items"])
async def update_item(item: Item, item_id: int = Path(..., description="ID of the item to update")):
    """Update an existing item"""
    # Find the item
    item_index = next((i for i, existing in enumerate(items_db) if existing["id"] == item_id), None)
    
    if item_index is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    # Update the item
    item_dict = item.dict()
    items_db[item_index] = item_dict
    
    return item_dict

@app.delete("/items/{item_id}", tags=["Items"])
async def delete_item(item_id: int = Path(..., description="ID of the item to delete")):
    """Delete an item"""
    # Find the item
    item_index = next((i for i, existing in enumerate(items_db) if existing["id"] == item_id), None)
    
    if item_index is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    # Delete the item
    deleted_item = items_db.pop(item_index)
    
    return {"message": f"Item {item_id} deleted", "item": deleted_item}

@app.get("/simulate/cpu", tags=["Simulation"])
async def simulate_cpu_spike(
    duration: float = Query(5.0, description="Duration of CPU spike in seconds")
):
    """Simulate a CPU spike"""
    logger.info(f"Manually simulating CPU spike for {duration} seconds")
    
    # Start CPU spike in a separate thread
    def cpu_spike():
        end_time = time.time() + duration
        while time.time() < end_time:
            # Perform CPU-intensive calculation
            [i**2 for i in range(10000)]
        
        # Reset CPU usage metric after spike
        CPU_USAGE.set(30.0)  # Set back to normal
    
    # Set CPU usage metric
    CPU_USAGE.set(95.0)
    
    # Start thread
    threading.Thread(target=cpu_spike).start()
    
    return {"message": f"Simulating CPU spike for {duration} seconds"}

@app.get("/simulate/memory", tags=["Simulation"])
async def simulate_memory_leak_endpoint(
    size: int = Query(1000000, description="Size of memory leak in elements")
):
    """Simulate a memory leak"""
    logger.info(f"Manually simulating memory leak of {size} elements")
    
    global memory_leak_data
    # Add random data to the global list
    memory_leak_data.extend([random.random() for _ in range(size)])
    
    # Update memory usage metric
    MEMORY_USAGE.set(len(memory_leak_data) * 1024)  # Rough estimate
    
    return {"message": f"Simulated memory leak of {size} elements", "total_size": len(memory_leak_data)}

@app.get("/simulate/slow", tags=["Simulation"])
async def simulate_slow_response(
    delay: float = Query(10.0, description="Delay in seconds")
):
    """Simulate a slow response"""
    logger.info(f"Manually simulating slow response of {delay} seconds")
    
    # Sleep for the specified delay
    time.sleep(delay)
    
    return {"message": f"Response delayed by {delay} seconds"}

@app.get("/simulate/error", tags=["Simulation"])
async def simulate_error(
    status_code: int = Query(500, description="HTTP status code to return")
):
    """Simulate an error response"""
    logger.info(f"Manually simulating error with status code {status_code}")
    
    # Increment error counter
    ERROR_COUNT.labels(type="manual_error").inc()
    
    # Return error response
    raise HTTPException(
        status_code=status_code,
        detail=f"Simulated error with status code {status_code}"
    )

@app.get("/simulate/reset", tags=["Simulation"])
async def reset_simulation():
    """Reset all simulated failures"""
    logger.info("Resetting all simulations")
    
    global memory_leak_data
    memory_leak_data = []
    
    # Reset metrics
    CPU_USAGE.set(30.0)  # Normal CPU usage
    MEMORY_USAGE.set(50 * 1024 * 1024)  # Normal memory usage (50MB)
    
    return {"message": "All simulations reset"}

# Startup event
@app.on_event("startup")
def startup_event():
    """Initialize the application on startup"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set initial metric values
    CPU_USAGE.set(30.0)  # Initial CPU usage
    MEMORY_USAGE.set(50 * 1024 * 1024)  # Initial memory usage (50MB)
    
    logger.info("Demo application started successfully")

# Shutdown event
@app.on_event("shutdown")
def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Demo application shutting down")

# Signal handlers
def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, shutting down")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main function for running the application directly
def main():
    """Run the application server"""
    parser = argparse.ArgumentParser(description="SelfHealing Sentinel Demo Application")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--failures", type=str, default="true", help="Enable simulated failures (true/false)")
    parser.add_argument("--probability", type=float, default=0.1, help="Probability of failure (0-1)")
    parser.add_argument("--failure-types", type=str, default="high_cpu,memory_leak,slow_response,error_500", 
                        help="Comma-separated list of failure types to simulate")
    
    args = parser.parse_args()
    
    # Update failure configuration from command line arguments
    global failure_config
    failure_config = {
        "enabled": args.failures.lower() == "true",
        "probability": args.probability,
        "types": args.failure_types.split(",")
    }
    
    logger.info(f"Starting demo application on {args.host}:{args.port}")
    logger.info(f"Failure configuration: {failure_config}")
    
    # Start Prometheus metrics server on a different port
    metrics_port = int(os.environ.get("METRICS_PORT", "8001"))
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics available on port {metrics_port}")
    
    # Run the application
    uvicorn.run(
        "demo_app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()