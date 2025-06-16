# SelfHealing Sentinel - Setup Guide

## Prerequisites

Before setting up the SelfHealing Sentinel system, ensure you have the following prerequisites installed:

- **Docker and Docker Compose**: For running the components locally
- **Kubernetes Cluster**: For deploying in a production environment
- **Python 3.8+**: For running the Python scripts
- **Ansible**: For executing remediation playbooks
- **kubectl**: For interacting with the Kubernetes cluster

## Local Development Setup

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/self-healing-sentinel.git
cd self-healing-sentinel
```

### 2. Create Project Structure

Run the setup script to create the project directory structure:

```bash
# On Windows
powershell -ExecutionPolicy Bypass -File setup.ps1

# On Linux/macOS
chmod +x setup.sh
./setup.sh
```

### 3. Configure the System

Update the configuration files in the `config` directory to match your environment:

- `settings.yaml`: Global settings
- `thresholds.yaml`: Detection thresholds
- `remediation.yaml`: Remediation configuration

### 4. Start the System with Docker Compose

Start all components using Docker Compose:

```bash
docker-compose up -d
```

This will start the following components:

- Prometheus (http://localhost:9090)
- Grafana (http://localhost:3000)
- Alertmanager (http://localhost:9093)
- Anomaly Detector
- Remediation Orchestrator
- Sentinel API (http://localhost:8080)
- Demo Application (http://localhost:8000)

### 5. Access the Dashboards

- **Grafana**: Open http://localhost:3000 in your browser (default credentials: admin/sentinel123)
- **Prometheus**: Open http://localhost:9090 in your browser
- **Sentinel API**: Open http://localhost:8080/docs in your browser for the API documentation

## Kubernetes Deployment

### 1. Prepare Kubernetes Manifests

The Kubernetes manifests are located in the `kubernetes` directory. Review and update them as needed:

```bash
cd kubernetes
```

### 2. Create Namespace

Create a namespace for the SelfHealing Sentinel system:

```bash
kubectl create namespace self-healing-sentinel
```

### 3. Apply Manifests

Apply the Kubernetes manifests:

```bash
kubectl apply -f deployments/
kubectl apply -f services/
kubectl apply -f custom-resources/
```

### 4. Deploy KEDA Scalers

Apply the KEDA scaler configurations:

```bash
kubectl apply -f ../keda/scalers/
```

### 5. Verify Deployment

Verify that all components are running:

```bash
kubectl get pods -n self-healing-sentinel
kubectl get services -n self-healing-sentinel
```

## Testing the System

### 1. Generate Test Load

Generate test load on the demo application:

```bash
# Using hey (https://github.com/rakyll/hey)
hey -z 5m -c 50 http://localhost:8000
```

### 2. Simulate Failures

Simulate failures to test the self-healing capabilities:

```bash
# Simulate pod failure
kubectl delete pod -n self-healing-sentinel <pod-name>

# Simulate high CPU usage
python python/tests/simulate_high_cpu.py
```

### 3. Monitor Remediation Actions

Monitor the remediation actions in the logs:

```bash
kubectl logs -n self-healing-sentinel -l app=remediation-orchestrator -f
```

## Configuring Alerts

### 1. Update Alert Rules

The alert rules are defined in `prometheus/rules/`. Update them as needed:

```yaml
# Example: Lower the threshold for high CPU usage
- alert: NodeHighCPU
  expr: instance:node_cpu_utilisation:rate5m > 0.8  # Changed from 0.9 to 0.8
  for: 10m
  labels:
    severity: warning
    category: node
  annotations:
    summary: Node {{ $labels.instance }} high CPU usage
    description: Node {{ $labels.instance }} has high CPU usage (> 80%) for more than 10 minutes
    remediation_action: investigate
```

### 2. Reload Prometheus Configuration

After updating the alert rules, reload the Prometheus configuration:

```bash
curl -X POST http://localhost:9090/-/reload
```

## Customizing Remediation Actions

### 1. Update Remediation Configuration

Update the remediation configuration in `config/remediation.yaml`:

```yaml
actions:
  restart_pod:
    max_attempts: 3
    cooldown_period: 300  # seconds
    escalation: recreate_deployment
```

### 2. Create Custom Remediation Scripts

Create custom remediation scripts in `python/remediation/actions/`:

```python
def custom_remediation_action(target, context):
    # Implement custom remediation logic
    pass
```

### 3. Update Ansible Playbooks

Update the Ansible playbooks in `ansible/playbooks/` for infrastructure-level remediation:

```yaml
- name: Custom Remediation Playbook
  hosts: all
  tasks:
    - name: Execute custom remediation
      # Custom remediation tasks
```

## Troubleshooting

### Common Issues

1. **Components not starting**: Check Docker logs
   ```bash
   docker-compose logs -f
   ```

2. **Prometheus not scraping metrics**: Check Prometheus targets
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

3. **Remediation actions not triggering**: Check alert status
   ```bash
   curl http://localhost:9090/api/v1/alerts
   ```

4. **Kubernetes deployment issues**: Check pod status and logs
   ```bash
   kubectl describe pod -n self-healing-sentinel <pod-name>
   kubectl logs -n self-healing-sentinel <pod-name>
   ```

## Next Steps

- Integrate with your CI/CD pipeline
- Add custom metrics for your applications
- Develop additional remediation strategies
- Set up notification channels (email, Slack, etc.)
- Implement machine learning models for advanced anomaly detection