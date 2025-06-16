# SelfHealing Sentinel Automation - Project Structure

```
self-healing-sentinel/
├── kubernetes/                  # Kubernetes configuration files
│   ├── deployments/             # Application deployment manifests
│   │   ├── monitoring.yaml      # Monitoring stack deployment
│   │   ├── automation.yaml      # Automation components deployment
│   │   └── intelligence.yaml    # Intelligence components deployment
│   ├── services/                # Service definitions
│   │   ├── api-service.yaml     # API service configuration
│   │   └── metrics-service.yaml # Metrics service configuration
│   └── custom-resources/        # Custom resource definitions
│       └── remediation-crd.yaml # Remediation action CRD
│
├── keda/                        # KEDA scaler configurations
│   └── scalers/                 # Custom KEDA scalers
│       ├── health-scaler.yaml   # Health-based scaling configuration
│       └── load-scaler.yaml     # Load-based scaling configuration
│
├── ansible/                     # Ansible automation
│   ├── playbooks/               # Automation playbooks
│   │   ├── node-recovery.yml    # Node recovery playbook
│   │   ├── service-restart.yml  # Service restart playbook
│   │   └── scaling.yml          # Manual scaling playbook
│   ├── inventory/               # Infrastructure inventory
│   │   └── hosts.yml            # Host definitions
│   └── roles/                   # Reusable automation roles
│       ├── monitoring/          # Monitoring setup role
│       ├── remediation/         # Remediation actions role
│       └── verification/        # Verification checks role
│
├── prometheus/                  # Prometheus configuration
│   ├── rules/                   # Alert and recording rules
│   │   ├── node-alerts.yml      # Node-related alerts
│   │   ├── service-alerts.yml   # Service-related alerts
│   │   └── custom-metrics.yml   # Custom metrics recording rules
│   ├── dashboards/              # Grafana dashboard definitions
│   │   ├── overview.json        # System overview dashboard
│   │   ├── health.json          # Health metrics dashboard
│   │   └── remediation.json     # Remediation actions dashboard
│   └── exporters/               # Custom metric exporters
│       ├── app-metrics/         # Application-specific metrics
│       └── system-metrics/      # System-level metrics
│
├── python/                      # Python scripts and applications
│   ├── anomaly_detection/       # Anomaly detection algorithms
│   │   ├── ml_models/           # Machine learning models
│   │   │   ├── isolation_forest.py  # Isolation Forest implementation
│   │   │   └── lstm_predictor.py    # LSTM-based prediction model
│   │   └── detectors/           # Detector implementations
│   │       ├── metric_detector.py   # Metric-based anomaly detector
│   │       └── log_detector.py      # Log-based anomaly detector
│   ├── remediation/             # Remediation logic
│   │   ├── actions/             # Remediation actions
│   │   │   ├── node_actions.py      # Node-related actions
│   │   │   ├── pod_actions.py       # Pod-related actions
│   │   │   └── service_actions.py   # Service-related actions
│   │   └── orchestrator.py      # Remediation orchestrator
│   ├── api_integration/         # Kubernetes API integration
│   │   ├── k8s_client.py        # Kubernetes client wrapper
│   │   └── prometheus_client.py # Prometheus API client
│   └── tests/                   # Unit and integration tests
│       ├── test_detection.py    # Tests for detection algorithms
│       └── test_remediation.py  # Tests for remediation actions
│
├── api/                         # Management API
│   ├── server.py                # API server implementation
│   ├── routes/                  # API routes
│   │   ├── metrics.py           # Metrics endpoints
│   │   ├── remediation.py       # Remediation endpoints
│   │   └── configuration.py     # Configuration endpoints
│   └── models/                  # API data models
│       └── schemas.py           # API schema definitions
│
├── config/                      # Configuration files
│   ├── settings.yaml            # Global settings
│   ├── thresholds.yaml          # Detection thresholds
│   └── remediation.yaml         # Remediation configuration
│
└── docs/                        # Documentation
    ├── setup.md                 # Setup instructions
    ├── usage.md                 # Usage guide
    ├── architecture.md          # Detailed architecture description
    ├── api.md                   # API documentation
    └── development.md           # Development guidelines
```