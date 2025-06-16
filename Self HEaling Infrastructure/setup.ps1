# PowerShell script to create the project directory structure

$projectRoot = "$PSScriptRoot\self-healing-sentinel"

# Create main directories
$directories = @(
    "kubernetes\deployments",
    "kubernetes\services",
    "kubernetes\custom-resources",
    "keda\scalers",
    "ansible\playbooks",
    "ansible\inventory",
    "ansible\roles\monitoring",
    "ansible\roles\remediation",
    "ansible\roles\verification",
    "prometheus\rules",
    "prometheus\dashboards",
    "prometheus\exporters\app-metrics",
    "prometheus\exporters\system-metrics",
    "python\anomaly_detection\ml_models",
    "python\anomaly_detection\detectors",
    "python\remediation\actions",
    "python\api_integration",
    "python\tests",
    "api\routes",
    "api\models",
    "config",
    "docs"
)

# Create directories
foreach ($dir in $directories) {
    $path = Join-Path -Path $projectRoot -ChildPath $dir
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    Write-Host "Created directory: $path"
}

# Create placeholder files to maintain directory structure in git
foreach ($dir in $directories) {
    $path = Join-Path -Path $projectRoot -ChildPath $dir
    $placeholderFile = Join-Path -Path $path -ChildPath ".gitkeep"
    New-Item -ItemType File -Path $placeholderFile -Force | Out-Null
}

# Copy README.md to project root
$readmePath = Join-Path -Path $projectRoot -ChildPath "README.md"
Copy-Item -Path "$PSScriptRoot\README.md" -Destination $readmePath -Force

Write-Host "Project structure created successfully at $projectRoot"