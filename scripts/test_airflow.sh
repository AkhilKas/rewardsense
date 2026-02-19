#!/bin/bash
# Test Airflow DAGs locally via Docker Compose
#
# Usage:
#   chmod +x scripts/test_airflow.sh
#   ./scripts/test_airflow.sh
#
# Prerequisites:
#   - Docker Desktop must be running
#   - Port 8080 must be free (or the webserver will fail to start,
#     but the scheduler can still run DAG checks)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Starting Airflow ==="
docker-compose up -d airflow-postgres airflow-init
echo "Waiting for DB init..."
sleep 15

docker-compose up -d airflow-scheduler
echo "Waiting for scheduler to be ready..."
sleep 15

echo ""
echo "=== Listing DAGs ==="
docker-compose exec -T airflow-scheduler airflow dags list

echo ""
echo "=== Validating rewardsense_data_pipeline ==="
docker-compose exec -T airflow-scheduler airflow dags show rewardsense_data_pipeline

echo ""
echo "=== Running DAG tests ==="
docker-compose exec -T airflow-scheduler \
    pip install --quiet pytest 2>/dev/null
docker-compose exec -T airflow-scheduler \
    python -m pytest /opt/airflow/tests/dags/ -v

echo ""
echo "=== Done ==="
