#!/usr/bin/env bash
# scripts/test_docker_pipeline.sh
# Tests the full Docker model pipeline execution environment locally.

set -e

echo "============================================="
echo "Building docker image..."
echo "============================================="
docker build -t rewardsense-model -f Dockerfile.model .

echo "============================================="
echo "Ensuring artifact directory exists (/tmp/model_pipeline)..."
echo "============================================="
mkdir -p /tmp/model_pipeline
chmod 777 /tmp/model_pipeline

echo "============================================="
echo "Starting MLflow server..."
echo "============================================="
docker compose up -d mlflow-server
sleep 5

echo "============================================="
echo "Running training container natively..."
echo "============================================="
docker run --rm \
    --network rewardsense_default \
    -v $(pwd)/data:/app/data \
    -v /tmp/model_pipeline:/tmp/model_pipeline \
    -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
    rewardsense-model python src/model_pipeline/train.py

echo "============================================="
echo "Running Pytest downstream quality gates..."
echo "============================================="
PYTHONPATH=. pytest tests/model_pipeline/cd/ -v

echo "Done! Docker CI environment smoke test complete."
