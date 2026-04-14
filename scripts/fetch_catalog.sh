#!/bin/sh
# Fetch pipeline artifacts from GCS/MLflow at container startup.
# Both steps fail gracefully — catalog.py falls back to 9 curated cards,
# and PersonalizedScorer falls back to cold-start if the model is missing.

GCS_BUCKET="us-central1-rewardsense-com-8e7127ac-bucket"
GCS_CATALOG_OBJECT="data/processed/current/offers/merged_cards.json"
LOCAL_CATALOG_PATH="/app/data/processed/current/offers/merged_cards.json"
LOCAL_MODEL_PATH="/tmp/model_cache/model.joblib"

mkdir -p "$(dirname "$LOCAL_CATALOG_PATH")"
mkdir -p "$(dirname "$LOCAL_MODEL_PATH")"

# ── 1. Card catalog ──────────────────────────────────────────────────────────
echo "[startup] Fetching card catalog from GCS..."
python3 - <<EOF
import sys
try:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("$GCS_BUCKET")
    blob = bucket.blob("$GCS_CATALOG_OBJECT")
    blob.download_to_filename("$LOCAL_CATALOG_PATH")
    import json
    with open("$LOCAL_CATALOG_PATH") as f:
        cards = json.load(f)
    count = len(cards) if isinstance(cards, list) else 0
    print(f"[startup] Card catalog OK — {count} cards.")
except Exception as e:
    print(f"[startup] WARNING: Card catalog fetch failed: {e} — using curated fallback (9 cards).")
EOF

# ── 2. Personalization model ─────────────────────────────────────────────────
echo "[startup] Fetching personalization model from MLflow..."
python3 - <<EOF
import sys, os

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
model_name   = os.getenv("REGISTERED_MODEL_NAME", "personalization")
model_stage  = os.getenv("MODEL_STAGE", "Production")
local_path   = "$LOCAL_MODEL_PATH"

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    versions = client.get_latest_versions(model_name, stages=[model_stage])
    if not versions:
        print(f"[startup] WARNING: No '{model_name}' model in stage '{model_stage}' — cold-start mode.")
        sys.exit(0)

    version_info = versions[0]
    model_uri = f"models:/{model_name}/{model_stage}"
    print(f"[startup] Downloading model version {version_info.version} from {model_uri}...")

    import joblib
    raw_model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(raw_model, local_path)
    print(f"[startup] Model cached to {local_path}.")
except Exception as e:
    print(f"[startup] WARNING: Model fetch failed: {e} — cold-start mode.")
EOF
