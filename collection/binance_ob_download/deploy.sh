#!/bin/bash
# Deploy binance_ob_download to Google Cloud Functions Gen2
# Run from this directory: bash deploy.sh

gcloud functions deploy binance_ob_download \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=binance_ob_download \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=120s \
  --memory=512MB \
  --env-vars-file=env.yaml
