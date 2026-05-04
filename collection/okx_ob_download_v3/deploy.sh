#!/bin/bash
# Deploy to Google Cloud Functions
# Run from this directory: bash deploy.sh

gcloud functions deploy okx_ob_download_v3 \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=okx_ob_download_v3 \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=120s \
  --memory=512MB \
  --env-vars-file=env.yaml
