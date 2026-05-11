#!/bin/bash
# Create Cloud Scheduler jobs for okx_ob_download_v3 (all tickers)
# Run from anywhere: bash deploy_scheduler.sh
# Project: crypto-scalping  Region: us-central1

FUNCTION_URI="https://us-central1-crypto-scalping.cloudfunctions.net/okx_ob_download_v3"

gcloud scheduler jobs create http okx-ob-download-v3-btc \
  --location=us-central1 \
  --project=crypto-scalping \
  --schedule="* * * * *" \
  --uri="$FUNCTION_URI" \
  --message-body='{"TOCKENONE":"BTC"}' \
  --headers="Content-Type=application/json" \
  --http-method=POST \
  --time-zone=UTC \
  --attempt-deadline=120s

gcloud scheduler jobs create http okx-ob-download-v3-eth \
  --location=us-central1 \
  --project=crypto-scalping \
  --schedule="* * * * *" \
  --uri="$FUNCTION_URI" \
  --message-body='{"TOCKENONE":"ETH"}' \
  --headers="Content-Type=application/json" \
  --http-method=POST \
  --time-zone=UTC \
  --attempt-deadline=120s

gcloud scheduler jobs create http okx-ob-download-v3-sol \
  --location=us-central1 \
  --project=crypto-scalping \
  --schedule="* * * * *" \
  --uri="$FUNCTION_URI" \
  --message-body='{"TOCKENONE":"SOL"}' \
  --headers="Content-Type=application/json" \
  --http-method=POST \
  --time-zone=UTC \
  --attempt-deadline=120s
