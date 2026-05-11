#!/bin/bash
# Create Cloud Scheduler jobs for binance_ob_download (all tickers)
# Run after deploy.sh: bash deploy_scheduler.sh
# Project: crypto-scalping  Region: us-central1

FUNCTION_URI="https://us-central1-crypto-scalping.cloudfunctions.net/binance_ob_download"

gcloud scheduler jobs create http binance-ob-download-btc \
  --location=us-central1 \
  --project=crypto-scalping \
  --schedule="* * * * *" \
  --uri="$FUNCTION_URI" \
  --message-body='{"TOCKENONE":"BTC"}' \
  --headers="Content-Type=application/json" \
  --http-method=POST \
  --time-zone=UTC \
  --attempt-deadline=120s

gcloud scheduler jobs create http binance-ob-download-eth \
  --location=us-central1 \
  --project=crypto-scalping \
  --schedule="* * * * *" \
  --uri="$FUNCTION_URI" \
  --message-body='{"TOCKENONE":"ETH"}' \
  --headers="Content-Type=application/json" \
  --http-method=POST \
  --time-zone=UTC \
  --attempt-deadline=120s

gcloud scheduler jobs create http binance-ob-download-sol \
  --location=us-central1 \
  --project=crypto-scalping \
  --schedule="* * * * *" \
  --uri="$FUNCTION_URI" \
  --message-body='{"TOCKENONE":"SOL"}' \
  --headers="Content-Type=application/json" \
  --http-method=POST \
  --time-zone=UTC \
  --attempt-deadline=120s
