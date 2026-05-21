#!/usr/bin/env bash
set -euo pipefail
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  --data @examples/sample_payload.json
