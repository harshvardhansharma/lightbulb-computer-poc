#!/usr/bin/env bash
# Start the Lightbulb Computer server and open it in Safari (Chrome can't see an iPhone Continuity Camera).
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
MODEL=web/models/hand_landmarker.task
[ -f "$MODEL" ] || { mkdir -p web/models && curl -sL -o "$MODEL" \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task; }
[ -f .env ] && set -a && . ./.env && set +a
(sleep 1 && open -a "${LB_BROWSER:-Safari}" "http://localhost:${PORT:-8000}") &
exec .venv/bin/python -u server.py
