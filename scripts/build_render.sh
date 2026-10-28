#!/usr/bin/env bash
# Optional native Python Render build; Dockerfile is the default deployment.
set -euo pipefail
if [ "${INSTALL_INFERENCE:-true}" = "true" ]; then
  python -m pip install -r requirements-inference.txt
else
  python -m pip install -r requirements.txt
fi
npm --prefix frontend ci
npm --prefix frontend run build
python -m scripts.install_models --if-configured
