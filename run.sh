#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install --quiet \
  "numpy" \
  "scipy" \
  "scikit-learn>=1.3" \
  "matplotlib" \
  "pandas" \
  "certifi" \
  "folktables" \
  "torch" \
  "sentence-transformers" \
  "datasets"
# macOS framework Python lacks system CA certs — point SSL at certifi.
SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
python extract.py
