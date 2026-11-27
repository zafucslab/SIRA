#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" -m data_provider.standardize
"${PYTHON:-python}" -m data_provider.lorm
