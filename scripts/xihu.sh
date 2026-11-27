#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/train.sh" xihu "${1:-2021}"
