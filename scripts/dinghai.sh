#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/train.sh" dinghai "${1:-2021}"
