#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

exec "$repo_dir/scripts/test-rig.sh" \
  --features postgres \
  --test integrations \
  postgres \
  -- \
  --nocapture
