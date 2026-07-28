#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

bash scripts/clean-orphan-testcontainers.sh
cargo test -p rig --features postgres --test integrations postgres -- --nocapture
