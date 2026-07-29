#!/usr/bin/env bash
# Remove Rig PostgreSQL testcontainers left behind by an uncatchable runner death.
set -uo pipefail

readonly LABEL="com.cluster.test-suite=rig-postgres"
readonly AGE_MIN="${TESTCONTAINERS_ORPHAN_AGE_MIN:-30}"

command -v docker >/dev/null 2>&1 || {
  echo "docker not found; skipping Rig PostgreSQL testcontainer sweep"
  exit 0
}

now=$(date +%s)
removed=0

for id in $(docker ps -aq \
  --filter label=org.testcontainers.managed-by=testcontainers \
  --filter "label=${LABEL}" 2>/dev/null); do
  runner_pid=$(
    docker inspect --format '{{index .Config.Labels "com.cluster.test-runner-pid"}}' \
      "$id" 2>/dev/null
  ) || continue
  case "$runner_pid" in
    ''|*[!0-9]*) continue ;;
  esac
  if ps -p "$runner_pid" -o pid= >/dev/null 2>&1; then
    continue
  fi

  created=$(docker inspect --format '{{.Created}}' "$id" 2>/dev/null) || continue
  if ! created_epoch=$(date -d "$created" +%s 2>/dev/null); then
    created_epoch=$(
      date -j -u -f '%Y-%m-%dT%H:%M:%S' "${created%%.*}" +%s 2>/dev/null
    ) || continue
  fi
  [ "$created_epoch" -gt 0 ] 2>/dev/null || continue

  age_min=$(( (now - created_epoch) / 60 ))
  if [ "$age_min" -ge "$AGE_MIN" ]; then
    docker rm -fv "$id" >/dev/null 2>&1 && removed=$((removed + 1))
  fi
done

if [ "$removed" -gt 0 ]; then
  echo "removed $removed orphaned Rig PostgreSQL testcontainer(s)"
fi
