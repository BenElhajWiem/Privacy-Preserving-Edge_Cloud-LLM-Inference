#!/usr/bin/env bash
set -eo pipefail
CLUSTER_HOST=${CLUSTER_HOST:-cluster}
REMOTE_DIR=${REMOTE_DIR:-gap-aware-delegation}
cd "$(dirname "$0")/.."

case "${1:-}" in
  push)
    rsync -az --exclude .git --exclude '.venv*' --exclude /runs --exclude /models --exclude .env \
          --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' --exclude .DS_Store \
          ./ "$CLUSTER_HOST:$REMOTE_DIR/"
    ssh "$CLUSTER_HOST" "mkdir -p $REMOTE_DIR/runs/cache"
    rsync -az runs/cache/ "$CLUSTER_HOST:$REMOTE_DIR/runs/cache/"
    echo "pushed to $REMOTE_DIR" ;;
  pull)
    rsync -az --update "$CLUSTER_HOST:$REMOTE_DIR/runs/" runs/
    echo "pulled runs/ from $REMOTE_DIR" ;;
  *)
    echo "usage: cluster/sync.sh push|pull" >&2; exit 2 ;;
esac
