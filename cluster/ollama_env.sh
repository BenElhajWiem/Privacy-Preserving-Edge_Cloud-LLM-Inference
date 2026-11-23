OLLAMA_VERSION=${OLLAMA_VERSION:-0.40.3}
ODIR=${SCRATCH:-$HOME/scratch}/ollama
OBIN=$ODIR/$OLLAMA_VERSION
mkdir -p $ODIR/models logs
(
  flock 9
  if [ ! -x $OBIN/bin/ollama ]; then
    mkdir -p $OBIN
    curl -fsSL https://github.com/ollama/ollama/releases/download/v$OLLAMA_VERSION/ollama-linux-amd64.tar.zst \
      | tar --zstd -xf - -C $OBIN
  fi
) 9> $ODIR/.install-$OLLAMA_VERSION.lock
export PATH=$OBIN/bin:$PATH OLLAMA_MODELS=$ODIR/models

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
export OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_HOST_URL=http://127.0.0.1:$PORT
ollama serve > logs/ollama-${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}.log 2>&1 &
trap "kill $! 2>/dev/null" EXIT
for _ in $(seq 60); do curl -sf $OLLAMA_HOST_URL/api/version >/dev/null && break; sleep 2; done
curl -sf $OLLAMA_HOST_URL/api/version || { echo "ollama did not start; see logs/"; exit 1; }
echo " ollama on $(hostname -s):$PORT"
