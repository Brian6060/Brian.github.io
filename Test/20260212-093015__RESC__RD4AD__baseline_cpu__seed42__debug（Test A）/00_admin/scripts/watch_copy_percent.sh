#!/usr/bin/env bash
set -euo pipefail

PORT="${1:?port}"
HOST="${2:?host}"
REMOTE_FILE="${3:?remote_file}"
LOCAL_FILE="${4:?local_file}"
INTERVAL="${5:-2}"

# mac 本地文件大小
TOTAL=$(stat -f%z "$LOCAL_FILE")
if [ "$TOTAL" -le 0 ]; then
  echo "Local file size invalid: $LOCAL_FILE"
  exit 1
fi

echo "TOTAL=$TOTAL bytes"
echo "Monitoring remote: $HOST:$REMOTE_FILE"
echo "Interval: ${INTERVAL}s"
echo

prev_remote=0
prev_ts=$(date +%s)

while true; do
  # 远端已写入字节数（Linux stat）
  remote=$( /usr/bin/ssh -p "$PORT" -o ProxyCommand=none -o ProxyJump=none \
    "$HOST" "stat -c %s '$REMOTE_FILE' 2>/dev/null || echo 0" )

  ts=$(date +%s)
  dt=$((ts - prev_ts))
  if [ "$dt" -le 0 ]; then dt=1; fi

  delta=$((remote - prev_remote))
  speed=$((delta / dt)) # bytes/s
  pct=$(python3 - <<PY
total=$TOTAL
remote=$remote
print(f"{min(100.0, (remote/total)*100):6.2f}")
PY
)

  eta=$(python3 - <<PY
total=$TOTAL
remote=$remote
speed=$speed
if speed<=0:
    print("INF")
else:
    print(int((total-remote)/speed))
PY
)

  printf "\rRemote: %12d / %12d bytes | %6.2f%% | %8d B/s | ETA(s): %-6s" \
    "$remote" "$TOTAL" "$pct" "$speed" "$eta"

  if [ "$remote" -ge "$TOTAL" ]; then
    echo
    echo "==> Reached local size. (Not equal to zip integrity; run unzip -tq to verify.)"
    exit 0
  fi

  prev_remote=$remote
  prev_ts=$ts
  sleep "$INTERVAL"
done
