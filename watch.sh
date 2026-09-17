#!/usr/bin/env bash
# Polls the SprudelJump dsh session log + GPU + workspace files, printing a
# line whenever something changes. Ctrl+C to stop.
set -euo pipefail

SESSION_LOG=~/.dsh/sessions/--home-jo-projects-sprudelJump--/session-721949c8-4a5a-4d45-9daf-997c1f1ff203/session.v3.jsonl.zstd
PROJECT_DIR=/home/jo/projects/sprudelJump
INTERVAL=15

last_seq=-1
last_env_mtime=""
last_status_mtime=""

echo "watching (poll every ${INTERVAL}s, Ctrl+C to stop)..."

while true; do
  # --- session log: print any new events since the last poll ---
  zstd -dc "$SESSION_LOG" 2>/dev/null | python3 -c "
import sys, json, datetime

last_seq = $last_seq
compactions = {}
for line in sys.stdin:
    d = json.loads(line)
    seq = d.get('seq', -1)
    if seq <= last_seq:
        continue
    t = d.get('time')
    ts = datetime.datetime.fromtimestamp(t/1000).strftime('%H:%M:%S') if t else '?'
    typ = d.get('type')
    data = d.get('data', {})

    if typ == 'compaction/start':
        compactions[data.get('compactionId')] = t
        print(f'{ts}  compaction started (turn {data.get(\"turn\")})')
    elif typ == 'compaction/end':
        cid = data.get('compactionId')
        started = compactions.pop(cid, None)
        dur = f'{(t-started)/1000:.0f}s' if started else '?'
        err = data.get('error')
        if err:
            print(f'{ts}  compaction FAILED after {dur}: {err}')
        else:
            print(f'{ts}  compaction finished ({dur})')
    elif typ == 'turn/end':
        reason = data.get('reason', {})
        print(f'{ts}  turn {data.get(\"turn\")} ended: {reason.get(\"kind\")}')
    elif typ == 'tool/call':
        name = data.get('name')
        print(f'{ts}  tool call: {name}')
    elif typ == 'assistant/message':
        msg = data.get('message', {})
        for c in msg.get('content', []):
            if c.get('type') == 'reasoning':
                snippet = c['text'].strip().splitlines()[0][:120]
                print(f'{ts}  reasoning: {snippet}')
            elif c.get('type') == 'text':
                snippet = c['text'].strip().splitlines()[0][:120]
                print(f'{ts}  message: {snippet}')

    print(f'__LAST_SEQ__{seq}')
" | tee /tmp/claude-1000/-home-jo/40b895f2-edec-4e32-adbc-d77739a1ae91/scratchpad/sprudel-watch-out.txt | grep -v '^__LAST_SEQ__' || true

  new_seq=$(grep '^__LAST_SEQ__' /tmp/claude-1000/-home-jo/40b895f2-edec-4e32-adbc-d77739a1ae91/scratchpad/sprudel-watch-out.txt 2>/dev/null | tail -1 | sed 's/__LAST_SEQ__//')
  if [ -n "${new_seq:-}" ]; then
    last_seq=$new_seq
  fi

  # --- workspace files: report env.py / STATUS.md appearing or changing ---
  if [ -f "$PROJECT_DIR/env.py" ]; then
    mtime=$(stat -c %Y "$PROJECT_DIR/env.py")
    if [ "$mtime" != "$last_env_mtime" ]; then
      lines=$(wc -l < "$PROJECT_DIR/env.py")
      echo "$(date +%H:%M:%S)  env.py updated (${lines} lines)"
      last_env_mtime=$mtime
    fi
  fi
  if [ -f "$PROJECT_DIR/STATUS.md" ]; then
    mtime=$(stat -c %Y "$PROJECT_DIR/STATUS.md")
    if [ "$mtime" != "$last_status_mtime" ]; then
      echo "$(date +%H:%M:%S)  STATUS.md updated:"
      sed 's/^/    /' "$PROJECT_DIR/STATUS.md"
      last_status_mtime=$mtime
    fi
  fi

  sleep "$INTERVAL"
done
