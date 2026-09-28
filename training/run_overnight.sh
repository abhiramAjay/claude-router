#!/bin/sh
# Unattended corpus build: keep every provider generating until the target is
# reached, and snapshot the learning curve as it goes.
#
#   sh training/run_overnight.sh 5000
#
# Each provider runs in its own restart loop. Free tiers fail constantly and in
# ways that kill a process -- quota exhausted, a model retired mid-run, a proxy
# timeout -- so a supervisor that just restarts them is worth more than any
# in-process retry logic. A provider whose key is dead simply burns its restarts
# and stops; the others carry on.
#
# Restarting on exit is not enough: a generator can hang instead of dying, and a
# hung process never exits. Measured -- after the machine slept mid-run, all four
# generators came back holding dead sockets and sat idle for seven hours without
# one of them exiting. So the watchdog below restarts on *silence*: if a
# provider's log has not grown in STALL_AFTER seconds, it is killed and started
# again. Liveness is judged by output, not by the process table.
#
# NOTE ON SLEEP: this keeps nothing awake. Closing the lid on battery triggers
# an unconditional clamshell sleep and everything here freezes with it. Run it
# with the lid open and the charger in:
#     caffeinate -is sh training/run_overnight.sh 5000
#
# Snapshots matter as much as the corpus. Reaching 5000 is worthless if the
# real-40 score went flat at 800, and only a curve over time shows that. Every
# snapshot trains on what exists at that moment and appends one row to
# training/curve.tsv, so the morning question -- "did more data keep helping?"
# -- is answered by a file rather than by another hour of work.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/.venvs/laya/bin/python}"
# Provider keys, one per line: GROQ_KEY=... HF_KEY=... GEMINI_KEY=... AION_KEY=...
# training/keys.env is gitignored; point KEYS_ENV elsewhere to keep it out of the tree.
KEYS="${KEYS_ENV:-$ROOT/training/keys.env}"
TARGET="${1:-5000}"
OUTDIR="$ROOT/training/seeded"
LOGDIR="${LOGDIR:-$ROOT/training/logs}"
CURVE="$ROOT/training/curve.tsv"
SNAPSHOT_EVERY=1800          # seconds
STALL_AFTER=600              # restart a provider whose log has been silent this long
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo /opt/homebrew/opt/ollama/bin/ollama)}"

mkdir -p "$OUTDIR" "$LOGDIR"

# Ollama must be up before its generator starts, and it is the main producer.
if ! curl -s --max-time 3 http://127.0.0.1:11434/api/version > /dev/null 2>&1; then
  if [ -x "$OLLAMA_BIN" ]; then
    echo "starting ollama"
    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 "$OLLAMA_BIN" serve \
      >> "$LOGDIR/ollama-serve.log" 2>&1 &
    sleep 5
  else
    echo "ollama not found at $OLLAMA_BIN -- continuing without it" >&2
  fi
fi
[ -f "$KEYS" ] || { echo "no key file at $KEYS" >&2; exit 1; }
. "$KEYS"

count() { cat "$OUTDIR"/*.jsonl 2>/dev/null | wc -l | tr -d ' '; }

# name, base url, key, model, per-call, sleep. --base-url is used for every
# provider rather than the --provider presets, so one code path covers all four.
run_provider() {
  name="$1"; url="$2"; key="$3"; model="$4"; percall="$5"; pause="$6"
  if [ -z "$key" ]; then echo "  $name skipped (no key)"; return; fi
  ( fails=0
    while [ "$(count)" -lt "$TARGET" ]; do
      "$PY" "$ROOT/training/generate_data.py" --base-url "$url" \
            --api-key "$key" --model "$model" --n 2000 --seed-from-real \
            --out "$OUTDIR/briefs-$name.jsonl" --per-call "$percall" \
            --sleep "$pause" >> "$LOGDIR/$name.log" 2>&1
      rc=$?
      # Exit 2 means the key itself is finished -- no credits, revoked, blocked.
      # No number of restarts changes that.
      if [ "$rc" -eq 2 ]; then
        echo "--- $name: key exhausted or rejected, not restarting $(date '+%H:%M:%S')" >> "$LOGDIR/$name.log"
        break
      fi
      fails=$((fails + 1))
      echo "--- $name exited (restart $fails) $(date '+%H:%M:%S')" >> "$LOGDIR/$name.log"
      # A provider that dies instantly over and over is out of quota or has a
      # dead key; stop hammering it rather than spin all night.
      [ "$fails" -ge 40 ] && { echo "$name: giving up after $fails restarts" >> "$LOGDIR/$name.log"; break; }
      sleep 20
    done ) &
  echo "  $name started (pid $!)"
}

echo "target $TARGET briefs, currently $(count)"
# Ollama first: it is local, so it has no quota, no key and no rate limit, and
# it measured 800 briefs/hour against ~65/hour for the throttled free tiers
# combined. The hosted providers stay in the mix for diversity -- a corpus from
# one model teaches the head that model's habits -- but they are no longer what
# the throughput depends on.
run_provider ollama "http://127.0.0.1:11434/v1"                            "ollama"          "qwen2.5-coder:7b"                   4 0
run_provider groq   "https://api.groq.com/openai/v1"                       "${GROQ_KEY:-}"   "openai/gpt-oss-120b"                4 3
run_provider hf     "https://router.huggingface.co/v1"                     "${HF_KEY:-}"     "Qwen/Qwen3-235B-A22B-Instruct-2507" 4 1
run_provider gemini "https://generativelanguage.googleapis.com/v1beta/openai" "${GEMINI_KEY:-}" "gemini-3.6-flash"                   4 2
run_provider aion   "https://api.aionlabs.ai/v1"                           "${AION_KEY:-}"   "aion-labs/aion-3.0-mini"            3 2

# Watchdog: kill any generator whose log has gone quiet. Its restart loop then
# starts a fresh one, which is the only way a hang gets cleared.
( while [ "$(count)" -lt "$TARGET" ]; do
    sleep 60
    now=$(date +%s)
    for name in ollama groq hf gemini aion; do
      log="$LOGDIR/$name.log"
      [ -f "$log" ] || continue
      last=$(stat -f %m "$log" 2>/dev/null || echo "$now")
      if [ $((now - last)) -ge "$STALL_AFTER" ]; then
        pid=$(pgrep -f "briefs-$name.jsonl" | head -1)
        if [ -n "$pid" ]; then
          echo "--- watchdog: $name silent $((now - last))s, killing $pid $(date '+%H:%M:%S')" >> "$log"
          kill -9 "$pid" 2>/dev/null
        fi
      fi
    done
  done ) &
echo "  watchdog started (pid $!)"

[ -f "$CURVE" ] || printf 'time\tn\tsynthetic_cv\treal_40\n' > "$CURVE"

while [ "$(count)" -lt "$TARGET" ]; do
  sleep "$SNAPSHOT_EVERY"
  n=$(count)
  out=$("$PY" "$ROOT/training/train_head.py" --data "$OUTDIR/*.jsonl" 2>/dev/null)
  cv=$(printf '%s' "$out"  | sed -n 's/.*cross-val on synthetic: *\([0-9]*\)%.*/\1/p' | head -1)
  real=$(printf '%s' "$out" | sed -n 's/.*head, on the REAL 40 *[0-9]*\/[0-9]* *\([0-9]*\)%.*/\1/p' | head -1)
  printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$n" "${cv:-?}" "${real:-?}" >> "$CURVE"
  echo "snapshot n=$n synthetic=${cv:-?}% real=${real:-?}%"
  pgrep -f generate_data.py > /dev/null || { echo "all providers stopped"; break; }
done

echo "finished at $(count) briefs"
