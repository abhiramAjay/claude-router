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
# Snapshots matter as much as the corpus. Reaching 5000 is worthless if the
# real-40 score went flat at 800, and only a curve over time shows that. Every
# snapshot trains on what exists at that moment and appends one row to
# training/curve.tsv, so the morning question -- "did more data keep helping?"
# -- is answered by a file rather than by another hour of work.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HOME/.venvs/laya/bin/python"
KEYS="${KEYS_ENV:-/private/tmp/claude-501/-Users-dev-IdeaProjects-claude-router/88366300-54cc-433c-a733-ff18ab6db560/scratchpad/keys.env}"
TARGET="${1:-5000}"
OUTDIR="$ROOT/training/seeded"
LOGDIR="${LOGDIR:-$ROOT/training/logs}"
CURVE="$ROOT/training/curve.tsv"
SNAPSHOT_EVERY=1800          # seconds

mkdir -p "$OUTDIR" "$LOGDIR"
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
run_provider groq   "https://api.groq.com/openai/v1"                       "${GROQ_KEY:-}"   "openai/gpt-oss-120b"                4 3
run_provider hf     "https://router.huggingface.co/v1"                     "${HF_KEY:-}"     "Qwen/Qwen3-235B-A22B-Instruct-2507" 4 1
run_provider gemini "https://generativelanguage.googleapis.com/v1beta/openai" "${GEMINI_KEY:-}" "gemini-3.6-flash"                   4 2
run_provider aion   "https://api.aionlabs.ai/v1"                           "${AION_KEY:-}"   "aion-labs/aion-3.0-mini"            3 2

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
