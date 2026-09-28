#!/bin/sh
# Install and start the routing sidecar as a macOS login agent.
#
#   sh launchd/install.sh                      # uses ~/.venvs/laya
#   LAYA_VENV=/path/to/venv sh launchd/install.sh
set -e
LABEL="com.claude-router.laya-routerd"
VENV="${LAYA_VENV:-$HOME/.venvs/laya}"
BIN="$VENV/bin/laya-routerd"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

[ -x "$BIN" ] || { echo "no laya-routerd at $BIN -- pip install '.[server]' into that venv first, or set LAYA_VENV" >&2; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s#__LABEL__#$LABEL#" -e "s#__BIN__#$BIN#" -e "s#__HOME__#$HOME#g" \
    "$(dirname "$0")/laya-routerd.plist.template" > "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed. it takes ~20s to load the model, then:"
echo "  curl -s http://127.0.0.1:8129/health"
echo "logs: ~/Library/Logs/laya-routerd.log"
echo "stop: sh $(dirname "$0")/uninstall.sh"
