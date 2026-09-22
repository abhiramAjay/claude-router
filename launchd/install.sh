#!/bin/sh
# Install and start the routing sidecar as a login agent.
set -e
PLIST="$HOME/Library/LaunchAgents/com.claude-router.laya-routerd.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cp "$(dirname "$0")/com.claude-router.laya-routerd.plist" "$PLIST"
launchctl bootout "gui/$(id -u)/com.claude-router.laya-routerd" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed. it takes ~20s to load the model, then:"
echo "  curl -s http://127.0.0.1:8129/health"
echo "logs: ~/Library/Logs/laya-routerd.log"
echo "stop: launchctl bootout gui/$(id -u)/com.claude-router.laya-routerd"
