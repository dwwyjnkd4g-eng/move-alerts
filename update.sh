#!/usr/bin/env bash
# Pull the latest code from GitHub and restart everything:  bash ~/move-alerts/update.sh
set -euo pipefail
cd "$HOME/move-alerts"
git pull --ff-only
.venv/bin/pip install -q -r requirements.txt
sudo systemctl restart move-alerts-runner move-alerts-listener move-alerts-bot
sleep 3
systemctl --no-pager --lines=0 status move-alerts-runner move-alerts-listener move-alerts-bot | grep -E "●|Active"
