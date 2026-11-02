#!/usr/bin/env bash
# One-time server setup (Ubuntu 22.04/24.04). Run as your normal user:
#   curl -sSL https://raw.githubusercontent.com/<you>/move-alerts/main/setup.sh | bash -s -- https://github.com/<you>/move-alerts.git
set -euo pipefail

REPO_URL="${1:-}"
DIR="$HOME/move-alerts"

echo "==> Installing system packages"
sudo apt-get update -y -qq
sudo apt-get install -y -qq python3-venv python3-pip git

if ! swapon --show | grep -q swapfile; then
  echo "==> Adding 2GB swap (small servers need it for the 500-stock downloads)"
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

if [ ! -d "$DIR/.git" ]; then
  [ -n "$REPO_URL" ] || { echo "Pass your repo URL: bash setup.sh https://github.com/<you>/move-alerts.git"; exit 1; }
  git clone "$REPO_URL" "$DIR"
fi
cd "$DIR"
git pull --ff-only || true

echo "==> Python environment"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

[ -f .env ] || cp env.example .env
chmod 600 .env

echo "==> Services"
for svc in runner listener bot; do
  case $svc in
    runner)   script=run_forever.py ;;
    listener) script=listener.py ;;
    bot)      script=bot.py ;;
  esac
  sudo tee /etc/systemd/system/move-alerts-$svc.service >/dev/null <<EOF
[Unit]
Description=move-alerts $svc
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
EnvironmentFile=$DIR/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=$DIR/.venv/bin/python $DIR/$script
Restart=always
RestartSec=20

[Install]
WantedBy=multi-user.target
EOF
done
sudo systemctl daemon-reload

cat <<'EOF'

==> Setup done. Next:
  1. nano ~/move-alerts/.env            (paste your keys, save with Ctrl+O, Enter, Ctrl+X)
  2. cd ~/move-alerts && set -a && . ./.env && set +a && .venv/bin/python listener.py --login
  3. .venv/bin/python holdings.py --seed "..."     (your holdings)
  4. sudo systemctl enable --now move-alerts-runner move-alerts-listener move-alerts-bot
EOF
