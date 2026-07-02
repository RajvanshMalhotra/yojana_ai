#!/bin/bash
# Pull latest code and restart. Run from /home/ubuntu/yojan_ai.
set -e

echo "==> Pulling latest"
git pull origin main

echo "==> Installing any new Python deps"
source venv/bin/activate
pip install -r requirements.txt -q

echo "==> Rebuilding frontend"
cd frontend && npm install --silent && npm run build && cd ..

echo "==> Restarting service"
sudo systemctl restart yojan
sudo systemctl status yojan --no-pager
