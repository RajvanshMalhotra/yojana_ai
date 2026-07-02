#!/bin/bash
# Run once on a fresh Ubuntu 22.04 t3.small to set up Yojan AI.
# Usage: bash setup.sh
set -e

APP_DIR="/home/ubuntu/yojan_ai"
REPO="https://github.com/RajvanshMalhotra/yojana_ai.git"

echo "==> Updating system"
sudo apt-get update -y && sudo apt-get upgrade -y

echo "==> Installing system packages"
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
    build-essential nginx git curl

echo "==> Installing Node.js 20"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

echo "==> Cloning repo"
git clone "$REPO" "$APP_DIR"
cd "$APP_DIR"

echo "==> Creating Python venv and installing deps"
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pipecat-ai[silero,websocket] sarvamai groq

echo "==> Downloading NLTK data"
python3 -c "import nltk; nltk.download('stopwords')"

echo "==> Building frontend"
cd frontend
npm install
npm run build
cd ..

echo "==> Creating .env — fill in your keys"
cat > .env <<'ENV'
PINECONE_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
GROQ_API_KEYS=
FIRECRAWL_API_KEY=
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
SARVAM_API_KEY=
RUMIK_API_KEY=
ENV

echo "==> Installing systemd service"
sudo cp deploy/yojan.service /etc/systemd/system/yojan.service
sudo systemctl daemon-reload
sudo systemctl enable yojan

echo "==> Installing nginx config"
sudo cp deploy/nginx.conf /etc/nginx/sites-available/yojan
sudo ln -sf /etc/nginx/sites-available/yojan /etc/nginx/sites-enabled/yojan
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo ""
echo "Done! Next steps:"
echo "  1. Fill in /home/ubuntu/yojan_ai/.env with your API keys"
echo "  2. rsync your data/ directory:  rsync -avz data/ ubuntu@<IP>:~/yojan_ai/data/"
echo "  3. sudo systemctl start yojan"
echo "  4. sudo systemctl status yojan"
