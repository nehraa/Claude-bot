#!/bin/bash
# Start Claude-bot. Sources .env, activates venv, runs the bot.
# Use this for foreground testing. For production, use:
#   systemctl --user start claude-bot
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example and fill in TELEGRAM_BOT_TOKEN." >&2
    exit 1
fi

if grep -q "PASTE_.*n" .env; then
    echo "ERROR: TELEGRAM_BOT_TOKEN still has placeholder. Edit .env first." >&2
    exit 1
fi

if [ ! -d venv ]; then
    echo "Creating venv..."
    python3 -m venv venv
    ./venv/bin/pip install -q -r requirements.txt
fi

set -a
source .env
set +a

exec ./venv/bin/python3 telegram-claude-channel.py
