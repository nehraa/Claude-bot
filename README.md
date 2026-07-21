# Claude Code Telegram Bridge

Remote Claude Code interface via Telegram — with voice, vision, video understanding, TTS, smart summarization, and conversation logging powered by MiniMax M3.

## Features

| Input | What happens |
|-------|-------------|
| **Text message** | Claude Code runs, smart status reports stream back |
| **Photo** | MiniMax M3 vision (base64) → Claude analyzes |
| **Video** | MiniMax M3 multimodal (base64) → Claude analyzes |
| **Voice message** | Whisper STT → text → Claude (MiniMax M3 has no audio input yet) |
| `/imagine <prompt>` | MiniMax image generation → photo sent back |
| `/speak <text>` | MiniMax TTS → MP3 audio reply |
| Slash commands (`/ask`, `/commit`, etc.) | Claude Code CLI invoked directly |

## Smart Summarization

After **10 seconds of silence** from Claude, MiniMax M3 produces a structured status report:

```
**Goal**: What Claude was asked to do
**Decisions Made**: Key choices so far
**Issues / Errors**: Problems encountered (or "None")
**On Track?**: Yes / No / Partially + reason
**Remaining Steps**: What still needs doing
**Needs Steering?**: Yes if stuck or going wrong direction
```

No more terminal floods — check in from anywhere and get an executive summary.

## Setup

### 1. Telegram Bot
1. Chat with **@BotFather** on Telegram
2. Send `/newbot`, follow prompts
3. Copy the bot token

### 2. MiniMax API Key
Get from [platform.minimax.io](https://platform.minimax.io) → API Keys

### 3. (Optional) Whisper for Voice
MiniMax M3 supports image + video natively but not audio input yet. Set either:
```bash
export WHISPER_API_KEY="your_openai_key"   # or
export OPENAI_API_KEY="your_openai_key"    # Whisper-1 used automatically
```

### 4. Run

```bash
pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN="your:telegram_token"
export MINIMAX_API_KEY="your:minimax_key"
export MINIMAX_MODEL="MiniMax-M3"   # default — change if needed

python3 telegram-claude-channel.py
```

### 5. tmux (recommended for server)

```bash
tmux new -s claude-bot
# run the above + python command
# Detach: Ctrl+B, D
# Rejoin: tmux attach -t claude-bot
```

## Environment Variables

| Var | Required | Default | Description |
|-----|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Telegram bot token |
| `MINIMAX_API_KEY` | Yes | — | MiniMax API key |
| `MINIMAX_MODEL` | No | `MiniMax-M3` | Model for chat (M3 = multimodal) |
| `WHISPER_API_KEY` | For voice | — | OpenAI key for Whisper STT |
| `OPENAI_API_KEY` | For voice | — | Fallback if WHISPER_API_KEY not set |
| `CLAUDE_COMMAND` | No | `claude` | Path to claude CLI |
| `STREAM_TIMEOUT` | No | `10` | Seconds of silence before summarization |
| `TTS_ENABLED` | No | `true` | Enable TTS |
| `TTS_VOICE` | No | `English_Graceful_Lady` | MiniMax voice ID for TTS |

## Architecture

```
Telegram message
    │
    ├─ text prompt ──────────────────────────────→ claude CLI subprocess
    ├─ photo (base64) ──→ MiniMax M3 vision ────→ claude CLI subprocess
    ├─ video (base64) ──→ MiniMax M3 multimodal → claude CLI subprocess
    ├─ voice ───────────→ Whisper STT ──────────→ claude CLI subprocess
    └─ /imagine ───────→ MiniMax image gen ─────→ photo reply
         /speak ───────→ MiniMax TTS ───────────→ audio reply

claude output ──[10s silence]──→ MiniMax summarizer ──→ status report → Telegram
                              ↓
                     SQLite conversation log
```
