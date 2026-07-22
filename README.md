# Claude Code Telegram Bridge

Remote Claude Code interface via Telegram — text, image, voice, TTS, smart summarization, conversation logging.

**Model:** MiniMax `MiniMax-M2.7-highspeed` via Anthropic-format API at `api.minimax.io`.

## Features

| Input | What happens |
|-------|-------------|
| **Text message** | Claude Code runs, smart status reports stream back |
| **Photo** | MiniMax M2.7 vision (base64) → Claude analyzes |
| **Voice message** | Whisper STT → text → Claude (MiniMax has no native audio input) |
| `/imagine <prompt>` | MiniMax image generation → photo sent back |
| `/speak <text>` | MiniMax TTS → MP3 audio reply |
| Slash commands (`/ask`, `/commit`, etc.) | Claude Code CLI invoked directly |

> Note: video is wired but MiniMax has no native video block. Photo + image gen + TTS are the
> working multimodal surface.

## Smart Summarization

After **10 seconds of silence** from Claude, MiniMax produces a structured status report:

```
**Goal**: What Claude was asked to do
**Decisions Made**: Key choices so far
**Issues / Errors**: Problems encountered (or "None")
**On Track?**: Yes / No / Partially + reason
**Remaining Steps**: What still needs doing
**Needs Steering?**: Yes if stuck or going wrong direction
```

No more terminal floods — check in from anywhere and get an executive summary.

## Quick start

```bash
# 1. Telegram bot
#    Chat with @BotFather → /newbot → copy token

# 2. MiniMax key (Anthropic format, api.minimax.io)
#    Same provider as OpenMAIC core

# 3. Optional: OpenAI key for Whisper voice STT

# 4. Install + run
cp .env.example .env
$EDITOR .env           # fill in TELEGRAM_BOT_TOKEN + MINIMAX_API_KEY
./run.sh               # foreground — for systemd, see below
```

## systemd (recommended for server)

```bash
systemctl --user enable --now claude-bot
journalctl --user -u claude-bot -f
```

The unit at `~/.config/systemd/user/claude-bot.service` reads `.env` via
`EnvironmentFile`, runs in the project venv, restarts on crash, lingers across
logout (requires `loginctl enable-linger Hermes`).

## Environment variables

| Var | Required | Default | Description |
|-----|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | yes | — | Telegram bot token |
| `MINIMAX_API_KEY` | yes | — | MiniMax Anthropic-format key |
| `MINIMAX_MODEL` | no | `MiniMax-M2.7-highspeed` | Model for chat / vision / summarizer |
| `MIN_TOKENS` | no | `8000` | max_tokens for M2.7 (thinking eats 400-800) |
| `WHISPER_API_KEY` / `OPENAI_API_KEY` | for voice | — | Whisper-1 STT |
| `CLAUDE_COMMAND` | no | `claude` | Path to claude CLI |
| `STREAM_TIMEOUT` | no | `10` | Seconds of silence before summarization |
| `TTS_ENABLED` | no | `true` | |
| `TTS_VOICE` | no | `English_Graceful_Lady` | MiniMax voice ID |

## Architecture

```
Telegram message
    │
    ├─ text prompt ──────────────────────────────→ claude CLI subprocess
    ├─ photo (base64) ──→ MiniMax M2.7 vision ───→ claude CLI subprocess
    ├─ voice ───────────→ Whisper STT ──────────→ claude CLI subprocess
    └─ /imagine ───────→ MiniMax image gen ─────→ photo reply
         /speak ───────→ MiniMax TTS ───────────→ audio reply

claude output ──[10s silence]──→ MiniMax summarizer ──→ status report → Telegram
                              ↓
                     SQLite conversation log
```

## Known M2.7 thinking behavior

`MiniMax-M2.7-highspeed` emits a `thinking` content block before any `text`
output (~400-800 tokens of internal reasoning). Implications:

- `max_tokens` must be ≥ 4000 or the text gets cut off mid-thinking
- Bot strips the thinking block from responses (only text is sent to user)
- The thinking block IS visible in summarizer calls (sent as a separate system
  prompt, not interleaved)

