#!/usr/bin/env python3
"""
Telegram + MiniMax bot that bridges messages to Claude Code CLI.
Features: audio/image input, TTS output, smart summarization, conversation logging.

Setup:
1. Create a bot via @BotFather, get the token
2. Set env vars ( TELEGRAM_BOT_TOKEN, MINIMAX_API_KEY)
3. Run: python3 telegram-claude-channel.py
"""

import os
import re
import io
import json
import time
import asyncio
import logging
import sqlite3
import subprocess
import base64
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    ContextTypes,
    CommandHandler,
)
from telegram.constants import ParseMode

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BOT_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN")
MINIMAX_API_KEY  = os.getenv("MINIMAX_API_KEY")

# MiniMax endpoints — Anthropic-format (M2.7) is what works on api.minimax.io
MX_BASE    = "https://api.minimax.io"
MX_CHAT    = f"{MX_BASE}/anthropic/v1/messages"   # text + vision + video via Anthropic format
MX_TTS     = f"{MX_BASE}/v1/t2a_v2"
MX_IMAGE   = f"{MX_BASE}/v1/image_generation"
MX_MODEL   = os.getenv("MINIMAX_MODEL", "MiniMax-M3")
# M2.7-highspeed thinks heavily (400-800 token thinking block per call).
# M3 has NO thinking block — returns text directly, ~10x faster.
# Default max_tokens is generous; M3 uses way less, M2.7 needs the headroom.
MIN_TOKENS = int(os.getenv("MIN_TOKENS", "8000"))

# Optional: OpenAI-compatible Whisper for STT (MiniMax has no STT)
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")  # or set OPENAI_API_KEY
WHISPER_BASE    = os.getenv("WHISPER_BASE", "https://api.openai.com/v1")

CLAUDE_COMMAND = os.getenv("CLAUDE_COMMAND", "claude")
STREAM_TIMEOUT = int(os.getenv("STREAM_TIMEOUT", "10"))   # silence secs before summarizing
MAX_MSG_LEN    = 4096
TTS_ENABLED    = os.getenv("TTS_ENABLED", "true").lower() == "true"
TTS_VOICE      = os.getenv("TTS_VOICE", "English_Graceful_Lady")
DB_PATH        = Path(__file__).parent / "conversations.db"

# ─────────────────────────────────────────────────────────────
# MiniMax API helpers
# ─────────────────────────────────────────────────────────────

def mx_post(url: str, payload: dict, stream: bool = False) -> requests.Response:
    """OpenAI-compatible POST (TTS, image gen, etc.) — uses Bearer auth."""
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    return requests.post(url, headers=headers, json=payload, timeout=60, stream=stream)


def mx_anthropic_post(url: str, payload: dict) -> requests.Response:
    """Anthropic-format POST (chat / vision / video) — uses X-Api-Key."""
    headers = {
        "X-Api-Key": MINIMAX_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    return requests.post(url, headers=headers, json=payload, timeout=120)


def _flatten_content(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI-style messages (text OR content-list with image_url/video_url)
    into Anthropic-format messages (content-list with text/image/video blocks).

    The bot's call sites use OpenAI format:
        {"role": "user", "content": "..."}
        {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}

    Anthropic wants:
        {"role": "user", "content": [{"type": "text", ...}, {"type": "image", "source": {...}}]}

    We map image_url→image (with base64 source) and video_url→... Anthropic doesn't have
    a native video block, so we send a placeholder text + a request to the caller to
    pre-process. For now video passthrough keeps the URL as a hint.
    """
    out = []
    for m in messages:
        role = m["role"]
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            new_blocks = []
            for blk in content:
                btype = blk.get("type")
                if btype == "text":
                    new_blocks.append({"type": "text", "text": blk.get("text", "")})
                elif btype == "image_url":
                    url = blk.get("image_url", {}).get("url", "")
                    # Extract base64 data URI: data:image/jpeg;base64,XXXX
                    if url.startswith("data:") and ";base64," in url:
                        header, b64 = url.split(";base64,", 1)
                        media_type = header.split(":", 1)[1] or "image/jpeg"
                        new_blocks.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        })
                    else:
                        # URL — Anthropic wants base64 source; treat as text fallback
                        new_blocks.append({"type": "text", "text": f"[image URL: {url}]"})
                elif btype == "video_url":
                    # No native video block in Anthropic format — extract description hint
                    new_blocks.append({"type": "text", "text": "[video content attached — describe from base64 separately]"})
                else:
                    new_blocks.append(blk)
            out.append({"role": role, "content": new_blocks})
        else:
            out.append(m)
    return out


def mx_chat(
    messages: list[dict],
    model: str = None,
    max_tokens: int | None = None,
    system: str | None = None,
) -> str:
    """
    Chat via MiniMax Anthropic-format endpoint. M2.7 emits long thinking blocks
    before text, so we strip thinking and return only the text content.
    """
    if model is None:
        model = MX_MODEL
    if max_tokens is None:
        max_tokens = MIN_TOKENS
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": _flatten_content(messages),
    }
    if system:
        payload["system"] = system
    resp = mx_anthropic_post(MX_CHAT, payload)
    resp.raise_for_status()
    data = resp.json()
    blocks = data.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if not text:
        # All thinking, no actual answer — surface so caller knows
        thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        return f"(model produced no text — thinking: {thinking[:300]})"
    return text


def mx_tts(text: str, voice: str = TTS_VOICE) -> bytes:
    """
    Convert text to speech via MiniMax TTS v2.
    Returns raw audio bytes (mp3).
    """
    resp = mx_post(MX_TTS, {
        "model": "speech-02-hd",
        "text": text[:2000],   # safety truncate
        "stream": False,
        "voice_setting": {
            "voice_id": voice,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
        "output_format": "url",   # get a URL back to download
    })
    resp.raise_for_status()
    data = resp.json()

    # Response: { "data": { "audio": "<url>", "status": 2, ... } }
    # Field name is 'audio' (not 'audio_url' as the original code assumed).
    audio_url = data.get("data", {}).get("audio") or data.get("data", {}).get("audio_url")
    if not audio_url:
        raise ValueError(f"No audio URL in TTS response: {data}")

    # Download the audio
    audio_resp = requests.get(audio_url, timeout=30)
    audio_resp.raise_for_status()
    return audio_resp.content


def mx_image(prompt: str, aspect_ratio: str = "16:9") -> bytes:
    """
    Generate image via MiniMax text-to-image.
    Returns decoded image bytes.
    """
    resp = mx_post(MX_IMAGE, {
        "model": "image-01",
        "prompt": prompt[:1500],
        "aspect_ratio": aspect_ratio,
        "response_format": "base64",
    })
    resp.raise_for_status()
    data = resp.json()
    images = data.get("data", {}).get("image_base64") or []
    if not images:
        raise ValueError(f"No image in response: {data}")
    return base64.b64decode(images[0])


# ─────────────────────────────────────────────────────────────
# STT: use Whisper (OpenAI-compatible) — MiniMax has no STT
# ─────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    """
    Transcribe audio using Whisper API.
    Set WHISPER_API_KEY or OPENAI_API_KEY env var.
    Falls back to OpenAI if WHISPER_API_KEY not set.
    """
    api_key = WHISPER_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No WHISPER_API_KEY or OPENAI_API_KEY set — "
            "voice transcription requires a Whisper API key"
        )

    base = WHISPER_BASE or "https://api.openai.com/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (filename, audio_bytes, "audio/ogg")}
    data = {"model": "whisper-1", "language": "en"}

    resp = requests.post(
        f"{base}/audio/transcriptions",
        headers=headers,
        data=data,
        files=files,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["text"]


# ─────────────────────────────────────────────────────────────
# Summarizer
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a monitoring assistant for a Claude Code CLI session.
Your job is to produce a brief, structured status report from the output so far.
Be concise and direct. Write for a human who can't watch the terminal."""

USER_SUMMARIZER_TEMPLATE = """## Goal
{goal}

## Output so far
{output}

---

Produce a status report with exactly these sections (keep each section 1-3 sentences/bullets):

**Goal**: What Claude was asked to do
**Decisions Made**: Key choices made so far (max 3 bullets)
**Issues / Errors**: Problems encountered (or "None")
**On Track?**: Yes / No / Partially + reason
**Remaining Steps**: What still needs doing (max 3 bullets)
**Needs Steering?**: Yes if stuck, going wrong direction, or making no progress; No otherwise"""


async def summarize_output(output_lines: list[str], goal: str) -> str:
    if not output_lines:
        return "No output yet."
    if len(output_lines) < 3:
        return "".join(output_lines)

    output_text = "".join(output_lines[-200:])   # last 200 lines to avoid token limit
    user_prompt = USER_SUMMARIZER_TEMPLATE.format(
        goal=goal,
        output=output_text,
    )

    try:
        summary = mx_chat(
            [
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4000,
            system=SYSTEM_PROMPT,
        )
        return summary
    except Exception as e:
        return f"(summarizer error: {e})\n\n---\nRaw output:\n" + "".join(output_lines[-30:])


# ─────────────────────────────────────────────────────────────
# Conversation DB
# ─────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role    TEXT,
            content TEXT,
            media   TEXT DEFAULT 'text',
            ts      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id   INTEGER,
            goal      TEXT,
            started   DATETIME DEFAULT CURRENT_TIMESTAMP,
            ended     DATETIME,
            status    TEXT DEFAULT 'active'
        )
    """)
    conn.commit()
    conn.close()


def log_msg(chat_id: int, role: str, content: str, media: str = "text"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, media) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, media),
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Claude runner with silence-triggered summarization
# ─────────────────────────────────────────────────────────────

class ClaudeRunner:
    def __init__(self, chat_id: int, goal: str):
        self.chat_id = chat_id
        self.goal    = goal
        self.output_lines: list[str] = []
        self.last_time = time.monotonic()
        self.summarize_task: asyncio.Task | None = None
        self.finished  = False
        self._cancel   = asyncio.Event()

    async def run(self, args: list[str], send_fn):
        """
        Run claude; after STREAM_TIMEOUT seconds of silence, ask MiniMax
        to summarize status and send it via send_fn.
        send_fn: async fn(text: str) -> None
        """
        self._cancel.clear()
        self.finished = False

        process = await asyncio.create_subprocess_exec(
            CLAUDE_COMMAND, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "NO_COLOR": "1"},
        )

        async def on_line(line: str):
            self.output_lines.append(line)
            self.last_time = time.monotonic()
            # Reschedule summarization on new output
            self._schedule_summarize(send_fn)

        async def on_done():
            self.finished = True
            self._cancel.set()
            log_msg(self.chat_id, "claude", "".join(self.output_lines))
            if self.output_lines:
                await send_fn("--- Claude finished ---\n")
                summary = await summarize_output(self.output_lines, self.goal)
                await send_fn(summary)

        await self._read_stream(process, on_line, on_done)

    def _schedule_summarize(self, send_fn):
        if self.summarize_task and not self.summarize_task.done():
            return
        self.summarize_task = asyncio.create_task(
            self._wait_summarize(send_fn)
        )

    async def _wait_summarize(self, send_fn):
        sleeper = asyncio.create_task(asyncio.wait_for(
            self._cancel.wait(), timeout=STREAM_TIMEOUT
        ))
        try:
            await sleeper
        except asyncio.TimeoutError:
            if not self.finished and self.output_lines:
                await send_fn(
                    f"⏳ *Status update* (silent for {STREAM_TIMEOUT}s):\n"
                )
                summary = await summarize_output(self.output_lines, self.goal)
                await send_fn(summary)
                self._schedule_summarize(send_fn)  # schedule next

    async def _read_stream(self, process, on_line, on_done):
        empty = 0
        reader = asyncio.create_task(process.stdout.readline())

        while True:
            done, _ = await asyncio.wait(
                {reader}, timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            if done:
                line = reader.result()
                if line:
                    empty = 0
                    await on_line(line.decode("utf-8", errors="replace"))
                    reader = asyncio.create_task(process.stdout.readline())
                else:
                    empty += 1
                    if empty > 3 or process.returncode is not None:
                        break
                    reader = asyncio.create_task(process.stdout.readline())
            else:
                if process.returncode is not None:
                    break

        try:
            remaining = await process.stdout.read()
            if remaining:
                await on_line(remaining.decode("utf-8", errors="replace"))
        except Exception:
            pass

        await on_done()


# ─────────────────────────────────────────────────────────────
# Per-chat active run tracking
# ─────────────────────────────────────────────────────────────
_active_runs: dict[int, ClaudeRunner] = {}
_run_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────
# Telegram handlers
# ─────────────────────────────────────────────────────────────

async def send_long(
    update: Update,
    text: str,
    quote: bool = True,
    parse_mode: str | None = ParseMode.MARKDOWN,
):
    for i in range(0, len(text), MAX_MSG_LEN):
        chunk = text[i : i + MAX_MSG_LEN]
        try:
            await update.message.reply_text(chunk, do_quote=quote, parse_mode=parse_mode)
        except Exception:
            await update.message.reply_text(chunk, do_quote=quote, parse_mode=None)


async def claude_runner_handler(
    update: Update,
    goal: str,
    args: list[str],
    media: str = "text",
):
    """Shared logic for text prompts and slash commands."""
    chat_id = update.message.chat_id

    log_msg(chat_id, "user", goal, media)
    await update.message.reply_text("🤖 Starting Claude...", do_quote=True)

    runner = ClaudeRunner(chat_id, goal)

    async with _run_lock:
        _active_runs[chat_id] = runner

    try:
        await runner.run(args, lambda msg: update.message.reply_text(
            msg, do_quote=False, parse_mode=ParseMode.MARKDOWN
        ))
    finally:
        async with _run_lock:
            _active_runs.pop(chat_id, None)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text.strip()
    if not prompt or prompt.startswith("/"):
        return
    await claude_runner_handler(update, prompt, [prompt])


async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = update.message.text.strip().lstrip("/")
    parts = raw.split(None, 1)
    if not parts:
        return
    cmd, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    prompt = f"/{cmd}" + (f" {rest}" if rest else "")
    args = [cmd] + ([rest] if rest else [])
    await claude_runner_handler(update, prompt, args)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Voice → Whisper STT → Claude.

    Note: MiniMax M3 supports image+video natively but not audio input yet,
    so we transcribe with Whisper first then feed text to Claude.
    Set WHISPER_API_KEY or OPENAI_API_KEY to enable."""
    await update.message.reply_text("🎙 Transcribing...", do_quote=True)

    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        ogg_bytes  = await voice_file.download_as_bytearray()
        text = transcribe_audio(bytes(ogg_bytes))
    except ValueError as e:
        await update.message.reply_text(str(e), do_quote=True)
        return
    except Exception as e:
        await update.message.reply_text(f"Transcription error: {e}", do_quote=True)
        return

    if not text.strip():
        await update.message.reply_text("(no speech detected)", do_quote=True)
        return

    log_msg(update.message.chat_id, "user", f"[voice]: {text}", "audio")
    await update.message.reply_text(f"🎙: {text}", do_quote=True)
    await claude_runner_handler(update, text, [text], media="audio")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Video → MiniMax multimodal understanding → Claude."""
    if not MINIMAX_API_KEY:
        await update.message.reply_text("MINIMAX_API_KEY not set", do_quote=True)
        return

    await update.message.reply_text("🎬 Analyzing video...", do_quote=True)

    try:
        video_file = await context.bot.get_file(update.message.video.file_id)
        vid_bytes  = await video_file.download_as_bytearray()
        vid_b64    = base64.b64encode(bytes(vid_bytes)).decode()

        description = mx_chat([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video in detail — what happens, what is shown, any text visible."},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{vid_b64}",
                        "detail": "default",
                    },
                },
            ],
        }], max_tokens=4000)

        log_msg(update.message.chat_id, "user", f"[video]: {description}", "video")
        await update.message.reply_text(f"🎬: {description}", do_quote=True)
        await claude_runner_handler(
            update,
            f"Analyze this video: {description}",
            [f"Analyze this video: {description}"],
            media="video",
        )

    except Exception as e:
        await update.message.reply_text(f"Video analysis error: {e}", do_quote=True)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Photo → MiniMax vision (base64) → Claude."""
    if not MINIMAX_API_KEY:
        await update.message.reply_text("MINIMAX_API_KEY not set", do_quote=True)
        return

    await update.message.reply_text("🖼 Analyzing image...", do_quote=True)

    try:
        photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
        img_bytes  = await photo_file.download_as_bytearray()
        img_b64    = base64.b64encode(bytes(img_bytes)).decode()

        # OpenAI-compatible format: data URI with base64 image
        description = mx_chat([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "high",
                    },
                },
            ],
        }], max_tokens=4000)

        log_msg(update.message.chat_id, "user", f"[image]: {description}", "image")
        await update.message.reply_text(f"🖼: {description}", do_quote=True)
        await claude_runner_handler(
            update,
            f"Analyze this image: {description}",
            [f"Analyze this image: {description}"],
            media="image",
        )

    except Exception as e:
        await update.message.reply_text(f"Image analysis error: {e}", do_quote=True)


async def handle_image_gen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/imagine <prompt> — generate image via MiniMax and send back."""
    prompt = update.message.text.replace("/imagine", "").replace("/image", "").strip()
    if not prompt:
        await update.message.reply_text("Usage: `/imagine <prompt>`", do_quote=True)
        return

    await update.message.reply_text("🎨 Generating...", do_quote=True)

    try:
        img_bytes = mx_image(prompt)
        bio = io.BytesIO(img_bytes)
        bio.name = "generated.png"
        await update.message.reply_photo(photo=bio, do_quote=True)
        log_msg(update.message.chat_id, "bot", f"[image generated: {prompt}]", "image")
    except Exception as e:
        await update.message.reply_text(f"Image gen error: {e}", do_quote=True)


async def handle_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/speak <text> — convert text to speech via MiniMax TTS."""
    if not TTS_ENABLED or not MINIMAX_API_KEY:
        await update.message.reply_text("TTS not configured", do_quote=True)
        return

    text = update.message.text.replace("/speak", "").replace("/tts", "").strip()
    if not text:
        await update.message.reply_text("Usage: `/speak <text>`", do_quote=True)
        return

    await update.message.reply_text("🔊 Generating audio...", do_quote=True)

    try:
        mp3_bytes = mx_tts(text)
        bio = io.BytesIO(mp3_bytes)
        bio.name = "speech.mp3"
        await update.message.reply_audio(audio=bio, do_quote=True)
    except Exception as e:
        await update.message.reply_text(f"TTS error: {e}", do_quote=True)


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show recent messages from DB."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, media, ts FROM messages "
        "WHERE chat_id = ? ORDER BY id DESC LIMIT 20",
        (update.message.chat_id,),
    ).fetchall()
    conn.close()

    if not rows:
        await update.message.reply_text("No history yet.", do_quote=True)
        return

    lines = ["📋 *Recent messages:*\n"]
    for role, content, media, ts in reversed(rows):
        prefix = "👤" if role == "user" else "🤖"
        short  = content[:100] + ("..." if len(content) > 100 else "")
        lines.append(f"{prefix} [{ts[:16]}] {short}")

    await send_long(update, "\n".join(lines), parse_mode=None)


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with _run_lock:
        runner = _active_runs.get(update.message.chat_id)
    if runner:
        await update.message.reply_text(
            "⚠️ Cancellation requested — Claude will try to stop.",
            do_quote=True,
        )
        # asyncio subprocess can't be killed cleanly, but we signal
        runner._cancel.set()
    else:
        await update.message.reply_text("No active Claude process.", do_quote=True)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Claude Code Telegram Bridge*\n\n"
        "📝 Send a message → Claude Code runs, status reports stream back\n"
        "🖼 Send a photo → MiniMax M2.7 vision describes it, Claude analyzes\n"
        "🎬 Send a video → handled via base64 description, Claude analyzes (experimental)\n"
        "🎙 Send voice → transcribed via Whisper, Claude responds\n"
        "🎨 `/imagine <prompt>` → MiniMax image generation\n"
        "🔊 `/speak <text>` → MiniMax TTS audio reply\n\n"
        "*Smart summarization:* After 10s of silence, MiniMax summarizes:\n"
        "goal, decisions made, remaining steps, issues, on track?\n\n"
        "*Commands:*\n"
        "`/ask <q>` — ask Claude\n"
        "`/status` — show conversation log\n"
        "`/cancel` — try to stop current run\n"
        "`/imagine <prompt>` — generate image\n"
        "`/speak <text>` — text-to-speech\n"
    )
    await update.message.reply_text(text, do_quote=True, parse_mode=ParseMode.MARKDOWN)


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

def main():
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set — get one from @BotFather")
        return

    if not MINIMAX_API_KEY:
        print("WARN: MINIMAX_API_KEY not set — TTS, vision, and image gen disabled")

    init_db()

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler(["help", "start"],  handle_help))
    app.add_handler(CommandHandler(["status"],         handle_status))
    app.add_handler(CommandHandler(["cancel"],         handle_cancel))
    app.add_handler(CommandHandler(["imagine", "image"], handle_image_gen))
    app.add_handler(CommandHandler(["speak", "tts"],  handle_tts))
    app.add_handler(MessageHandler(filters.VIDEO,     handle_video))
    app.add_handler(MessageHandler(filters.PHOTO,     handle_photo))
    app.add_handler(MessageHandler(filters.VOICE,     handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^/"), handle_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot starting...")
    print(f"  STREAM_TIMEOUT = {STREAM_TIMEOUT}s")
    print(f"  TTS_ENABLED   = {TTS_ENABLED}  (voice: {TTS_VOICE})")
    print(f"  DB             = {DB_PATH}")
    print(f"  STT            = {'Whisper (OpenAI-compatible)' if os.getenv('WHISPER_API_KEY') or os.getenv('OPENAI_API_KEY') else 'NOT CONFIGURED — set WHISPER_API_KEY or OPENAI_API_KEY'}")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
