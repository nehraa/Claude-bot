#!/usr/bin/env python3
"""
Claude-bot v2 — Persistent Claude session per Telegram chat.

Architecture:
- One long-running `claude` subprocess per Telegram chat (chat_id → session)
- Stream JSON in/out via stdin/stdout (--input-format stream-json --output-format stream-json)
- User messages → JSON injected to claude stdin → responses streamed back to Telegram
- Steering Manager: every N claude responses, M3 summarizes state into a checkpoint.
  If "needs steering", bot asks user via Telegram; user reply is injected as next prompt.
- All events logged to JSONL for future training data.

Setup:
1. Set env vars (TELEGRAM_BOT_TOKEN, MINIMAX_API_KEY)
2. Run: python3 telegram-claude-channel.py
"""

import os
import re
import io
import json
import time
import uuid
import asyncio
import logging
import sqlite3
import subprocess
import base64
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Callable

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

MX_BASE    = "https://api.minimax.io"
MX_CHAT    = f"{MX_BASE}/anthropic/v1/messages"   # for steering manager
MX_TTS     = f"{MX_BASE}/v1/t2a_v2"
MX_IMAGE   = f"{MX_BASE}/v1/image_generation"
MX_MODEL   = os.getenv("MINIMAX_MODEL", "MiniMax-M3")
MIN_TOKENS = int(os.getenv("MIN_TOKENS", "4000"))

CLAUDE_COMMAND = os.getenv("CLAUDE_COMMAND", "claude")
CLAUDE_MODEL    = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
CLAUDE_EXTRA_ARGS = os.getenv("CLAUDE_EXTRA_ARGS", "--bare --verbose --dangerously-skip-permissions")

CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "10"))   # checkpoint after N responses
SILENT_FOR_STEER = int(os.getenv("SILENT_FOR_STEER", "60"))   # sec of no output before asking user
MAX_MSG_LEN      = 4096
DB_PATH          = Path(__file__).parent / "conversations.db"
LOG_DIR          = Path(__file__).parent / "data" / "training_log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR      = Path(__file__).parent / "data" / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MiniMax API (steering manager)
# ─────────────────────────────────────────────────────────────

def mx_anthropic_post(url: str, payload: dict) -> requests.Response:
    headers = {
        "X-Api-Key": MINIMAX_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    return requests.post(url, headers=headers, json=payload, timeout=120)


def mx_chat(messages: list[dict], system: str | None = None, max_tokens: int | None = None) -> str:
    if max_tokens is None:
        max_tokens = MIN_TOKENS
    payload = {
        "model": MX_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    resp = mx_anthropic_post(MX_CHAT, payload)
    resp.raise_for_status()
    data = resp.json()
    blocks = data.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if not text:
        thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        return f"(no text — thinking: {thinking[:200]})"
    return text


def mx_tts(text: str, voice: str = "English_Graceful_Lady") -> bytes:
    resp = requests.post(
        MX_TTS,
        headers={"Authorization": f"Bearer {MINIMAX_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "speech-02-hd",
            "text": text[:2000],
            "stream": False,
            "voice_setting": {"voice_id": voice, "speed": 1.0, "vol": 1.0, "pitch": 0},
            "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1},
            "output_format": "url",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    audio_url = data.get("data", {}).get("audio") or data.get("data", {}).get("audio_url")
    if not audio_url:
        raise ValueError(f"No audio URL in TTS response: {data}")
    return requests.get(audio_url, timeout=30).content


def mx_image(prompt: str, aspect_ratio: str = "16:9") -> bytes:
    resp = requests.post(
        MX_IMAGE,
        headers={"Authorization": f"Bearer {MINIMAX_API_KEY}", "Content-Type": "application/json"},
        json={"model": "image-01", "prompt": prompt[:1500], "aspect_ratio": aspect_ratio, "response_format": "base64"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    images = data.get("data", {}).get("image_base64") or []
    if not images:
        raise ValueError(f"No image in response: {data}")
    return base64.b64decode(images[0])


# ─────────────────────────────────────────────────────────────
# Conversation DB (lightweight, for /status)
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
            chat_id   INTEGER PRIMARY KEY,
            claude_session_id TEXT,
            goal      TEXT,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0
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


def get_session_row(chat_id: int):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT chat_id, claude_session_id, goal, last_active, message_count FROM sessions WHERE chat_id = ?",
        (chat_id,),
    ).fetchone()
    conn.close()
    return row


def upsert_session(chat_id: int, claude_session_id: str | None, goal: str | None = None):
    conn = sqlite3.connect(DB_PATH)
    if goal is None:
        conn.execute(
            "INSERT INTO sessions (chat_id, claude_session_id) VALUES (?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET claude_session_id = excluded.claude_session_id, last_active = CURRENT_TIMESTAMP",
            (chat_id, claude_session_id),
        )
    else:
        conn.execute(
            "INSERT INTO sessions (chat_id, claude_session_id, goal) VALUES (?, ?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET claude_session_id = excluded.claude_session_id, goal = COALESCE(sessions.goal, excluded.goal), last_active = CURRENT_TIMESTAMP",
            (chat_id, claude_session_id, goal),
        )
    conn.commit()
    conn.close()


def bump_message_count(chat_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE sessions SET message_count = message_count + 1, last_active = CURRENT_TIMESTAMP WHERE chat_id = ?",
        (chat_id,),
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Training log (JSONL, one event per line)
# ─────────────────────────────────────────────────────────────

def log_event(event_type: str, chat_id: int, data: dict):
    """Append a structured event to the daily JSONL log for later training."""
    log_file = LOG_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "chat_id": chat_id,
        **data,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
# ClaudeRunner — persistent subprocess per chat, stream-json I/O
# ─────────────────────────────────────────────────────────────

class ClaudeRunner:
    """
    Manages one long-running `claude` subprocess per chat.
    - Prompts are sent as JSON to stdin.
    - Responses are parsed from stdout stream-json events.
    - SteeringManager is notified on each response.
    - User can inject mid-run prompts via inject_prompt().
    """

    def __init__(self, chat_id: int, steer_manager: "SteeringManager"):
        self.chat_id = chat_id
        self.steer_manager = steer_manager
        self.session_id = str(uuid.uuid4())  # our session id (separate from claude's)
        self.claude_session_id: str | None = None  # claude's internal id (for --resume)
        self.process: asyncio.subprocess.Process | None = None
        self.goal: str = ""
        self.response_count = 0
        self.last_assistant_text: str = ""
        self.last_event_ts: float = 0.0
        self.steer_asked_at: float = 0.0
        self._stdout_task: asyncio.Task | None = None
        self._pending_inject: list[str] = []
        self._inject_event = asyncio.Event()
        self._send_to_user_fn: Optional[Callable] = None
        self._user_chat: Any | None = None  # telegram Chat object for steering question
        self._running = False

    async def start(self, goal: str, send_fn, user_chat):
        """Spawn the claude subprocess, send initial goal, start streaming output."""
        self.goal = goal
        self._send_to_user_fn = send_fn
        self._user_chat = user_chat
        self._running = True

        # Build claude command
        cmd = [
            CLAUDE_COMMAND, "--print",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--session-id", self.session_id,
            "--model", CLAUDE_MODEL,
        ] + CLAUDE_EXTRA_ARGS.split()

        log_event("runner_start", self.chat_id, {
            "session_id": self.session_id,
            "goal": goal,
            "cmd": " ".join(cmd),
        })

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "NO_COLOR": "1", "CLAUDE_CODE_SIMPLE": "1"},
        )

        upsert_session(self.chat_id, self.session_id, goal)
        log_msg(self.chat_id, "user", f"[goal] {goal}", "goal")

        # Send initial prompt
        await self._send_prompt(goal)

        # Start streaming stdout
        self._stdout_task = asyncio.create_task(self._read_stdout())

    async def inject_prompt(self, prompt: str):
        """Queue a prompt to be sent to claude on next opportunity.
        Use this for steering interventions from the user."""
        log_event("user_inject", self.chat_id, {"prompt": prompt, "goal": self.goal})
        log_msg(self.chat_id, "user", f"[steer] {prompt}", "steer")
        self._pending_inject.append(prompt)
        self._inject_event.set()

    async def _send_prompt(self, content: str):
        """Write a user-message JSON to claude's stdin."""
        if not self.process or self.process.stdin.is_closing():
            return
        msg = {"type": "user", "message": {"role": "user", "content": content}}
        line = json.dumps(msg) + "\n"
        try:
            self.process.stdin.write(line.encode("utf-8"))
            await self.process.stdin.drain()
        except Exception as e:
            log_event("inject_error", self.chat_id, {"error": str(e)})

    async def _flush_pending_injects(self):
        """If user sent prompts while claude was running, send them now."""
        while self._pending_inject:
            prompt = self._pending_inject.pop(0)
            await self._send_prompt(prompt)
        self._inject_event.clear()

    async def _read_stdout(self):
        """Read stream-json events from claude's stdout, forward to user + steer manager."""
        send_fn = self._send_to_user_fn
        buffer = ""
        try:
            while self._running and self.process and self.process.stdout:
                chunk = await self.process.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                # Parse newline-delimited JSON
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        log_event("parse_error", self.chat_id, {"line": line[:300]})
                        continue
                    await self._handle_event(ev, send_fn)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_event("stdout_error", self.chat_id, {"error": str(e), "trace": str(e)})
        finally:
            self._running = False

    async def _handle_event(self, ev: dict, send_fn):
        et = ev.get("type")
        log_event("claude_event", self.chat_id, {"event_type": et, "event": ev})

        if et == "system" and ev.get("subtype") == "init":
            self.claude_session_id = ev.get("session_id") or self.session_id
            self.last_event_ts = time.monotonic()
            return

        if et == "assistant":
            msg = ev.get("message", {})
            content = msg.get("content", [])
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if text and text != self.last_assistant_text:
                        self.last_assistant_text = text
                        self.last_event_ts = time.monotonic()
                        # Stream to user (truncate huge outputs)
                        preview = text[:3500] + ("\n\n[…truncated]" if len(text) > 3500 else "")
                        await send_fn(preview)
                        log_msg(self.chat_id, "claude", text, "text")
                        self.response_count += 1
                        bump_message_count(self.chat_id)
                        # Notify steering manager
                        await self.steer_manager.on_response(self, text)
                elif btype == "tool_use":
                    name = block.get("name", "?")
                    inp = block.get("input", {})
                    summary = inp.get("file_path", inp.get("command", inp.get("description", str(inp)[:200])))
                    await send_fn(f"🔧 **{name}**\n`{summary}`")
                    log_event("tool_use", self.chat_id, {"tool": name, "summary": summary})
                elif btype == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        # Log thinking but don't spam user
                        log_event("claude_thinking", self.chat_id, {"thinking": thinking[:500]})
            # After assistant message, flush any pending user injects (user wants to steer mid-stream)
            await self._flush_pending_injects()
            return

        if et == "user":
            # tool_result coming back
            tr = ev.get("message", {}).get("content", [])
            for block in tr:
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
                    preview = (content or "")[:1500]
                    if preview.strip():
                        await send_fn(f"📥 tool result:\n```\n{preview}\n```")
                        log_event("tool_result", self.chat_id, {"content": preview[:500]})
            return

        if et == "result":
            result_text = ev.get("result", "")
            duration = ev.get("duration_ms", 0)
            cost = ev.get("total_cost_usd", 0)
            await send_fn(f"✅ done ({duration/1000:.1f}s, ${cost:.4f})")
            log_event("claude_result", self.chat_id, {"result": result_text, "duration_ms": duration, "cost_usd": cost})
            # Flush any pending injects (this might trigger next turn)
            await self._flush_pending_injects()
            return

    async def stop(self):
        self._running = False
        if self._stdout_task and not self._stdout_task.done():
            self._stdout_task.cancel()
        if self.process and self.process.returncode is None:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        log_event("runner_stop", self.chat_id, {"session_id": self.session_id})


# ─────────────────────────────────────────────────────────────
# SteeringManager — M3-driven, persistent state, asks user when needed
# ─────────────────────────────────────────────────────────────

class SteeringManager:
    """
    Tracks session state across all chats. Every CHECKPOINT_EVERY responses,
    calls M3 to summarize recent activity and decide if steering is needed.
    If yes, sends a Telegram question to the user; user reply is routed to
    ClaudeRunner.inject_prompt() on next opportunity.
    """

    def __init__(self):
        self.lock = asyncio.Lock()
        self.state: dict[int, dict] = {}  # chat_id -> state dict

    def _state_file(self, chat_id: int) -> Path:
        return SESSION_DIR / f"{chat_id}.json"

    def _load_state(self, chat_id: int) -> dict:
        if chat_id in self.state:
            return self.state[chat_id]
        path = self._state_file(chat_id)
        if path.exists():
            try:
                self.state[chat_id] = json.loads(path.read_text())
                return self.state[chat_id]
            except Exception:
                pass
        self.state[chat_id] = {
            "goal": "",
            "user_intent": "",
            "decisions_made": [],
            "current_understanding": "",
            "last_responses": [],
            "checkpoint_count": 0,
            "steering_history": [],
        }
        return self.state[chat_id]

    def _save_state(self, chat_id: int):
        path = self._state_file(chat_id)
        path.write_text(json.dumps(self.state[chat_id], indent=2, ensure_ascii=False))

    async def on_response(self, runner: ClaudeRunner, text: str):
        """Called by ClaudeRunner after each assistant text response."""
        async with self.lock:
            st = self._load_state(runner.chat_id)
            st["goal"] = st.get("goal") or runner.goal
            # Keep last 20 responses
            st["last_responses"] = (st.get("last_responses", []) + [{"ts": time.time(), "text": text[:1000]}])[-20:]
            self._save_state(runner.chat_id)

        if runner.response_count % CHECKPOINT_EVERY == 0:
            await self.checkpoint(runner)

    async def checkpoint(self, runner: ClaudeRunner):
        """Call M3 to summarize state and decide if steering is needed."""
        async with self.lock:
            st = self._load_state(runner.chat_id)
            last = st.get("last_responses", [])[-CHECKPOINT_EVERY:]
            recent_text = "\n\n---\n\n".join(r["text"][:500] for r in last)
            goal = st.get("goal", "")

        system_prompt = """You are a steering manager for an AI coding agent.
You monitor its work and decide when the human user needs to intervene.
Reply in EXACTLY this format (no extra text):

<user_intent>
What the user is trying to accomplish, inferred from their goal + the agent's recent work.
</user_intent>

<decisions>
- bullet 1
- bullet 2
</decisions>

<understanding>
What the agent is currently doing / just finished.
</understanding>

<needs_steering>
yes or no
</needs_steering>

<steer_reason>
If needs_steering=yes, write ONE specific question to ask the user (max 200 chars).
If no, write "none".
</steer_reason>"""

        user_prompt = f"""## User's goal
{goal}

## Recent agent responses (last {len(last)})
{recent_text}

Now produce your structured assessment."""

        try:
            raw = mx_chat(
                [{"role": "user", "content": user_prompt}],
                system=system_prompt,
                max_tokens=1500,
            )
        except Exception as e:
            log_event("steer_error", runner.chat_id, {"error": str(e)})
            return

        parsed = _parse_steer_output(raw)
        log_event("checkpoint", runner.chat_id, {"parsed": parsed, "raw": raw[:500]})

        async with self.lock:
            st = self._load_state(runner.chat_id)
            st["user_intent"] = parsed.get("user_intent", "")
            st["current_understanding"] = parsed.get("understanding", "")
            st["decisions_made"] = parsed.get("decisions", [])
            st["checkpoint_count"] = st.get("checkpoint_count", 0) + 1
            if parsed.get("needs_steering"):
                st["steering_history"].append({
                    "ts": time.time(),
                    "reason": parsed.get("steer_reason", ""),
                    "checkpoint": st["checkpoint_count"],
                })
            self._save_state(runner.chat_id)

        # Send the checkpoint summary to the user
        summary_msg = _format_checkpoint_summary(parsed, st.get("checkpoint_count", 0))
        if runner._send_to_user_fn:
            await runner._send_to_user_fn(summary_msg)

        # If steering needed, ask the user (debounced — only once per 5 min)
        if parsed.get("needs_steering") and runner._user_chat:
            now = time.time()
            if now - runner.steer_asked_at > 300:
                runner.steer_asked_at = now
                question = parsed.get("steer_reason", "Steering needed, please advise.")
                await runner._user_chat.send_message(
                    f"❓ *Steering needed*\n\n{question}\n\n_(your next message will be forwarded to the agent)_",
                    parse_mode=ParseMode.MARKDOWN,
                )
                log_event("steer_asked", runner.chat_id, {"question": question})


def _parse_steer_output(raw: str) -> dict:
    """Parse the structured <tag>value</tag> output from the steering manager."""
    def extract(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL)
        return m.group(1).strip() if m else ""

    decisions_raw = extract("decisions")
    decisions = [d.strip("- \n") for d in decisions_raw.split("\n") if d.strip().startswith("-")]

    return {
        "user_intent": extract("user_intent"),
        "decisions": decisions,
        "understanding": extract("understanding"),
        "needs_steering": extract("needs_steering").lower().startswith("y"),
        "steer_reason": extract("steer_reason"),
    }


def _format_checkpoint_summary(parsed: dict, n: int) -> str:
    lines = [f"📊 *Checkpoint #{n}*", ""]
    if parsed.get("user_intent"):
        lines.append(f"🎯 *Goal*: {parsed['user_intent'][:300]}")
    if parsed.get("understanding"):
        lines.append(f"📍 *Status*: {parsed['understanding'][:300]}")
    if parsed.get("decisions"):
        lines.append("🧠 *Decisions*:")
        for d in parsed["decisions"][:5]:
            lines.append(f"  • {d[:200]}")
    if parsed.get("needs_steering"):
        lines.append("")
        lines.append(f"⚠️ *Needs steering*: {parsed.get('steer_reason', '')[:200]}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Per-chat active run tracking
# ─────────────────────────────────────────────────────────────
_active_runs: dict[int, ClaudeRunner] = {}
_runs_lock = asyncio.Lock()
_steer_manager = SteeringManager()


# ─────────────────────────────────────────────────────────────
# Telegram message helpers
# ─────────────────────────────────────────────────────────────

async def send_long(update: Update, text: str, parse_mode=ParseMode.MARKDOWN):
    for i in range(0, len(text), MAX_MSG_LEN):
        chunk = text[i : i + MAX_MSG_LEN]
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode)
        except Exception:
            await update.message.reply_text(chunk, parse_mode=None)


# ─────────────────────────────────────────────────────────────
# Telegram handlers
# ─────────────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text.strip()
    if not prompt or prompt.startswith("/"):
        return
    chat_id = update.message.chat_id
    user = update.message.from_user
    log_msg(chat_id, "user", prompt, "text")
    log_event("user_message", chat_id, {"text": prompt, "user_id": user.id if user else None})

    async with _runs_lock:
        runner = _active_runs.get(chat_id)

    if runner is not None and runner._running:
        # Mid-run injection — user is steering
        await runner.inject_prompt(prompt)
        await update.message.reply_text(
            f"↪️ injected into running session (response #{runner.response_count + 1} pending)",
            do_quote=True,
        )
        return

    # No active run — start a new persistent session
    await update.message.reply_text("🤖 starting claude session…", do_quote=True)

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    async with _runs_lock:
        # If a runner exists but is dead, clean it up first
        old = _active_runs.get(chat_id)
        if old is not None:
            await old.stop()
            _active_runs.pop(chat_id, None)
        runner = ClaudeRunner(chat_id, _steer_manager)
        _active_runs[chat_id] = runner
        await runner.start(prompt, send_fn, update.message.chat)


async def handle_image_gen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text.replace("/imagine", "").replace("/image", "").strip()
    if not prompt:
        await update.message.reply_text("Usage: `/imagine <prompt>`", do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return
    if not MINIMAX_API_KEY:
        await update.message.reply_text("MINIMAX_API_KEY not set", do_quote=True)
        return
    await update.message.reply_text("🎨 generating…", do_quote=True)
    try:
        img_bytes = mx_image(prompt)
        bio = io.BytesIO(img_bytes)
        bio.name = "generated.png"
        await update.message.reply_photo(photo=bio, do_quote=True)
        log_event("image_gen", update.message.chat_id, {"prompt": prompt})
    except Exception as e:
        await update.message.reply_text(f"image gen error: {e}", do_quote=True)


async def handle_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.replace("/speak", "").replace("/tts", "").strip()
    if not text:
        await update.message.reply_text("Usage: `/speak <text>`", do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return
    if not MINIMAX_API_KEY:
        await update.message.reply_text("MINIMAX_API_KEY not set", do_quote=True)
        return
    await update.message.reply_text("🔊 generating audio…", do_quote=True)
    try:
        mp3_bytes = mx_tts(text)
        bio = io.BytesIO(mp3_bytes)
        bio.name = "speech.mp3"
        await update.message.reply_audio(audio=bio, do_quote=True)
        log_event("tts", update.message.chat_id, {"text": text[:200]})
    except Exception as e:
        await update.message.reply_text(f"tts error: {e}", do_quote=True)


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    row = get_session_row(chat_id)
    if not row:
        await update.message.reply_text("no session yet", do_quote=True)
        return
    state_path = SESSION_DIR / f"{chat_id}.json"
    state_summary = ""
    if state_path.exists():
        try:
            st = json.loads(state_path.read_text())
            state_summary = (
                f"\n🎯 *Goal*: {st.get('goal','')[:200]}"
                f"\n📍 *Status*: {st.get('current_understanding','')[:200]}"
                f"\n📊 Checkpoints: {st.get('checkpoint_count', 0)}"
            )
        except Exception:
            pass
    await update.message.reply_text(
        f"🆔 Session: `{row[1]}`\n"
        f"💬 Messages: {row[4]}\n"
        f"🕐 Last active: {row[3]}"
        f"{state_summary}",
        do_quote=True,
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if runner:
        await runner.stop()
        async with _runs_lock:
            _active_runs.pop(chat_id, None)
        await update.message.reply_text("🛑 session stopped", do_quote=True)
    else:
        await update.message.reply_text("no active session", do_quote=True)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Claude-bot v2 — persistent sessions*\n\n"
        "• Send any message → starts a *persistent* Claude session for this chat\n"
        "• Send another message while it's running → *injected as steering input*\n"
        "• Every 10 responses, a checkpoint summary is posted + steering check\n"
        "• If steering needed, you'll be asked a question — just reply\n\n"
        "*Commands:*\n"
        "`/status` — show session state + checkpoint info\n"
        "`/cancel` — stop the running session\n"
        "`/imagine <prompt>` — generate image (MiniMax)\n"
        "`/speak <text>` — text-to-speech (MiniMax)\n",
        do_quote=True,
        parse_mode=ParseMode.MARKDOWN,
    )


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

def main():
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set", flush=True)
        return
    if not MINIMAX_API_KEY:
        print("WARN: MINIMAX_API_KEY not set — steering manager + image + TTS disabled", flush=True)

    init_db()

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler(["help", "start"], handle_help))
    app.add_handler(CommandHandler(["status"], handle_status))
    app.add_handler(CommandHandler(["cancel"], handle_cancel))
    app.add_handler(CommandHandler(["imagine", "image"], handle_image_gen))
    app.add_handler(CommandHandler(["speak", "tts"], handle_tts))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print(f"Bot v2 starting…", flush=True)
    print(f"  CLAUDE_MODEL      = {CLAUDE_MODEL}", flush=True)
    print(f"  CHECKPOINT_EVERY  = {CHECKPOINT_EVERY}", flush=True)
    print(f"  Training log dir  = {LOG_DIR}", flush=True)
    print(f"  Session state dir = {SESSION_DIR}", flush=True)
    print(f"  DB                = {DB_PATH}", flush=True)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
