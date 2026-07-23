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
import sys
import time
import uuid
import asyncio
import logging
import shlex
import sqlite3
import subprocess
import base64
import tempfile
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
# Persisted session dir (used by /resume, /sessions, /fork)
CLAUDE_PROJECTS_DIR = Path(os.path.expanduser("~/.claude/projects"))
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# Local Whisper: model name (tiny/base/small/medium/large) and venv path to
# import openai-whisper from. Tiny is fast on CPU (~2.5s per 1s of audio) and
# fine for Telegram voice notes. Set WHISPER_LOCAL=0 to force API-only.
WHISPER_LOCAL     = os.getenv("WHISPER_LOCAL", "1") not in ("0", "false", "no", "")
WHISPER_LOCAL_MODEL = os.getenv("WHISPER_LOCAL_MODEL", "tiny")
# Herms-agent venv has openai-whisper + torch installed (used by the
# practicepteonline pipeline). Reuse it instead of installing into the bot venv.
HERMES_VENV_SITE = "/home/Hermes/.hermes/hermes-agent/venv/lib/python3.11/site-packages"
_whisper_module = None  # lazy-loaded whisper module
_whisper_model = None    # lazy-loaded whisper model object

CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "10"))   # checkpoint after N responses
SILENT_FOR_STEER = int(os.getenv("SILENT_FOR_STEER", "60"))   # sec of no output before asking user
# Stuck-session watchdog: if claude produces no events for this long AND the
# user has pending input, kill the session so the next message starts fresh.
# Set to ~2-3x a normal long response (which can take 30-60s on heavy queries).
STUCK_SESSION_TIMEOUT_S = int(os.getenv("STUCK_SESSION_TIMEOUT_S", "90"))
MAX_MSG_LEN      = 4096

# Telegram MarkdownV2 reserved chars that need escaping when user content is
# embedded in a parse_mode=MARKDOWN message. Without escaping, a tool_result
# containing "_" or "*" or "[" makes the entire message fail to parse and the
# user sees nothing.
_MDV2_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")

def md_escape(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters in untrusted text."""
    if not text:
        return ""
    return _MDV2_SPECIAL.sub(r"\\\1", text)

def md_code_escape(text: str) -> str:
    """Escape only backticks and backslashes for content inside a ``` block."""
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace("```", "ʼʼʼ")


def suggest_directory_match(missing: str, max_suggestions: int = 3) -> list[str]:
    """Find close-match directories when a path doesn't exist.

    Searches immediate parent for siblings whose name is a case-insensitive
    near-match to the missing basename. Used by /cd, /adddir, /ls, /tree
    to give the user a hint instead of a dead-end error.
    """
    missing_path = Path(missing)
    parent = missing_path.parent
    target = missing_path.name.lower()
    if not parent.exists() or not parent.is_dir():
        return []
    try:
        candidates = [p.name for p in parent.iterdir() if p.is_dir()]
    except (PermissionError, OSError):
        return []
    # Score: case-insensitive exact match > prefix match > contains > Levenshtein
    from difflib import get_close_matches
    exact_ci = [c for c in candidates if c.lower() == target]
    if exact_ci:
        return [str(parent / exact_ci[0])]
    prefix = [c for c in candidates if c.lower().startswith(target[:3]) and c.lower() != target]
    if prefix:
        return [str(parent / p) for p in prefix[:max_suggestions]]
    fuzzy = get_close_matches(target, [c.lower() for c in candidates], n=max_suggestions, cutoff=0.6)
    if fuzzy:
        # Map back to original-case names
        lc_to_orig = {c.lower(): c for c in candidates}
        return [str(parent / lc_to_orig[f]) for f in fuzzy]
    return []


def path_not_found_message(missing: str) -> str:
    """Build a 'not a directory' error with close-match suggestions."""
    msg = f"❌ not a directory: `{missing}`"
    suggestions = suggest_directory_match(missing)
    if suggestions:
        msg += "\n\nDid you mean:\n" + "\n".join(f"  • `{s}`" for s in suggestions)
    return msg


# Map of leading bytes → MIME type for image formats Telegram can send.
# (We don't need to support every format — just enough that an actual
# PNG/GIF/WebP photo doesn't get rejected by claude CLI for a wrong MIME.)
_IMAGE_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP — check more carefully below
    (b"BM", "image/bmp"),
)


def sniff_image_mime(img_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes. Falls back to image/jpeg."""
    for magic, mime in _IMAGE_MAGIC:
        if img_bytes.startswith(magic):
            if mime == "image/webp" and not img_bytes[8:12] == b"WEBP":
                return "image/jpeg"  # RIFF but not WEBP — fall through
            return mime
    return "image/jpeg"


async def safe_send(update, text: str, parse_mode=ParseMode.MARKDOWN, **kwargs):
    """Send a Telegram message, falling back through MarkdownV2 -> plain text on parse failure.

    Use this for any message that includes user-generated or tool-result content
    that may contain MarkdownV2 special chars (_, *, [, ], etc.).
    """
    bot = update.effective_message.get_bot() if update.effective_message else update.bot
    chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id
    do_quote = kwargs.pop("do_quote", True)
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode, do_quote=do_quote, **kwargs)
    except Exception as e:
        # Markdown parse failed — try with everything escaped
        try:
            await bot.send_message(chat_id=chat_id, text=md_escape(text), parse_mode=parse_mode, do_quote=do_quote, **kwargs)
        except Exception:
            # Still failed — send as plain text
            await bot.send_message(chat_id=chat_id, text=text, do_quote=do_quote, **kwargs)
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
    """Initialize DB schema. Drops + recreates sessions table if schema is stale.
    (The messages table is preserved — it has real history.)"""
    conn = sqlite3.connect(DB_PATH)
    # Check if sessions table has the new schema
    cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    needs_recreate = cols and (
        "claude_session_id" not in cols or "last_active" not in cols or "message_count" not in cols
    )
    if needs_recreate:
        # Backup old data, recreate
        conn.execute("ALTER TABLE sessions RENAME TO sessions_old")
        conn.execute("""
            CREATE TABLE sessions (
                chat_id   INTEGER PRIMARY KEY,
                claude_session_id TEXT,
                goal      TEXT,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0
            )
        """)
        # Migrate any old rows (best-effort)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO sessions (chat_id, goal, last_active, message_count)
                SELECT chat_id, goal, started, 0 FROM sessions_old
            """)
        except Exception:
            pass
        conn.execute("DROP TABLE sessions_old")
        conn.commit()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            chat_id   INTEGER PRIMARY KEY,
            claude_session_id TEXT,
            goal      TEXT,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0
        )
    """)
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
            "ON CONFLICT(chat_id) DO UPDATE SET claude_session_id = excluded.claude_session_id, goal = excluded.goal, last_active = CURRENT_TIMESTAMP",
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
        self._session_started_at: float = 0.0  # monotonic clock when start() began
        self.steer_asked_at: float = 0.0
        self._stdout_task: asyncio.Task | None = None
        self._pending_inject: list = []  # each item is str OR list[dict] (Anthropic content blocks)
        self._inject_event = asyncio.Event()
        self._send_to_user_fn: Optional[Callable] = None
        self._user_chat: Any | None = None  # telegram Chat object for steering question
        self._running = False
        # Serialize all writes to claude's stdin. Without this, concurrent
        # inject_prompt() calls interleave at the byte level and corrupt the
        # JSON stream — claude CLI then hangs or crashes mid-session.
        self._stdin_lock = asyncio.Lock()
        # Configurable flags (set via /agent, /add-dir, /model, /effort, /cd)
        self.config: dict = {
            "model": CLAUDE_MODEL,
            "cwd": os.getcwd(),    # claude's working directory (set by /cd)
            "add_dirs": [],        # list of paths → --add-dir
            "agent": None,         # → --agent
            "agents": None,        # → --agents (json string)
            "effort": None,        # → --effort (low/medium/high/xhigh/max)
            "permission_mode": None,  # → --permission-mode
            "append_system_prompt": None,
            "resume": None,        # → --resume <id> (start by resuming)
            "fork_session": False, # → --fork-session
        }

    async def start(self, goal: str, send_fn, user_chat):
        """Spawn a new claude subprocess with the given goal, stream results to user."""
        self.goal = goal
        self._send_to_user_fn = send_fn
        self._user_chat = user_chat
        self._session_started_at = time.monotonic()
        # Reset event timestamp so the watchdog doesn't trigger immediately
        # for sessions that take >90s to produce their first event (e.g. cold
        # model load or first API call after a long pause).
        if self.last_event_ts == 0.0:
            self.last_event_ts = self._session_started_at
        self._running = True

        # Reset steering state for the new session so the new goal wins over any old one
        if self.steer_manager is not None:
            await self.steer_manager.reset_for_new_session(self.chat_id, goal)

        # Build claude command from current config
        cmd = self._build_cmd()

        log_event("runner_start", self.chat_id, {
            "session_id": self.session_id,
            "goal": goal,
            "cmd": " ".join(cmd),
            "config": self.config,
        })

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "NO_COLOR": "1", "CLAUDE_CODE_SIMPLE": "1"},
                cwd=self.config.get("cwd") or os.getcwd(),
            )
        except FileNotFoundError as e:
            log_event("runner_start_failed", self.chat_id, {"error": str(e)})
            self._running = False
            if self._send_to_user_fn:
                await self._send_to_user_fn(f"❌ failed to start claude: {e}")
            return

        upsert_session(self.chat_id, self.session_id, goal)
        log_msg(self.chat_id, "user", f"[goal] {goal}", "goal")

        # Send initial prompt
        await self._send_prompt(goal)

        # Start streaming stdout
        self._stdout_task = asyncio.create_task(self._read_stdout())

        # Surface early failures: if claude CLI exits within 2s (bad model name,
        # missing --bare flag, auth error, etc.), tell the user instead of
        # silently leaving them with a session that never produces output.
        try:
            await asyncio.wait_for(self.process.wait(), timeout=2.0)
            # Process exited — read stderr and report
            stderr = b""
            try:
                stderr = await self.process.stderr.read() if self.process.stderr else b""
            except Exception:
                pass
            err_text = stderr.decode("utf-8", errors="replace").strip()[-500:]
            log_event("runner_early_exit", self.chat_id, {
                "returncode": self.process.returncode,
                "stderr": err_text,
            })
            if self._send_to_user_fn:
                await self._send_to_user_fn(
                    f"❌ claude exited immediately (code {self.process.returncode})\n{err_text}"
                )
        except asyncio.TimeoutError:
            # Normal: process is still running after 2s
            pass

        # Start the stuck-session watchdog. claude can hang silently when the
        # LLM API is throttled or wedged — without this, the user sees
        # "injected into running session" and then nothing forever.
        # We track last_event_ts; if it goes >STUCK_SESSION_TIMEOUT_S with
        # pending user input, kill the runner so the next user message
        # starts a fresh one.
        self._watchdog_task = asyncio.create_task(self._stuck_session_watchdog())

    async def _stuck_session_watchdog(self):
        """Kill the session if it's silent too long with pending user input.
        Runs forever; called as a task at session start.
        """
        try:
            while self._running:
                await asyncio.sleep(15)
                if not self._running or not self.process:
                    return
                # If claude is alive but has produced no event AND the user
                # has something queued to send, it's stuck.
                age = time.monotonic() - self.last_event_ts if self.last_event_ts else 0
                has_pending = bool(self._pending_inject)
                if has_pending and age > STUCK_SESSION_TIMEOUT_S:
                    log_event("stuck_session_kill", self.chat_id, {
                        "silent_seconds": int(age),
                        "pending_injects": len(self._pending_inject),
                    })
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                    if self._send_to_user_fn:
                        await self._send_to_user_fn(
                            f"⚠️ claude hung for {int(age)}s — killed session. "
                            f"Send any message to start a fresh one."
                        )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_event("watchdog_error", self.chat_id, {"error": str(e)})

    def _build_cmd(self) -> list[str]:
        """Build the claude CLI command from current config."""
        cmd = [
            CLAUDE_COMMAND, "--print",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--session-id", self.session_id,
            "--model", self.config["model"],
        ]
        if self.config.get("resume"):
            cmd += ["--resume", self.config["resume"]]
        if self.config.get("fork_session"):
            cmd += ["--fork-session"]
        for d in self.config.get("add_dirs", []):
            cmd += ["--add-dir", d]
        if self.config.get("agent"):
            cmd += ["--agent", self.config["agent"]]
        if self.config.get("agents"):
            cmd += ["--agents", self.config["agents"]]
        if self.config.get("effort"):
            cmd += ["--effort", self.config["effort"]]
        if self.config.get("permission_mode"):
            cmd += ["--permission-mode", self.config["permission_mode"]]
        if self.config.get("append_system_prompt"):
            cmd += ["--append-system-prompt", self.config["append_system_prompt"]]
        # shlex.split respects quoted values ("--system-prompt 'my prompt.md'")
        # while .split() would mangle the quoted path into 3 tokens.
        cmd += shlex.split(CLAUDE_EXTRA_ARGS)
        return cmd

    async def restart_with_config(self, new_config: dict, send_fn):
        """Stop the current subprocess, apply new config, restart with the same goal.
        Used when user changes /model, /agent, /add-dir etc mid-session."""
        # Drain any steering messages queued during the restart window — they
        # were addressed at the OLD session and would be sent as the first
        # user message of the NEW session, polluting its history.
        dropped = self._pending_inject
        self._pending_inject = []
        if self._running:
            await send_fn("🔄 restarting session with new config…")
            log_event("runner_restart", self.chat_id, {
                "new_config": new_config,
                "dropped_pending_injects": dropped,
            })
            await self.stop()
        # Update config
        for k, v in new_config.items():
            if k in self.config:
                self.config[k] = v
        # New session id (because --resume points to old session)
        if not self.config.get("resume"):
            self.session_id = str(uuid.uuid4())
        # Reset response counter so the new session doesn't fire a premature
        # checkpoint on its first response (counter was carried over from old).
        self.response_count = 0
        # Restart
        await self.start(self.goal, send_fn, self._user_chat)

    async def inject_prompt(self, prompt: str):
        """Queue a prompt to be sent to claude on next opportunity.
        Use this for steering interventions from the user."""
        log_event("user_inject", self.chat_id, {"prompt": prompt, "goal": self.goal})
        log_msg(self.chat_id, "user", f"[steer] {prompt}", "steer")
        self._pending_inject.append(prompt)
        self._inject_event.set()

    async def _send_prompt(self, content):
        """Write a user-message JSON to claude's stdin.
        content can be a string OR a list of content blocks (Anthropic format)."""
        if not self.process or self.process.stdin.is_closing():
            return
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        msg = {"type": "user", "message": {"role": "user", "content": content}}
        line = json.dumps(msg) + "\n"
        async with self._stdin_lock:
            try:
                self.process.stdin.write(line.encode("utf-8"))
                await self.process.stdin.drain()
            except Exception as e:
                log_event("inject_error", self.chat_id, {"error": str(e)})

    async def start_multimodal(self, content: list, send_fn, user_chat):
        """Start a session with multi-part content (text + images).
        Like start() but with Anthropic-format content list as initial prompt."""
        self.goal = "(multimodal prompt)"
        self._send_to_user_fn = send_fn
        self._user_chat = user_chat
        self._running = True

        cmd = self._build_cmd()
        log_event("runner_start_multimodal", self.chat_id, {
            "session_id": self.session_id,
            "cmd": " ".join(cmd),
            "config": self.config,
        })

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "NO_COLOR": "1", "CLAUDE_CODE_SIMPLE": "1"},
                cwd=self.config.get("cwd") or os.getcwd(),
            )
        except FileNotFoundError as e:
            log_event("runner_start_failed", self.chat_id, {"error": str(e)})
            self._running = False
            if self._send_to_user_fn:
                await self._send_to_user_fn(f"❌ failed to start claude: {e}")
            return

        upsert_session(self.chat_id, self.session_id, "(multimodal)")
        log_msg(self.chat_id, "user", "[multimodal prompt]", "image")

        # Send the multi-part content directly
        await self._send_prompt(content)

        # Start streaming stdout
        self._stdout_task = asyncio.create_task(self._read_stdout())

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
                        await send_fn(f"📥 tool result:\n```\n{md_code_escape(preview)}\n```")
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

    async def reset_for_new_session(self, chat_id: int, new_goal: str) -> None:
        """Wipe the per-session steering state when a fresh /new is launched.

        Without this, the old session's goal, responses, and checkpoint survive
        a /new — the steering manager would keep reasoning against stale context.
        """
        async with self.lock:
            st = self._load_state(chat_id)
            st["goal"] = new_goal
            st["last_responses"] = []
            st["checkpoint"] = None
            st["steer_history"] = []
            self._save_state(chat_id)

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
# chat_id -> True when a bare /new is awaiting the next text message as the goal
_pending_new_goal: dict[int, bool] = {}
_pending_lock = asyncio.Lock()


async def _set_pending_goal(chat_id: int) -> None:
    async with _pending_lock:
        _pending_new_goal[chat_id] = True


async def _consume_pending_goal(chat_id: int) -> bool:
    """Return True (and clear) if this chat has a pending /new goal awaiting text."""
    async with _pending_lock:
        return _pending_new_goal.pop(chat_id, False)


async def _clear_pending_goal(chat_id: int) -> None:
    """Clear the pending-goal flag without consuming it (e.g. on /cancel)."""
    async with _pending_lock:
        _pending_new_goal.pop(chat_id, None)
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

    # If user just ran `/new` with no goal, this next text is the goal
    if await _consume_pending_goal(chat_id):
        # Don't double-log the goal — runner.start() will log it as a [goal] message
        await update.message.reply_text(
            f"🆕 starting new session with goal: {prompt}",
            do_quote=True,
        )
        log_event("new_session", chat_id, {"goal": prompt, "via": "interactive_prompt"})

        async def send_fn(text):
            try:
                await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                await update.message.reply_text(text, do_quote=False)

        async with _runs_lock:
            old = _active_runs.get(chat_id)
            if old is not None:
                await old.stop()
                _active_runs.pop(chat_id, None)
            runner = ClaudeRunner(chat_id, _steer_manager)
            _active_runs[chat_id] = runner
            await runner.start(prompt, send_fn, update.message.chat)
        return

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
    async with _runs_lock:
        runner = _active_runs.get(chat_id)

    state_summary = ""
    state_path = SESSION_DIR / f"{chat_id}.json"
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

    config_summary = ""
    if runner:
        c = runner.config
        dirs = ", ".join(c.get("add_dirs", [])) or "(none)"
        config_summary = (
            f"\n\n⚙️ *Config*:"
            f"\n  cwd: `{c.get('cwd') or os.getcwd()}`"
            f"\n  model: `{c.get('model', CLAUDE_MODEL)}`"
            f"\n  agent: `{c.get('agent') or '(default)'}`"
            f"\n  add-dirs: {dirs}"
            f"\n  effort: `{c.get('effort') or '(default)'}`"
            f"\n  resume: `{c.get('resume') or '(new session)'}`"
            f"\n  responses so far: {runner.response_count}"
            f"\n  running: {'✅' if runner._running else '❌'}"
        )
    elif row:
        config_summary = f"\n🆔 Last session: `{row[1]}` (use `/resume {row[1][:8]}…` to continue)"

    if not row and not runner:
        await update.message.reply_text("no session yet", do_quote=True)
        return

    await update.message.reply_text(
        f"💬 Messages: {row[4] if row else 0}\n"
        f"🕐 Last active: {row[3] if row else 'n/a'}"
        f"{state_summary}"
        f"{config_summary}",
        do_quote=True,
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    await _clear_pending_goal(chat_id)
    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if runner:
        await runner.stop()
        async with _runs_lock:
            _active_runs.pop(chat_id, None)
        await update.message.reply_text("🛑 session stopped", do_quote=True)
    else:
        await update.message.reply_text("no active session", do_quote=True)


# ─────────────────────────────────────────────────────────────
# Slash commands — full Claude Code parity
# ─────────────────────────────────────────────────────────────

async def _get_or_create_runner(update, chat_id, send_fn):
    """Get the active runner, or return None if no session exists."""
    async with _runs_lock:
        return _active_runs.get(chat_id)


async def _stop_runner(chat_id):
    async with _runs_lock:
        runner = _active_runs.get(chat_id)
        if runner:
            await runner.stop()
            _active_runs.pop(chat_id, None)
            return runner
    return None


HELP_TEXT = (
    "🤖 *Claude-bot v3 — full Claude Code parity over Telegram*\n\n"
    "*Conversation:*\n"
    "• Send a message → starts a persistent Claude session for this chat\n"
    "• Send another while running → injected as a *steering input*\n"
    "• Send a photo → analyzed + injected into session (Anthropic vision)\n"
    "• Send a voice note → Whisper transcribes + injects text\n"
    "• Every 10 responses: auto-checkpoint + steering check\n\n"
    "*Session control:*\n"
    "`/new <goal>` — start a fresh session with the given goal\n"
    "`/resume <id>` — resume a saved claude session (use `/sessions` to list)\n"
    "`/fork` — fork current session into an independent branch\n"
    "`/cancel` — stop the running session\n"
    "`/status` — session state, checkpoint, current config\n"
    "`/pwd` — show bot cwd + claude cwd + config\n\n"
    "*Directory navigation:*\n"
    "`/cd <path>` — change claude's working directory (restarts session)\n"
    "`/ls [path]` — list directory contents (defaults to claude cwd)\n"
    "`/tree [depth] [path]` — show directory tree (default depth 2)\n"
    "`/here` — add current cwd to `--add-dir` (gives claude write access)\n"
    "`/adddir <path>` — give claude access to an extra directory\n\n"
    "*Claude config (each restarts session):*\n"
    "`/model <name>` — switch model (`sonnet`, `opus`, full name)\n"
    "`/agent <name>` — use a subagent (`Explore`, `Plan`, `general-purpose`)\n"
    "`/effort <low|medium|high|xhigh|max>` — reasoning effort\n\n"
    "*Tools (one-shot, no session):*\n"
    "`/imagine <prompt>` — generate image (MiniMax)\n"
    "`/speak <text>` — text-to-speech (MiniMax)\n"
)


async def handle_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/new [goal]` — always show the menu + start a session (goal optional)."""
    chat_id = update.message.chat_id
    raw = update.message.text.split(maxsplit=1)
    goal = raw[1].strip() if len(raw) > 1 else ""
    await _clear_pending_goal(chat_id)  # any prior pending state is now superseded
    if not goal:
        # Show command menu + ask for goal interactively
        await update.message.reply_text(
            HELP_TEXT + "\n\n💬 Reply with your goal for the new session\n"
            "_(or send `/new <goal>` in one message to skip this prompt)_",
            do_quote=True, parse_mode=ParseMode.MARKDOWN,
        )
        await _set_pending_goal(chat_id)
        return

    await update.message.reply_text(
        f"🆕 starting new session with goal: {goal}",
        do_quote=True,
    )
    log_event("new_session", chat_id, {"goal": goal, "via": "inline"})

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    await _stop_runner(chat_id)
    async with _runs_lock:
        runner = ClaudeRunner(chat_id, _steer_manager)
        _active_runs[chat_id] = runner
        await runner.start(goal, send_fn, update.message.chat)


async def handle_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/resume <session_id>` — resume a saved claude session."""
    chat_id = update.message.chat_id
    raw = update.message.text.split(maxsplit=1)
    if len(raw) < 2 or not raw[1].strip():
        await update.message.reply_text(
            "Usage: `/resume <claude-session-id>`\n"
            "Use `/sessions` to list available sessions.",
            do_quote=True, parse_mode=ParseMode.MARKDOWN,
        )
        return
    target = raw[1].strip()

    await update.message.reply_text(f"↩️ resuming session `{target[:8]}…`", do_quote=True)
    log_event("resume_session", chat_id, {"target": target})

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    await _stop_runner(chat_id)
    async with _runs_lock:
        runner = ClaudeRunner(chat_id, _steer_manager)
        runner.config["resume"] = target
        _active_runs[chat_id] = runner
        # Start with no goal — claude loads the existing session
        await runner.start("(resumed session — continue from where you left off)", send_fn, update.message.chat)


async def handle_fork(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/fork` — fork current session into a new one (keeps history, new id)."""
    chat_id = update.message.chat_id
    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if not runner:
        await update.message.reply_text("no active session to fork", do_quote=True)
        return
    if not runner.claude_session_id:
        await update.message.reply_text("session hasn't initialized yet — wait a moment", do_quote=True)
        return

    log_event("fork_session", chat_id, {"from": runner.claude_session_id})
    await update.message.reply_text("🍴 forking session…", do_quote=True)

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    # Capture old session id, then restart with --resume + --fork-session
    old_id = runner.claude_session_id
    await _stop_runner(chat_id)
    async with _runs_lock:
        runner = ClaudeRunner(chat_id, _steer_manager)
        runner.config["resume"] = old_id
        runner.config["fork_session"] = True
        _active_runs[chat_id] = runner
        await runner.start("(forked — independent branch)", send_fn, update.message.chat)


async def handle_set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/model <name>` — switch model. Restarts session."""
    chat_id = update.message.chat_id
    raw = update.message.text.split(maxsplit=1)
    if len(raw) < 2 or not raw[1].strip():
        await update.message.reply_text(
            "Usage: `/model <model-name>`\n"
            "Examples: `claude-sonnet-4-5-20250929`, `claude-opus-4-1`, `sonnet`, `opus`",
            do_quote=True, parse_mode=ParseMode.MARKDOWN,
        )
        return
    new_model = raw[1].strip()

    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if not runner:
        await update.message.reply_text("no active session — send a message first", do_quote=True)
        return

    log_event("set_model", chat_id, {"from": runner.config["model"], "to": new_model})

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    await runner.restart_with_config({"model": new_model}, send_fn)


async def handle_set_agent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/agent <name>` — use a specific subagent. Restarts session."""
    chat_id = update.message.chat_id
    raw = update.message.text.split(maxsplit=1)
    if len(raw) < 2 or not raw[1].strip():
        await update.message.reply_text(
            "Usage: `/agent <agent-name>`\n"
            "Example: `/agent Explore`, `/agent Plan`, `/agent general-purpose`\n"
            "Pass `/agent default` to clear.",
            do_quote=True, parse_mode=ParseMode.MARKDOWN,
        )
        return
    name = raw[1].strip()
    new_agent = None if name.lower() in ("default", "none", "clear") else name

    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if not runner:
        await update.message.reply_text("no active session — send a message first", do_quote=True)
        return

    log_event("set_agent", chat_id, {"agent": new_agent})

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    await runner.restart_with_config({"agent": new_agent}, send_fn)


async def handle_set_effort(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/effort <level>` — reasoning effort. Restarts session."""
    chat_id = update.message.chat_id
    raw = update.message.text.split(maxsplit=1)
    if len(raw) < 2 or not raw[1].strip():
        await update.message.reply_text(
            "Usage: `/effort <low|medium|high|xhigh|max>`",
            do_quote=True, parse_mode=ParseMode.MARKDOWN,
        )
        return
    level = raw[1].strip().lower()
    if level not in ("low", "medium", "high", "xhigh", "max"):
        await update.message.reply_text(f"❌ invalid effort level: {level}", do_quote=True)
        return

    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if not runner:
        await update.message.reply_text("no active session — send a message first", do_quote=True)
        return

    log_event("set_effort", chat_id, {"effort": level})

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    await runner.restart_with_config({"effort": level}, send_fn)


async def handle_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/sessions` — list recent claude sessions from ~/.claude/projects/."""
    chat_id = update.message.chat_id
    if not CLAUDE_PROJECTS_DIR.exists():
        await update.message.reply_text("no claude session history found", do_quote=True)
        return

    # Find session JSONL files (each is a saved session)
    sessions = []
    for jsonl in CLAUDE_PROJECTS_DIR.rglob("*.jsonl"):
        try:
            stat = jsonl.stat()
            if stat.st_size == 0:
                continue
            # First line of session JSONL has session metadata
            with open(jsonl) as f:
                first_line = f.readline()
            if not first_line.strip():
                continue
            meta = json.loads(first_line)
            if meta.get("type") != "user":
                continue
            sid = meta.get("sessionId", jsonl.stem)
            msg = meta.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(b.get("text","") for b in content if b.get("type") == "text")
            preview = (content or "(empty)")[:80].replace("\n", " ")
            sessions.append({
                "id": sid,
                "ts": stat.st_mtime,
                "size": stat.st_size,
                "preview": preview,
            })
        except Exception:
            continue

    if not sessions:
        await update.message.reply_text("no parseable sessions found", do_quote=True)
        return

    # Sort newest first, limit to 15
    sessions.sort(key=lambda s: s["ts"], reverse=True)
    sessions = sessions[:15]

    lines = [f"📂 *Recent sessions* ({len(sessions)}):\n"]
    for s in sessions:
        from datetime import datetime as _dt
        age = _dt.fromtimestamp(s["ts"]).strftime("%Y-%m-%d %H:%M")
        # s['preview'] is user-controlled (first line of session transcript).
        # Escape it so underscore/asterisk/bracket inside preview can't
        # break the entire Markdown message ("Can't find end of entity").
        lines.append(f"`{s['id'][:8]}`  {age}  {md_escape(s['preview'])}")

    lines.append("\nUse `/resume <id>` to continue one.")
    await safe_send(update, "\n".join(lines))


async def handle_pwd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/pwd` — show bot cwd + claude's session cwd + config."""
    chat_id = update.message.chat_id
    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    bot_cwd = os.getcwd()
    info = f"📍 *Bot cwd*: `{bot_cwd}`\n"
    if runner:
        c = runner.config
        session_cwd = c.get("cwd") or bot_cwd
        info += (
            f"\n📂 *Claude cwd*: `{session_cwd}`"
            f"\n\n*Session config*:"
            f"\n• model: `{c['model']}`"
            f"\n• agent: `{c.get('agent') or '(default)'}`"
            f"\n• add-dirs: {c.get('add_dirs') or '[]'}"
            f"\n• effort: `{c.get('effort') or '(default)'}`"
            f"\n• session-id: `{runner.session_id[:8]}…`"
            f"\n• resume: `{c.get('resume') or '(new)'}`"
        )
    await update.message.reply_text(info, do_quote=True, parse_mode=ParseMode.MARKDOWN)


async def handle_cd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/cd <path>` — change claude's working directory. Restarts session.
    Accepts absolute paths, ~ paths, and .. relative traversal.
    If no path given, goes to bot's cwd.
    """
    chat_id = update.message.chat_id
    raw = update.message.text.split(maxsplit=1)
    target = raw[1].strip() if len(raw) > 1 else ""

    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if not runner:
        await update.message.reply_text(
            "no active session — send a message first to start one",
            do_quote=True,
        )
        return

    if not target:
        # /cd with no arg → bot cwd
        target = os.getcwd()
    else:
        # Resolve ~ and relative paths against current session cwd
        target = os.path.expanduser(target)
        if not os.path.isabs(target):
            target = os.path.join(runner.config.get("cwd") or os.getcwd(), target)
        target = os.path.abspath(target)

    if not os.path.isdir(target):
        await update.message.reply_text(path_not_found_message(target), do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return

    if target == runner.config.get("cwd"):
        await update.message.reply_text(f"📂 already there: `{target}`", do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return

    log_event("cd", chat_id, {"from": runner.config.get("cwd"), "to": target})

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    await runner.restart_with_config({"cwd": target}, send_fn)


async def handle_ls(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/ls [path]` — list directory contents. Defaults to claude's cwd."""
    chat_id = update.message.chat_id
    raw = update.message.text.split(maxsplit=1)
    arg = raw[1].strip() if len(raw) > 1 else ""

    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    base = (runner.config.get("cwd") if runner else None) or os.getcwd()

    if arg:
        target = os.path.expanduser(arg)
        if not os.path.isabs(target):
            target = os.path.join(base, target)
        target = os.path.abspath(target)
    else:
        target = base

    if not os.path.isdir(target):
        await update.message.reply_text(path_not_found_message(target), do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return

    try:
        entries = sorted(os.listdir(target))
    except PermissionError:
        await update.message.reply_text(f"❌ permission denied: `{target}`", do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return

    # Mark dirs vs files, show sizes for files
    lines = [f"📁 `{target}`\n"]
    dirs = []
    files = []
    for name in entries:
        if name.startswith("."):
            continue  # skip hidden by default (user can ls -a separately if they want)
        full = os.path.join(target, name)
        try:
            if os.path.isdir(full):
                # Filenames on disk are user-controlled; escape markdown
                # chars so an underscore in a file name doesn't break parsing.
                dirs.append(f"📂 {md_escape(name)}/")
            else:
                size = os.path.getsize(full)
                files.append(f"📄 {md_escape(name)}  ({_format_size(size)})")
        except OSError:
            continue

    # Cap at 60 entries to keep telegram message short
    shown_dirs = dirs[:30]
    shown_files = files[:30]
    lines.extend(shown_dirs)
    if shown_dirs and shown_files:
        lines.append("")
    lines.extend(shown_files)

    if len(dirs) > 30 or len(files) > 30:
        lines.append(f"\n_…showing 30 of {len(dirs) + len(files)} entries_")
    if not dirs and not files:
        lines.append("_(empty)_")

    await safe_send(update, "\n".join(lines))


async def handle_tree(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/tree [depth] [path]` — show directory tree. Default depth=2, path=cwd."""
    chat_id = update.message.chat_id
    parts = update.message.text.split()
    # /tree [depth] [path]
    depth = 2
    arg = ""
    if len(parts) > 1:
        try:
            depth = int(parts[1])
            depth = min(max(depth, 1), 5)  # cap at 5
        except ValueError:
            arg = parts[1]
    if len(parts) > 2:
        arg = parts[2]

    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    base = (runner.config.get("cwd") if runner else None) or os.getcwd()

    target = arg or base
    target = os.path.expanduser(target)
    if not os.path.isabs(target):
        target = os.path.join(base, target)
    target = os.path.abspath(target)

    if not os.path.isdir(target):
        await update.message.reply_text(path_not_found_message(target), do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return

    lines = [f"🌳 `{target}` (depth {depth})\n"]
    _tree_walk(target, "", depth, lines, count=[0])
    if lines[-1].startswith("…"):
        pass
    else:
        total = lines[-1] if "entries" in lines[-1] else None
    if len(lines) > 1 and "entries" not in (lines[-1] or ""):
        lines.append(f"\n_({len(lines)-1} entries)_")

    # Cap output
    output = "\n".join(lines)
    if len(output) > 3500:
        output = output[:3500] + "\n\n_…truncated_"
    await safe_send(update, output)


def _tree_walk(path: str, prefix: str, depth: int, lines: list, count: list, max_entries=80):
    if depth == 0:
        return
    if count[0] > max_entries:
        lines.append(f"{prefix}…(truncated)")
        return
    try:
        entries = sorted(
            [e for e in os.listdir(path) if not e.startswith(".")],
            key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()),
        )
    except PermissionError:
        lines.append(f"{prefix}❌ permission denied")
        return
    for i, name in enumerate(entries):
        if count[0] > max_entries:
            lines.append(f"{prefix}…(truncated)")
            return
        full = os.path.join(path, name)
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        is_dir = os.path.isdir(full)
        # Escape filenames: underscores/asterisks in names are common
        # (e.g. IELTS_Trainer, [role]) and break the whole message otherwise.
        display = f"{md_escape(name)}/" if is_dir else md_escape(name)
        lines.append(f"{prefix}{connector}{display}")
        count[0] += 1
        if is_dir and depth > 1:
            extension = "    " if is_last else "│   "
            _tree_walk(full, prefix + extension, depth - 1, lines, count, max_entries)


def _format_size(n: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"


async def handle_here(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/here` — shortcut for `/adddir .` — adds the session cwd to --add-dir.
    Useful after `/cd` to give claude write access to the directory."""
    chat_id = update.message.chat_id
    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if not runner:
        await update.message.reply_text("no active session", do_quote=True)
        return
    cwd = runner.config.get("cwd") or os.getcwd()
    if cwd in runner.config["add_dirs"]:
        await update.message.reply_text(f"📁 cwd already in add-dirs: `{cwd}`", do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return

    new_dirs = runner.config["add_dirs"] + [cwd]
    log_event("here", chat_id, {"path": cwd, "all_dirs": new_dirs})

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    await runner.restart_with_config({"add_dirs": new_dirs}, send_fn)


async def handle_adddir(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """`/adddir <path>` — give claude access to a directory. Restarts session.
    Aliases: /adddir, /adddirs, /cd <path> (when no session)"""
    chat_id = update.message.chat_id
    raw = update.message.text.split(maxsplit=1)
    if len(raw) < 2 or not raw[1].strip():
        await update.message.reply_text(
            "Usage: `/adddir <absolute path>`\n"
            "_Tip: use `/here` to add the current session cwd._",
            do_quote=True, parse_mode=ParseMode.MARKDOWN,
        )
        return
    path = raw[1].strip()
    if not os.path.isabs(path):
        # Resolve relative to session cwd if available, else bot cwd
        async with _runs_lock:
            runner = _active_runs.get(chat_id)
        base = (runner.config.get("cwd") if runner else None) or os.getcwd()
        path = os.path.abspath(os.path.join(base, path))
    if not os.path.isdir(path):
        await update.message.reply_text(path_not_found_message(path), do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return

    async with _runs_lock:
        runner = _active_runs.get(chat_id)
    if not runner:
        await update.message.reply_text("no active session — send a message first to start one", do_quote=True)
        return
    if path in runner.config["add_dirs"]:
        await update.message.reply_text(f"📁 already in add-dirs: `{path}`", do_quote=True, parse_mode=ParseMode.MARKDOWN)
        return

    new_dirs = runner.config["add_dirs"] + [path]
    log_event("add_dir", chat_id, {"path": path, "all_dirs": new_dirs})

    async def send_fn(text):
        try:
            await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text, do_quote=False)

    await runner.restart_with_config({"add_dirs": new_dirs}, send_fn)


# ─────────────────────────────────────────────────────────────
# Photo + voice handlers — feed into active session
# ─────────────────────────────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Download photo, base64 it, inject as a message with image content."""
    chat_id = update.message.chat_id
    if not update.message.photo:
        return
    # Clear (not consume) the pending /new goal so the next typed text still wins
    # as the goal. Photo/voice don't act as a goal themselves.
    await _clear_pending_goal(chat_id)
    await update.message.reply_text("📷 downloading…", do_quote=True)

    try:
        photo = update.message.photo[-1]  # largest size
        file = await context.bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode()
        img_mime = sniff_image_mime(img_bytes)

        async with _runs_lock:
            runner = _active_runs.get(chat_id)

        if runner and runner._running:
            # Inject as a multi-part message: text description + image
            caption = update.message.caption or "(user sent an image, please analyze)"
            content = [
                {"type": "text", "text": caption},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img_mime,
                        "data": img_b64,
                    },
                },
            ]
            runner._pending_inject.append(content)  # type: ignore[arg-type]
            runner._inject_event.set()
            log_event("user_image", chat_id, {"caption": caption, "size": len(img_bytes), "mime": img_mime})
            await update.message.reply_text("↪️ image injected into session", do_quote=True)
        else:
            # No active session — start one with the image as initial context
            caption = update.message.caption or "analyze this image"
            content = [
                {"type": "text", "text": caption},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img_mime,
                        "data": img_b64,
                    },
                },
            ]
            # Use Anthropic-format content directly (claude supports it)
            await update.message.reply_text("🤖 starting session with image…", do_quote=True)
            # For first message, we need to send multi-part content via stream-json
            # stream-json accepts {"type":"user","message":{"role":"user","content":[...]}}
            async def send_fn(text):
                try:
                    await update.message.reply_text(text, do_quote=False, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await update.message.reply_text(text, do_quote=False)
            await _stop_runner(chat_id)
            async with _runs_lock:
                runner = ClaudeRunner(chat_id, _steer_manager)
                _active_runs[chat_id] = runner
                await runner.start_multimodal(content, send_fn, update.message.chat)
            log_event("user_image_start", chat_id, {"caption": caption, "size": len(img_bytes)})
    except Exception as e:
        log_event("photo_error", chat_id, {"error": str(e)})
        await update.message.reply_text(f"photo error: {e}", do_quote=True)


def _load_whisper():
    """Lazy-load openai-whisper from the hermes-agent venv (singleton).
    Returns the whisper module or None if not importable.
    """
    global _whisper_module
    if _whisper_module is not None:
        return _whisper_module
    if HERMES_VENV_SITE not in sys.path:
        sys.path.insert(0, HERMES_VENV_SITE)
    try:
        import whisper  # type: ignore
        _whisper_module = whisper
        return whisper
    except Exception as e:
        logging.warning(f"local whisper import failed: {e}")
        return None


def _load_whisper_model():
    """Lazy-load the whisper model (singleton, cached for process lifetime).
    Returns (model, error_str). First load downloads model from cache (~75MB tiny).
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model, None
    whisper = _load_whisper()
    if whisper is None:
        return None, "openai-whisper not importable from hermes-agent venv"
    try:
        # Force CPU since this server has no GPU
        os.environ.setdefault("WHISPER_NO_GPU", "1")
        logging.info(f"loading whisper model: {WHISPER_LOCAL_MODEL}")
        t0 = time.time()
        _whisper_model = whisper.load_model(WHISPER_LOCAL_MODEL)
        logging.info(f"whisper model loaded in {time.time() - t0:.1f}s")
        return _whisper_model, None
    except Exception as e:
        return None, f"whisper.load_model({WHISPER_LOCAL_MODEL!r}) failed: {e}"


def transcribe_audio(audio_bytes: bytes, suffix: str = ".ogg") -> str:
    """Transcribe audio to text. Tries local whisper first, falls back to OpenAI API.

    Returns the transcribed text. Raises on failure.
    """
    # Path 1: local openai-whisper (free, private, no key, no network)
    if WHISPER_LOCAL:
        model, err = _load_whisper_model()
        if model is not None:
            try:
                # whisper.transcribe needs a file path; write to a temp file
                # because it uses ffmpeg under the hood for .ogg/.webm/etc.
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                try:
                    t0 = time.time()
                    result = model.transcribe(
                        tmp_path,
                        fp16=False,  # CPU only
                        verbose=None,
                    )
                    text = (result.get("text") or "").strip()
                    logging.info(
                        f"whisper local: {len(audio_bytes)}B in {time.time() - t0:.1f}s "
                        f"-> {len(text)} chars"
                    )
                    return text
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            except Exception as e:
                logging.warning(f"local whisper transcribe failed: {e}")
                # fall through to API
        else:
            logging.info(f"local whisper unavailable ({err}), trying API")

    # Path 2: OpenAI Whisper API (paid, requires key)
    api_key = WHISPER_API_KEY or OPENAI_API_KEY
    if api_key:
        resp = requests.post(
            f"{OPENAI_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (f"voice{suffix}", audio_bytes, f"audio/{suffix.lstrip('.')}")},
            data={"model": "whisper-1", "language": "en"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()

    raise RuntimeError(
        "no transcription backend: local whisper failed AND "
        "WHISPER_API_KEY/OPENAI_API_KEY not set in .env"
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Download voice, transcribe via Whisper, inject text into session."""
    chat_id = update.message.chat_id
    if not update.message.voice:
        return
    # Clear (not consume) the pending /new goal so the next typed text still wins
    await _clear_pending_goal(chat_id)

    # If user has explicitly disabled local whisper AND has no API key, fail fast
    if not WHISPER_LOCAL and not (WHISPER_API_KEY or OPENAI_API_KEY):
        await update.message.reply_text(
            "voice transcription needs WHISPER_API_KEY or OPENAI_API_KEY in .env "
            "(local whisper disabled via WHISPER_LOCAL=0)",
            do_quote=True,
        )
        return
    # Show a quick "transcribing" hint unless local whisper loaded super fast
    await update.message.reply_text("🎙 transcribing…", do_quote=True)
    try:
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        ogg_bytes = buf.getvalue()

        # Transcribe (local first, API fallback). .ogg for Telegram voice notes.
        text = await asyncio.to_thread(transcribe_audio, ogg_bytes, ".ogg")
        if not text:
            await update.message.reply_text("(no speech detected)", do_quote=True)
            return

        await update.message.reply_text(f"🎙 {text}", do_quote=True)
        log_event("user_voice", chat_id, {
            "text": text, "duration": voice.duration,
            "backend": "local" if WHISPER_LOCAL else "api",
        })

        async with _runs_lock:
            runner = _active_runs.get(chat_id)
        if runner and runner._running:
            await runner.inject_prompt(text)
            await update.message.reply_text("↪️ injected into session", do_quote=True)
        else:
            await update.message.reply_text("🤖 starting session with voice…", do_quote=True)
            async def send_fn(t):
                try:
                    await update.message.reply_text(t, do_quote=False, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await update.message.reply_text(t, do_quote=False)
            await _stop_runner(chat_id)
            async with _runs_lock:
                runner = ClaudeRunner(chat_id, _steer_manager)
                _active_runs[chat_id] = runner
                await runner.start(text, send_fn, update.message.chat)
    except Exception as e:
        log_event("voice_error", chat_id, {"error": str(e)})
        await update.message.reply_text(f"voice error: {e}", do_quote=True)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, do_quote=True, parse_mode=ParseMode.MARKDOWN)


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

async def _pre_command_clear_pending(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Group -1 pre-handler: any slash command (except /new itself) clears a pending /new goal.

    This prevents the next text message from being silently consumed as a goal
    after the user runs e.g. /status or /ls in the middle of an interactive /new.
    """
    if not update.message or not update.message.text:
        return
    if not update.message.text.startswith("/"):
        return
    # Don't clear for /new — handle_new sets the flag itself
    first_token = update.message.text.split(maxsplit=1)[0].split("@")[0].lower()
    if first_token == "/new":
        return
    await _clear_pending_goal(update.message.chat_id)


def main():
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set", flush=True)
        return
    if not MINIMAX_API_KEY:
        print("WARN: MINIMAX_API_KEY not set — steering manager + image + TTS disabled", flush=True)

    init_db()

    app = Application.builder().token(BOT_TOKEN).build()

    # Group -1: clear pending /new goal whenever user runs any other slash command.
    # Without this, the next text message would silently become a new-session goal
    # after the user runs e.g. /status mid-prompt.
    app.add_handler(MessageHandler(filters.COMMAND, _pre_command_clear_pending), group=-1)

    # Order matters: more specific commands first
    app.add_handler(CommandHandler(["help", "start"], handle_help))
    app.add_handler(CommandHandler(["status", "st"], handle_status))
    app.add_handler(CommandHandler(["cancel", "stop"], handle_cancel))
    app.add_handler(CommandHandler(["new"], handle_new))
    app.add_handler(CommandHandler(["resume"], handle_resume))
    app.add_handler(CommandHandler(["fork"], handle_fork))
    app.add_handler(CommandHandler(["sessions", "history"], handle_sessions))
    app.add_handler(CommandHandler(["cd"], handle_cd))
    app.add_handler(CommandHandler(["ls", "ll"], handle_ls))
    app.add_handler(CommandHandler(["tree"], handle_tree))
    app.add_handler(CommandHandler(["here"], handle_here))
    app.add_handler(CommandHandler(["adddir", "adddirs"], handle_adddir))
    app.add_handler(CommandHandler(["model", "m"], handle_set_model))
    app.add_handler(CommandHandler(["agent", "a"], handle_set_agent))
    app.add_handler(CommandHandler(["effort"], handle_set_effort))
    app.add_handler(CommandHandler(["pwd", "where"], handle_pwd))
    app.add_handler(CommandHandler(["imagine", "image"], handle_image_gen))
    app.add_handler(CommandHandler(["speak", "tts"], handle_tts))
    # Media + text
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print(f"Bot v3 starting…", flush=True)
    print(f"  CLAUDE_MODEL      = {CLAUDE_MODEL}", flush=True)
    print(f"  CHECKPOINT_EVERY  = {CHECKPOINT_EVERY}", flush=True)
    print(f"  Training log dir  = {LOG_DIR}", flush=True)
    print(f"  Session state dir = {SESSION_DIR}", flush=True)
    print(f"  DB                = {DB_PATH}", flush=True)
    # Voice transcription status (helps debug "voice error: ..." at 3am)
    if WHISPER_LOCAL and (WHISPER_API_KEY or OPENAI_API_KEY):
        backend = "local+api"
    elif WHISPER_LOCAL:
        backend = "local"
    elif WHISPER_API_KEY or OPENAI_API_KEY:
        backend = "api"
    else:
        backend = "DISABLED"
    print(f"  Voice backend     = {backend} (model={WHISPER_LOCAL_MODEL})", flush=True)

    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
