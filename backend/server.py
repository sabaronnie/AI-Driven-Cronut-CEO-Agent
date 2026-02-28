#!/usr/bin/env python3
"""
Conut AI Chief of Operations — Backend API Server

Bridges the frontend chat UI to OpenClaw (Claude Code CLI).
All AI reasoning, tool execution, and skill routing is handled by OpenClaw.
This server just forwards messages and returns responses.

Usage:
    python3 backend/server.py
    → Opens http://localhost:8000
"""

import os
import sys
import json
import subprocess
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / "backend" / ".env"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

load_dotenv(ENV_PATH)

# ── Config ───────────────────────────────────────────────────────────────
CLAUDE_CMD = os.getenv("CLAUDE_CMD", "claude")  # Path to claude CLI
PORT = int(os.getenv("PORT", "8000"))

# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Conut AI Chief of Operations", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    tools_used: list[str] = []
    session_id: Optional[str] = None


# ── OpenClaw integration ─────────────────────────────────────────────────
def get_venv_env() -> dict:
    """
    Build environment dict that activates the project venv.
    This ensures Claude Code's Bash tool uses the venv Python
    (where pandas, numpy, scikit-learn are installed).
    """
    env = os.environ.copy()

    # Detect venv bin directory (Windows: Scripts, Unix: bin)
    venv_scripts = PROJECT_ROOT / "venv" / "Scripts"  # Windows
    venv_bin = PROJECT_ROOT / "venv" / "bin"           # macOS/Linux

    if venv_scripts.exists():
        bin_dir = str(venv_scripts)
    elif venv_bin.exists():
        bin_dir = str(venv_bin)
    else:
        # No venv found (e.g. Docker) — still pass API key if set
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        return env

    # Prepend venv to PATH so 'python3' and 'python' resolve to venv
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(PROJECT_ROOT / "venv")

    # Pass through ANTHROPIC_API_KEY if set (Docker uses this)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key

    return env


def call_openclaw(message: str, session_id: Optional[str] = None) -> dict:
    """
    Call OpenClaw (Claude Code CLI) with a message.

    First message starts a new session.
    Follow-up messages use --resume to continue the conversation,
    so OpenClaw remembers full context (no need to resend history).

    Returns: {"response": str, "session_id": str, "tools_used": list}
    """
    cmd = [CLAUDE_CMD]

    # Resume existing session or start new one
    if session_id:
        cmd += ["--resume", session_id]

    cmd += [
        "-p", message,
        "--output-format", "json",
        "--allowedTools", "Bash,Read"
    ]

    # Build env with venv activated so tools find pandas/numpy/sklearn
    env = get_venv_env()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "OpenClaw returned an error"
            # If resume failed (session expired), try without resume
            if session_id and ("session" in error_msg.lower() or "not found" in error_msg.lower()):
                return call_openclaw(message, session_id=None)
            raise RuntimeError(error_msg)

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("OpenClaw returned empty response")

        # Parse JSON output from Claude Code
        # Format: {"type":"result","result":"...","session_id":"...","cost_usd":...}
        try:
            data = json.loads(stdout)
            response_text = data.get("result", stdout)
            new_session_id = data.get("session_id", session_id)
        except json.JSONDecodeError:
            # If not JSON, treat as plain text response
            response_text = stdout
            new_session_id = session_id

        # Try to detect tool usage from the response text
        tools_used = []
        tool_keywords = {
            "combo": "Combo Recommendations",
            "demand": "Demand Forecast",
            "expansion": "Expansion Assessment",
            "staffing": "Staffing Recommendation",
            "coffee": "Coffee & Milkshake Analysis",
            "location": "Branch Location Recommendation",
        }
        response_lower = response_text.lower() if response_text else ""
        for keyword, label in tool_keywords.items():
            if f"get_{keyword}" in response_lower or f"run_tool.py" in response_lower:
                tools_used.append(label)

        return {
            "response": response_text,
            "session_id": new_session_id,
            "tools_used": tools_used,
        }

    except subprocess.TimeoutExpired:
        raise RuntimeError("OpenClaw request timed out (120s). Try a simpler question.")
    except FileNotFoundError:
        raise RuntimeError(
            f"OpenClaw CLI not found (tried: '{CLAUDE_CMD}'). "
            "Make sure Claude Code is installed and in PATH, "
            "or set CLAUDE_CMD in backend/.env"
        )


# ── API routes ───────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/api/health")
async def health_check():
    """Check if OpenClaw CLI is available."""
    try:
        result = subprocess.run(
            [CLAUDE_CMD, "--version"],
            capture_output=True, text=True, timeout=10
        )
        version = result.stdout.strip() or "unknown"
        return {
            "status": "ok",
            "openclaw_version": version,
            "project_root": str(PROJECT_ROOT),
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"OpenClaw CLI not found (tried: '{CLAUDE_CMD}'). Install Claude Code or set CLAUDE_CMD in backend/.env"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Forwards the message to OpenClaw which handles:
    - AI reasoning (Claude)
    - Skill matching
    - Tool execution (run_tool.py)
    - Response generation
    """
    try:
        result = await asyncio.to_thread(
            call_openclaw, request.message, request.session_id
        )
        return ChatResponse(
            response=result["response"],
            tools_used=result["tools_used"],
            session_id=result["session_id"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ── Serve static frontend files ─────────────────────────────────────────
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # Check if OpenClaw is available
    try:
        v = subprocess.run([CLAUDE_CMD, "--version"], capture_output=True, text=True, timeout=10)
        version = v.stdout.strip()
    except FileNotFoundError:
        version = None

    print("\n🍩 Conut AI Chief of Operations")
    print(f"   OpenClaw:  {version if version else '✗ NOT FOUND — install Claude Code or set CLAUDE_CMD in backend/.env'}")
    print(f"   Project:   {PROJECT_ROOT}")
    print(f"   Frontend:  http://localhost:{PORT}")
    print(f"   API docs:  http://localhost:{PORT}/docs\n")

    if not version:
        print("   ⚠  WARNING: OpenClaw CLI not found. Chat will not work until it's installed.\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
