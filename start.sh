#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Conut AI — Chief of Operations · Startup Script
# Run this from the Hackaton/ root directory.
# Requires: Python 3.10+, Claude Code (OpenClaw) installed
# ═══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  🍩 Conut AI — Chief of Operations"
echo "  ─────────────────────────────────"
echo ""

# Check for OpenClaw (Claude Code CLI)
if ! command -v claude &> /dev/null; then
    echo "  ✗ OpenClaw (Claude Code) not found in PATH."
    echo "    Install it first: npm install -g @anthropic-ai/claude-code"
    echo "    Or set CLAUDE_CMD in backend/.env to the correct path."
    echo ""
    exit 1
fi

echo "  ✓ OpenClaw found: $(claude --version 2>/dev/null || echo 'unknown version')"

# Create .env if missing
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "  ✓ Created backend/.env from template"
fi

# Check for venv
if [ ! -d "venv" ]; then
    echo "  📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "  📦 Activating virtual environment..."
source venv/bin/activate

echo "  📦 Installing dependencies..."
pip install -q -r backend/requirements.txt

echo ""
echo "  🚀 Starting server..."
echo "  → Open http://localhost:8000 in your browser"
echo "  → Press Ctrl+C to stop"
echo ""

python3 backend/server.py
