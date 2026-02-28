/**
 * Conut AI — Chief of Operations · Chat Frontend
 * Handles messaging, API integration, and UI state.
 */

// ── State ────────────────────────────────────────────────────────
const state = {
    history: [],
    isLoading: false,
    sessionId: null,  // OpenClaw session ID for multi-turn conversation
};

// ── DOM refs ─────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const messagesContainer = $("#messagesContainer");
const welcomeScreen     = $("#welcomeScreen");
const chatForm          = $("#chatForm");
const messageInput      = $("#messageInput");
const sendBtn           = $("#sendBtn");
const statusDot         = $("#statusDot");
const statusText        = $("#statusText");
const sidebar           = $("#sidebar");
const menuToggle        = $("#menuToggle");
const newChatBtn        = $("#newChatBtn");

// ── Markdown config ──────────────────────────────────────────────
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false,
});

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    bindEvents();
    messageInput.focus();
});

// ── Health check ─────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();

        if (data.status === "ok") {
            statusDot.className = "status-dot connected";
            statusText.textContent = "Connected to OpenClaw";
        } else if (data.status === "error") {
            statusDot.className = "status-dot error";
            statusText.textContent = "OpenClaw not found";
            showError(data.message || "OpenClaw CLI not found. Make sure Claude Code is installed.");
        }
    } catch {
        statusDot.className = "status-dot error";
        statusText.textContent = "Server offline";
        showError("Cannot reach backend server. Make sure it's running: python3 backend/server.py");
    }
}

// ── Events ───────────────────────────────────────────────────────
function bindEvents() {
    // Form submit
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        sendMessage();
    });

    // Enable/disable send button
    messageInput.addEventListener("input", () => {
        sendBtn.disabled = !messageInput.value.trim();
        autoResize(messageInput);
    });

    // Enter to send (Shift+Enter for newline)
    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Sidebar toggle (mobile)
    menuToggle.addEventListener("click", () => {
        sidebar.classList.toggle("open");
    });

    // Close sidebar when clicking outside (mobile)
    document.addEventListener("click", (e) => {
        if (sidebar.classList.contains("open") &&
            !sidebar.contains(e.target) &&
            !menuToggle.contains(e.target)) {
            sidebar.classList.remove("open");
        }
    });

    // New chat
    newChatBtn.addEventListener("click", resetChat);

    // Quick action buttons (welcome screen + sidebar)
    document.querySelectorAll("[data-question]").forEach((btn) => {
        btn.addEventListener("click", () => {
            messageInput.value = btn.dataset.question;
            sendBtn.disabled = false;
            sidebar.classList.remove("open");
            sendMessage();
        });
    });
}

// ── Send message ─────────────────────────────────────────────────
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || state.isLoading) return;

    // Hide welcome, show message
    welcomeScreen.style.display = "none";
    appendMessage("user", text);

    // Clear input
    messageInput.value = "";
    sendBtn.disabled = true;
    autoResize(messageInput);

    // Show typing indicator
    state.isLoading = true;
    const typingEl = showTyping();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                history: state.history,
                session_id: state.sessionId,
            }),
        });

        // Remove typing indicator
        typingEl.remove();

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Unknown error" }));
            throw new Error(err.detail || `Server error (${res.status})`);
        }

        const data = await res.json();

        // Store OpenClaw session ID for multi-turn conversation
        if (data.session_id) {
            state.sessionId = data.session_id;
        }

        // Append AI response
        appendMessage("assistant", data.response, data.tools_used);

        // Update history
        state.history.push({ role: "user", content: text });
        state.history.push({ role: "assistant", content: data.response });

    } catch (err) {
        typingEl.remove();
        showError(err.message);
    } finally {
        state.isLoading = false;
        messageInput.focus();
    }
}

// ── Append message to chat ───────────────────────────────────────
function appendMessage(role, content, toolsUsed = []) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "🍩";
    avatar.setAttribute("aria-hidden", "true");

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (role === "assistant") {
        bubble.innerHTML = marked.parse(content);
    } else {
        bubble.textContent = content;
    }

    contentDiv.appendChild(bubble);

    // Tools used badge
    if (toolsUsed && toolsUsed.length > 0) {
        const badge = document.createElement("div");
        badge.className = "tools-badge";
        const toolNames = toolsUsed.map(formatToolName).join(", ");
        badge.innerHTML = `<i class="fa-solid fa-gear"></i> Used: ${toolNames}`;
        contentDiv.appendChild(badge);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

// ── Typing indicator ─────────────────────────────────────────────
function showTyping() {
    const el = document.createElement("div");
    el.className = "typing-indicator";
    el.innerHTML = `
        <div class="message-avatar" style="background: var(--accent-light);" aria-hidden="true">🍩</div>
        <div class="typing-bubble">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
            <span class="typing-label">Analyzing...</span>
        </div>
    `;
    messagesContainer.appendChild(el);
    scrollToBottom();
    return el;
}

// ── Error display ────────────────────────────────────────────────
function showError(message) {
    const el = document.createElement("div");
    el.className = "error-banner";
    el.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> ${escapeHtml(message)}`;
    messagesContainer.appendChild(el);
    scrollToBottom();

    // Auto-dismiss after 10s
    setTimeout(() => {
        if (el.parentNode) {
            el.style.opacity = "0";
            el.style.transition = "opacity 0.3s";
            setTimeout(() => el.remove(), 300);
        }
    }, 10000);
}

// ── Reset chat ───────────────────────────────────────────────────
function resetChat() {
    state.history = [];
    state.sessionId = null;  // Start fresh OpenClaw session
    messagesContainer.innerHTML = "";
    messagesContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
    sidebar.classList.remove("open");
    messageInput.focus();
}

// ── Helpers ──────────────────────────────────────────────────────
function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}

function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function formatToolName(name) {
    return name
        .replace(/^get_/, "")
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());
}
