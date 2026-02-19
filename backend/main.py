#!/usr/bin/env python3
"""
JARVIS - Growth Equity Meeting Copilot
Real-time AI assistant for investor meetings.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import httpx
from dotenv import load_dotenv

# Local transcription
from transcriber import LocalTranscriber, AudioConverter, get_transcriber
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import hashlib
import secrets

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "")
GRANOLA_CACHE_PATH = os.path.expanduser("~/Library/Application Support/Granola/cache-v3.json")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")  # Set this to require password
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large-v3

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(title="JARVIS", description="Growth Equity Meeting Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple token store (in production, use Redis or similar)
valid_tokens = set()

def generate_token() -> str:
    """Generate a secure token."""
    return secrets.token_urlsafe(32)

def verify_token(token: str) -> bool:
    """Verify if token is valid."""
    if not APP_PASSWORD:  # No password set = no auth required
        return True
    return token in valid_tokens

def check_auth(token: str = Query(None)) -> bool:
    """Dependency to check authentication."""
    if not APP_PASSWORD:
        return True
    if not token or not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# =============================================================================
# State Management
# =============================================================================

class MeetingState:
    def __init__(self):
        self.transcript: list[dict] = []
        self.insights: list[dict] = []
        self.ui_connections: list[WebSocket] = []
        self.is_recording = False
        self.current_context = ""

    def add_transcript(self, text: str, is_final: bool = False):
        self.transcript.append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "is_final": is_final
        })
        # Keep last 50 utterances for context
        if len(self.transcript) > 50:
            self.transcript = self.transcript[-50:]

    def get_full_transcript(self) -> str:
        return " ".join([t["text"] for t in self.transcript if t.get("is_final")])

    def add_insight(self, insight: dict):
        self.insights.append({
            **insight,
            "timestamp": datetime.now().isoformat()
        })

meeting_state = MeetingState()

# =============================================================================
# Obsidian Integration
# =============================================================================

def search_obsidian(query: str, limit: int = 5) -> list[dict]:
    """Search Obsidian vault for relevant notes."""
    if not OBSIDIAN_VAULT_PATH or not os.path.exists(OBSIDIAN_VAULT_PATH):
        return []

    results = []
    vault_path = Path(OBSIDIAN_VAULT_PATH)

    query_lower = query.lower()

    for md_file in vault_path.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if query_lower in content.lower():
                # Extract relevant snippet
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if query_lower in line.lower():
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        snippet = "\n".join(lines[start:end])
                        results.append({
                            "file": md_file.name,
                            "path": str(md_file.relative_to(vault_path)),
                            "snippet": snippet[:500]
                        })
                        break
                if len(results) >= limit:
                    break
        except Exception:
            continue

    return results

# =============================================================================
# Granola Integration
# =============================================================================

def search_granola(query: str, limit: int = 5) -> list[dict]:
    """Search Granola meeting transcripts."""
    if not os.path.exists(GRANOLA_CACHE_PATH):
        return []

    try:
        with open(GRANOLA_CACHE_PATH, "r") as f:
            cache = json.load(f)

        results = []
        query_lower = query.lower()

        meetings = cache.get("meetings", [])
        for meeting in meetings:
            title = meeting.get("title", "")
            transcript = meeting.get("transcript", "")
            notes = meeting.get("notes", "")

            if query_lower in title.lower() or query_lower in transcript.lower() or query_lower in notes.lower():
                results.append({
                    "title": title,
                    "date": meeting.get("date", ""),
                    "snippet": (transcript or notes)[:500],
                    "participants": meeting.get("participants", [])
                })

                if len(results) >= limit:
                    break

        return results
    except Exception:
        return []

# =============================================================================
# Web Search (for market data)
# =============================================================================

async def web_search(query: str) -> list[dict]:
    """Search the web for market data, comps, etc."""
    # Using a simple approach - in production you'd use a proper search API
    try:
        async with httpx.AsyncClient() as client:
            # This is a placeholder - you'd integrate with a search API
            # For now, return empty and let Claude use its knowledge
            pass
    except Exception:
        pass
    return []

# =============================================================================
# Claude Analysis
# =============================================================================

JARVIS_SYSTEM_PROMPT = """You are JARVIS, a senior growth equity investor with 15+ years of experience across 500+ deals. You're sitting in on a live founder meeting, listening to the conversation in real-time.

Your job is to be the smartest person in the room, whispering insights to your junior colleague (the user).

## Your Personality
- Pattern-matching machine: You've seen every pitch, every business model, every red flag
- Data-driven: You always anchor to numbers, benchmarks, and comps
- Skeptical but fair: You probe claims without being dismissive
- Connected: You know the players, the deals, the market dynamics

## What You Do
1. **FLAG INTERESTING CLAIMS** - When a founder says something notable, flag it
2. **FACT-CHECK IN REAL-TIME** - Cross-reference claims against what you know
3. **SUGGEST QUESTIONS** - Give the user smart follow-up questions to ask
4. **DRAW PARALLELS** - Connect to public comps, recent transactions, similar companies
5. **SPOT RED FLAGS** - Inconsistencies, unrealistic metrics, concerning patterns

## Your Knowledge
- Public SaaS benchmarks (Bessemer, KeyBanc, etc.)
- Recent M&A and funding rounds
- Unit economics frameworks
- GTM efficiency metrics
- Growth patterns by business model (PLG, sales-led, hybrid)

## Output Format
Respond with JSON only:
{
    "type": "insight" | "question" | "flag" | "context",
    "priority": "high" | "medium" | "low",
    "content": "Your insight here",
    "reasoning": "Why this matters (optional)",
    "data_point": "Relevant benchmark or comp (optional)"
}

Keep insights SHORT and ACTIONABLE. The user is in a live meeting - they need quick hits, not essays.

## Example Outputs

Founder says: "Our NRR is 180%"
{
    "type": "context",
    "priority": "high",
    "content": "180% NRR is elite but verify the denominator. Top decile enterprise SaaS is ~130%.",
    "reasoning": "Usage-based can inflate NRR. Ask about gross retention separately.",
    "data_point": "Snowflake: 158%, Datadog: 130%, MongoDB: 120%"
}

Founder says: "We're the only ones doing this"
{
    "type": "question",
    "priority": "medium",
    "content": "Ask: 'How do customers solve this today without you?'",
    "reasoning": "No competition often means no market. Understand the alternative."
}

Founder says: "We'll be at $50M ARR next year"
{
    "type": "flag",
    "priority": "high",
    "content": "They're at $8M now. 6x growth would put them in top 1% of all SaaS.",
    "reasoning": "T2D3 benchmark is 3x, 3x, 2x, 2x, 2x. This implies 6x.",
    "data_point": "Fastest scale-ups: Slack, Zoom did ~4x at this stage"
}
"""

async def analyze_with_claude(
    transcript: str,
    obsidian_context: list[dict] = None,
    granola_context: list[dict] = None
) -> dict:
    """Analyze transcript with Claude and return insights."""
    if not ANTHROPIC_API_KEY:
        return {"type": "error", "content": "No Anthropic API key configured"}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build context from knowledge bases
    context_parts = []

    if obsidian_context:
        context_parts.append("## Relevant Notes from Your Playbook:")
        for note in obsidian_context:
            context_parts.append(f"**{note['file']}**:\n{note['snippet']}\n")

    if granola_context:
        context_parts.append("## Relevant Past Meetings:")
        for meeting in granola_context:
            context_parts.append(f"**{meeting['title']}** ({meeting['date']}):\n{meeting['snippet']}\n")

    context_str = "\n".join(context_parts) if context_parts else ""

    user_message = f"""## Current Meeting Transcript (last few minutes):
{transcript}

{context_str}

Based on what was just said, provide ONE insight, question, or flag. Be specific and actionable."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=JARVIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        # Parse JSON response
        content = response.content[0].text
        try:
            # Try to extract JSON from response
            if "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                insight = json.loads(content[start:end])
                return insight
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return as plain insight
        return {
            "type": "insight",
            "priority": "medium",
            "content": content
        }

    except Exception as e:
        return {"type": "error", "content": str(e)}

# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    """Receive audio from phone, transcribe locally with Whisper, analyze with Claude."""
    await websocket.accept()
    meeting_state.is_recording = True

    try:
        # Initialize local transcriber
        transcriber = get_transcriber(WHISPER_MODEL)
        last_analysis_time = datetime.now()
        audio_buffer = bytearray()

        await websocket.send_json({"status": "connected", "model": WHISPER_MODEL})

        while True:
            try:
                # Receive audio data from phone
                data = await websocket.receive_bytes()

                # Convert WebM to PCM if needed
                if AudioConverter.is_webm(data) or len(audio_buffer) == 0:
                    # Accumulate WebM chunks
                    audio_buffer.extend(data)

                    # Try to convert and transcribe when we have enough data
                    if len(audio_buffer) > 50000:  # ~1 second of WebM audio
                        try:
                            pcm_data = await AudioConverter.webm_to_pcm(bytes(audio_buffer))
                            audio_buffer = bytearray()

                            # Transcribe
                            transcript = await transcriber.process_audio_chunk(pcm_data)

                            if transcript:
                                meeting_state.add_transcript(transcript, is_final=True)

                                # Send transcript to UI
                                for ui_ws in meeting_state.ui_connections:
                                    try:
                                        await ui_ws.send_json({
                                            "type": "transcript",
                                            "text": transcript,
                                            "is_final": True
                                        })
                                    except:
                                        pass

                                # Analyze periodically
                                now = datetime.now()
                                if (now - last_analysis_time).seconds >= 15:
                                    last_analysis_time = now

                                    full_transcript = meeting_state.get_full_transcript()
                                    if full_transcript:
                                        # Search for context
                                        search_terms = transcript.split()[:5]
                                        search_query = " ".join(search_terms)
                                        obsidian_context = search_obsidian(search_query)
                                        granola_context = search_granola(search_query)

                                        # Analyze with Claude
                                        insight = await analyze_with_claude(
                                            full_transcript[-2000:],
                                            obsidian_context,
                                            granola_context
                                        )

                                        if insight.get("type") != "error":
                                            meeting_state.add_insight(insight)

                                            for ui_ws in meeting_state.ui_connections:
                                                try:
                                                    await ui_ws.send_json({
                                                        "type": "insight",
                                                        **insight
                                                    })
                                                except:
                                                    pass
                        except Exception as e:
                            print(f"Transcription error: {e}")
                            audio_buffer = bytearray()
                else:
                    audio_buffer.extend(data)

            except WebSocketDisconnect:
                break

        # Transcribe any remaining audio
        final_transcript = await transcriber.finalize()
        if final_transcript:
            meeting_state.add_transcript(final_transcript, is_final=True)

    except Exception as e:
        print(f"Error in audio websocket: {e}")
        import traceback
        traceback.print_exc()
    finally:
        meeting_state.is_recording = False

@app.websocket("/ws/ui")
async def ui_websocket(websocket: WebSocket):
    """Send insights to laptop UI."""
    await websocket.accept()
    meeting_state.ui_connections.append(websocket)

    # Send current state
    await websocket.send_json({
        "type": "state",
        "is_recording": meeting_state.is_recording,
        "insights": meeting_state.insights[-10:]
    })

    try:
        while True:
            # Keep connection alive, receive any commands
            data = await websocket.receive_json()

            if data.get("command") == "clear":
                meeting_state.insights = []
                meeting_state.transcript = []

    except WebSocketDisconnect:
        meeting_state.ui_connections.remove(websocket)

# =============================================================================
# REST Endpoints
# =============================================================================

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>JARVIS - Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f0f;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .container {
            text-align: center;
            padding: 40px;
            background: #18181b;
            border-radius: 16px;
            max-width: 400px;
            width: 90%;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #06b6d4, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #71717a; margin-bottom: 2rem; }
        input {
            width: 100%;
            padding: 14px;
            border-radius: 8px;
            border: 1px solid #3f3f46;
            background: #27272a;
            color: white;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        button {
            width: 100%;
            padding: 14px;
            border-radius: 8px;
            border: none;
            background: linear-gradient(135deg, #8b5cf6, #6366f1);
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
        }
        .error { color: #f87171; margin-top: 1rem; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>JARVIS</h1>
        <p class="subtitle">Growth Equity Copilot</p>
        <input type="password" id="password" placeholder="Enter password" onkeypress="if(event.key==='Enter')login()">
        <button onclick="login()">Enter</button>
        <p class="error" id="error">Invalid password</p>
    </div>
    <script>
        // Check if already authenticated
        const token = localStorage.getItem('jarvis-token');
        if (token) {
            window.location.href = '/laptop?token=' + token;
        }

        async function login() {
            const password = document.getElementById('password').value;
            const resp = await fetch('/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({password})
            });
            if (resp.ok) {
                const data = await resp.json();
                localStorage.setItem('jarvis-token', data.token);
                window.location.href = '/laptop?token=' + data.token;
            } else {
                document.getElementById('error').style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def root(token: str = Query(None)):
    """Root - show login or redirect to app."""
    if not APP_PASSWORD:
        # No password set, go straight to app
        return HTMLResponse(content=open(Path(__file__).parent.parent / "laptop-ui" / "index.html").read())
    if token and verify_token(token):
        return HTMLResponse(content=open(Path(__file__).parent.parent / "laptop-ui" / "index.html").read())
    return HTMLResponse(content=LOGIN_PAGE)

@app.post("/auth/login")
async def login(request: Request):
    """Login endpoint."""
    data = await request.json()
    password = data.get("password", "")
    if password == APP_PASSWORD:
        token = generate_token()
        valid_tokens.add(token)
        return {"token": token}
    raise HTTPException(status_code=401, detail="Invalid password")

@app.get("/phone")
async def phone_ui(token: str = Query(None)):
    """Serve phone mic capture UI."""
    if APP_PASSWORD and not verify_token(token):
        return HTMLResponse(content=LOGIN_PAGE)
    return HTMLResponse(content=open(Path(__file__).parent.parent / "phone-app" / "index.html").read())

@app.get("/laptop")
async def laptop_ui(token: str = Query(None)):
    """Serve laptop insights UI."""
    if APP_PASSWORD and not verify_token(token):
        return HTMLResponse(content=LOGIN_PAGE)
    return HTMLResponse(content=open(Path(__file__).parent.parent / "laptop-ui" / "index.html").read())

@app.get("/api/status")
async def api_status():
    return {"status": "JARVIS is online", "recording": meeting_state.is_recording}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "transcription": "local-whisper",
        "whisper_model": WHISPER_MODEL,
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "obsidian_configured": bool(OBSIDIAN_VAULT_PATH) and os.path.exists(OBSIDIAN_VAULT_PATH),
        "granola_configured": os.path.exists(GRANOLA_CACHE_PATH)
    }

@app.get("/insights")
async def get_insights():
    return {"insights": meeting_state.insights}

@app.post("/clear")
async def clear_session():
    meeting_state.insights = []
    meeting_state.transcript = []
    return {"status": "cleared"}

# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
