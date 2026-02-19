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
import websockets as ws_client
from dotenv import load_dotenv

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
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "")
GRANOLA_CACHE_PATH = os.path.expanduser("~/Library/Application Support/Granola/cache-v3.json")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")  # Set this to require password

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

JARVIS_SYSTEM_PROMPT = """You are JARVIS, a real-time context assistant. When you hear something mentioned in the conversation, IMMEDIATELY provide a quick explainer.

## Your Job
Give instant context on ANYTHING mentioned:
- Books → What's the book about, key themes
- People → Who they are, what they're known for
- Companies → What they do, key facts
- Concepts → Quick explanation
- Places → What's notable about them
- Historical events → Quick summary
- Movies/Shows → Plot, why it's relevant

## Output Format
{"type": "insight", "content": "Quick 1-2 sentence explainer", "source": "optional"}

## Examples

Mentioned: "Fountainhead"
{"type": "insight", "content": "Ayn Rand novel (1943) about architect Howard Roark who refuses to compromise his vision. Themes: individualism vs collectivism, integrity, creative independence.", "source": null}

Mentioned: "Peter Thiel"
{"type": "insight", "content": "PayPal co-founder, first Facebook investor ($500K for 10%). Founders Fund GP. Known for contrarian thinking, wrote 'Zero to One'.", "source": null}

Mentioned: "ARR"
{"type": "insight", "content": "Annual Recurring Revenue - the yearly value of subscription contracts. Key SaaS metric. $10M ARR = roughly $800K MRR.", "source": null}

Mentioned: "Series B"
{"type": "insight", "content": "Typically $15-50M raise at $50-200M valuation. Company usually has product-market fit, $2-10M ARR, looking to scale.", "source": null}

Mentioned: "YC"
{"type": "insight", "content": "Y Combinator - top startup accelerator. $500K for 7% equity. Notable alumni: Airbnb, Stripe, Dropbox, DoorDash.", "source": null}

Mentioned: "The Hard Thing About Hard Things"
{"type": "insight", "content": "Ben Horowitz book on the struggles of running a startup. War stories from Opsware. Key lesson: no formula for hard decisions.", "source": null}

ALWAYS provide context. If you hear a name, book, company, or concept - explain it briefly."""

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
    """Receive audio from phone, stream to Deepgram, analyze with Claude."""
    await websocket.accept()
    meeting_state.is_recording = True

    if not DEEPGRAM_API_KEY:
        await websocket.send_json({"error": "No Deepgram API key configured"})
        await websocket.close()
        return

    try:
        # Connect to Deepgram streaming API
        deepgram_url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&model=nova-2&punctuate=true&interim_results=true"
        import sys
        print(f"Connecting to Deepgram with key: {DEEPGRAM_API_KEY[:10]}...", flush=True)

        try:
            deepgram_ws = await ws_client.connect(
                deepgram_url,
                additional_headers=[("Authorization", f"Token {DEEPGRAM_API_KEY}")]
            )
            print("Deepgram connected!")
        except Exception as dg_err:
            print(f"Deepgram connection failed: {dg_err}")
            await websocket.send_json({"error": f"Deepgram connection failed: {str(dg_err)}"})
            return

        await websocket.send_json({"status": "connected", "transcription": "deepgram"})
        print("Sent status to client")

        full_transcript = ""
        last_analysis_time = datetime.now()

        async def receive_from_deepgram():
            """Receive transcripts from Deepgram and forward to UI."""
            nonlocal full_transcript, last_analysis_time

            try:
                async for message in deepgram_ws:
                    data = json.loads(message)
                    print(f"Deepgram msg type: {data.get('type')}", flush=True)

                    if data.get("type") == "Results":
                        transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        is_final = data.get("is_final", False)
                        print(f"Transcript: '{transcript}' (final={is_final})", flush=True)

                        if transcript:
                            # Send to UI
                            for ui_ws in meeting_state.ui_connections:
                                try:
                                    await ui_ws.send_json({
                                        "type": "transcript",
                                        "text": transcript,
                                        "is_final": is_final
                                    })
                                except:
                                    pass

                            if is_final:
                                full_transcript += " " + transcript
                                meeting_state.add_transcript(transcript, is_final=True)

                                # Analyze with Claude every 10 seconds if we have content
                                now = datetime.now()
                                print(f"Transcript length: {len(full_transcript)}, seconds since last: {(now - last_analysis_time).seconds}", flush=True)
                                if (now - last_analysis_time).seconds >= 10 and len(full_transcript) > 50:
                                    last_analysis_time = now
                                    print("Calling Claude for analysis...", flush=True)

                                    # Run Claude analysis
                                    insight = await analyze_with_claude(full_transcript[-2000:])
                                    print(f"Claude response: {insight}", flush=True)

                                    if insight.get("type") not in ["error", "skip"]:
                                        meeting_state.add_insight(insight)

                                        for ui_ws in meeting_state.ui_connections:
                                            try:
                                                await ui_ws.send_json(insight)
                                            except:
                                                pass
            except Exception as e:
                print(f"Deepgram receive error: {e}")

        async def forward_audio_to_deepgram():
            """Forward audio from client to Deepgram."""
            try:
                while True:
                    data = await websocket.receive_bytes()
                    await deepgram_ws.send(data)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"Audio forward error: {e}")

        # Run both tasks concurrently
        await asyncio.gather(
            receive_from_deepgram(),
            forward_audio_to_deepgram()
        )

    except Exception as e:
        print(f"Error in audio websocket: {e}")
        import traceback
        traceback.print_exc()
    finally:
        meeting_state.is_recording = False
        try:
            await deepgram_ws.close()
        except:
            pass

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
        "transcription": "deepgram",
        "deepgram_configured": bool(DEEPGRAM_API_KEY),
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
