#!/usr/bin/env python3
"""
JARVIS - Growth Equity Meeting Copilot
Real-time AI assistant for investor meetings.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import httpx
import websockets as ws_client
from dotenv import load_dotenv
import yfinance as yf
from duckduckgo_search import DDGS

# Session storage directory
SESSIONS_DIR = Path(__file__).parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

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
        self.session_id = None
        self.session_start = None

    def start_session(self):
        """Start a new recording session."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.session_start = datetime.now().isoformat()
        self.transcript = []
        self.insights = []
        print(f"Session started: {self.session_id}", flush=True)

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

    def save_session(self):
        """Save session to disk."""
        if not self.session_id:
            return

        session_data = {
            "session_id": self.session_id,
            "start_time": self.session_start,
            "end_time": datetime.now().isoformat(),
            "transcript": self.transcript,
            "full_transcript": self.get_full_transcript(),
            "insights": self.insights
        }

        session_file = SESSIONS_DIR / f"{self.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        print(f"Session saved: {session_file}", flush=True)
        return session_file

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

JARVIS_SYSTEM_PROMPT = """You are JARVIS - elite investor co-pilot for a growth equity investor in live meetings.

## WHEN TO RESPOND
Only respond when there's something valuable to add:
- A company, person, or fund is mentioned → give context
- A metric/benchmark is discussed → provide market context
- A direct question is asked → answer it
- A claim is made that needs verification → provide data

If it's just small talk, filler, or nothing actionable → return {"type": "skip"}

## STYLE: McKinsey top-down
- **Bold headline** (the answer in 1 line)
- **2-3 bullets** (supporting facts, numbers, benchmarks)
- **Source**

## Output Format
{"type": "insight", "content": "Your structured insight here", "source": "link"}
{"type": "skip"} ← use this when nothing valuable to add

## Examples

"Stripe"
{"type": "insight", "content": "**Stripe: $50B valuation, dominant in online payments**\n• Down from $95B peak (2021) - 47% markdown\n• Takes 2.9% + $0.30/txn, powers 14% of US e-commerce\n• Comps: Adyen $40B, Square $45B", "source": "stripe.com/newsroom"}

"150% NRR"
{"type": "insight", "content": "**150% NRR is elite - top 5% of SaaS**\n• Benchmarks: Snowflake 127%, Datadog 115%, MongoDB 120%\n• Rule of thumb: >130% = hit plan with zero new logos\n• Expansion levers: seats, usage, or upsells", "source": "bvp.com/atlas"}

"so anyway we were thinking about..."
{"type": "skip"}

"let me introduce the team"
{"type": "skip"}

Be selective. Quality over quantity."""

# =============================================================================
# Live Data Functions
# =============================================================================

def search_web(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No results found."
            output = []
            for r in results:
                output.append(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}")
            return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"

def get_stock_data(ticker: str) -> str:
    """Get stock/company data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract key metrics
        data = {
            "name": info.get("longName", ticker),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "ev": info.get("enterpriseValue"),
            "revenue": info.get("totalRevenue"),
            "ebitda": info.get("ebitda"),
            "gross_profit": info.get("grossProfits"),
            "pe_ratio": info.get("trailingPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }

        # Format output
        lines = [f"**{data['name']} ({ticker.upper()})**"]
        if data['price']: lines.append(f"• Price: ${data['price']:,.2f}")
        if data['market_cap']: lines.append(f"• Market Cap: ${data['market_cap']/1e9:,.1f}B")
        if data['ev']: lines.append(f"• Enterprise Value: ${data['ev']/1e9:,.1f}B")
        if data['revenue']: lines.append(f"• Revenue (TTM): ${data['revenue']/1e9:,.1f}B")
        if data['ebitda']: lines.append(f"• EBITDA: ${data['ebitda']/1e9:,.1f}B")
        if data['gross_profit']: lines.append(f"• Gross Profit: ${data['gross_profit']/1e9:,.1f}B")
        if data['ev'] and data['revenue']: lines.append(f"• EV/Revenue: {data['ev']/data['revenue']:,.1f}x")
        if data['ev'] and data['ebitda']: lines.append(f"• EV/EBITDA: {data['ev']/data['ebitda']:,.1f}x")
        if data['ev'] and data['gross_profit']: lines.append(f"• EV/GP: {data['ev']/data['gross_profit']:,.1f}x")
        if data['pe_ratio']: lines.append(f"• P/E: {data['pe_ratio']:,.1f}x")

        return "\n".join(lines) + f"\nSource: finance.yahoo.com/quote/{ticker.upper()}"
    except Exception as e:
        return f"Could not fetch data for {ticker}: {str(e)}"

def get_live_context(query: str) -> str:
    """Get live context - tries stock data first, then web search."""
    # Common company -> ticker mappings
    ticker_map = {
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
        "amazon": "AMZN", "meta": "META", "facebook": "META", "nvidia": "NVDA",
        "tesla": "TSLA", "netflix": "NFLX", "salesforce": "CRM", "adobe": "ADBE",
        "snowflake": "SNOW", "datadog": "DDOG", "mongodb": "MDB", "crowdstrike": "CRWD",
        "palantir": "PLTR", "stripe": "STRIPE", "shopify": "SHOP", "zoom": "ZM",
        "slack": "WORK", "twilio": "TWLO", "okta": "OKTA", "cloudflare": "NET",
        "confluent": "CFLT", "hashicorp": "HCP", "gitlab": "GTLB", "jfrog": "FROG",
    }

    # Check if it's asking about a public company
    query_lower = query.lower()
    for company, ticker in ticker_map.items():
        if company in query_lower:
            stock_data = get_stock_data(ticker)
            if "Could not fetch" not in stock_data:
                return stock_data

    # Otherwise do a web search
    return search_web(query)

def get_feedback_learnings() -> str:
    """Load recent feedback to learn from."""
    learnings = []
    try:
        for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True)[:10]:  # Last 10 sessions
            with open(f) as file:
                data = json.load(file)
                feedback_list = data.get("feedback", [])
                for fb in feedback_list:
                    if fb.get("missed_insights"):
                        learnings.append(f"- User wanted: {fb['missed_insights']}")
                    if fb.get("what_would_help"):
                        learnings.append(f"- Would help: {fb['what_would_help']}")
    except:
        pass
    return "\n".join(learnings[-10:]) if learnings else ""  # Last 10 feedback items

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

    # Add learnings from past feedback
    feedback_learnings = get_feedback_learnings()
    if feedback_learnings:
        context_parts.append(f"## LEARN FROM PAST FEEDBACK (what user wanted):\n{feedback_learnings}\n")

    # Get live data based on transcript (companies, metrics mentioned)
    # Extract potential search queries from transcript
    try:
        live_data = get_live_context(transcript)
        if live_data and "No results" not in live_data and "error" not in live_data.lower():
            context_parts.append(f"## LIVE DATA (real-time, verified):\n{live_data}\n")
    except Exception as e:
        print(f"Live data fetch error: {e}", flush=True)

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

Based on what was just said, provide ONE insight using the LIVE DATA if available. Include real numbers and hyperlink sources."""

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
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
    meeting_state.start_session()

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
        meeting_state.save_session()
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

@app.get("/sessions")
async def list_sessions():
    """List all saved sessions."""
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            with open(f) as file:
                data = json.load(file)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "transcript_length": len(data.get("full_transcript", "")),
                    "insight_count": len(data.get("insights", []))
                })
        except:
            pass
    return {"sessions": sessions}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    with open(session_file) as f:
        return json.load(f)

@app.post("/sessions/{session_id}/feedback")
async def add_feedback(session_id: str, request: Request):
    """Add feedback to a session - what insights were missed, what could be better."""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    feedback_data = await request.json()

    with open(session_file) as f:
        session = json.load(f)

    if "feedback" not in session:
        session["feedback"] = []

    session["feedback"].append({
        "timestamp": datetime.now().isoformat(),
        "missed_insights": feedback_data.get("missed_insights", ""),
        "what_would_help": feedback_data.get("what_would_help", ""),
        "rating": feedback_data.get("rating", 0),
        "notes": feedback_data.get("notes", "")
    })

    with open(session_file, "w") as f:
        json.dump(session, f, indent=2)

    return {"status": "feedback saved"}

@app.get("/feedback/all")
async def get_all_feedback():
    """Get all feedback across sessions for training/improvement."""
    all_feedback = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            with open(f) as file:
                data = json.load(file)
                if data.get("feedback"):
                    all_feedback.append({
                        "session_id": data.get("session_id"),
                        "transcript": data.get("full_transcript", "")[:500],
                        "insights_given": data.get("insights", []),
                        "feedback": data.get("feedback", [])
                    })
        except:
            pass
    return {"feedback": all_feedback}

# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
