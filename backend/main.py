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
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "")
GRANOLA_CACHE_PATH = os.path.expanduser("~/Library/Application Support/Granola/cache-v3.json")

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
    """Receive audio from phone, transcribe, analyze."""
    await websocket.accept()
    meeting_state.is_recording = True

    if not DEEPGRAM_API_KEY:
        await websocket.send_json({"error": "No Deepgram API key configured"})
        await websocket.close()
        return

    try:
        # Initialize Deepgram
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        dg_connection = deepgram.listen.live.v("1")

        transcript_buffer = []
        last_analysis_time = datetime.now()

        async def on_message(self, result, **kwargs):
            nonlocal last_analysis_time

            transcript = result.channel.alternatives[0].transcript
            if not transcript:
                return

            is_final = result.is_final
            meeting_state.add_transcript(transcript, is_final)

            # Send transcript to UI
            for ui_ws in meeting_state.ui_connections:
                try:
                    await ui_ws.send_json({
                        "type": "transcript",
                        "text": transcript,
                        "is_final": is_final
                    })
                except:
                    pass

            # Analyze every 10 seconds or on final transcript
            now = datetime.now()
            if is_final and (now - last_analysis_time).seconds >= 10:
                last_analysis_time = now

                # Get context from knowledge bases
                full_transcript = meeting_state.get_full_transcript()

                # Search for relevant context (use key terms from recent transcript)
                search_terms = transcript.split()[:5]
                search_query = " ".join(search_terms)

                obsidian_context = search_obsidian(search_query)
                granola_context = search_granola(search_query)

                # Analyze with Claude
                insight = await analyze_with_claude(
                    full_transcript[-2000:],  # Last 2000 chars
                    obsidian_context,
                    granola_context
                )

                if insight.get("type") != "error":
                    meeting_state.add_insight(insight)

                    # Send to UI
                    for ui_ws in meeting_state.ui_connections:
                        try:
                            await ui_ws.send_json({
                                "type": "insight",
                                **insight
                            })
                        except:
                            pass

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
            interim_results=True,
        )

        await dg_connection.start(options)

        # Receive audio from phone and send to Deepgram
        while True:
            try:
                data = await websocket.receive_bytes()
                await dg_connection.send(data)
            except WebSocketDisconnect:
                break

        await dg_connection.finish()

    except Exception as e:
        print(f"Error in audio websocket: {e}")
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

@app.get("/")
async def root():
    """Redirect to laptop UI."""
    return HTMLResponse(content=open(Path(__file__).parent.parent / "laptop-ui" / "index.html").read())

@app.get("/phone")
async def phone_ui():
    """Serve phone mic capture UI."""
    return HTMLResponse(content=open(Path(__file__).parent.parent / "phone-app" / "index.html").read())

@app.get("/laptop")
async def laptop_ui():
    """Serve laptop insights UI."""
    return HTMLResponse(content=open(Path(__file__).parent.parent / "laptop-ui" / "index.html").read())

@app.get("/api/status")
async def api_status():
    return {"status": "JARVIS is online", "recording": meeting_state.is_recording}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
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
