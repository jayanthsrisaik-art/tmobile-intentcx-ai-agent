"""FastAPI service exposing the IntentCX agent."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent import get_agent
from src.router import classify_keywords
from src.tools import get_customer_profile, list_tickets

app = FastAPI(
    title="T-Mobile IntentCX AI Agent",
    description="Intent-routed, tool-grounded AI agent for customer support and sales.",
    version="2.0.0",
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["Why did my bill go up?"])
    customer_id: str = Field("123", examples=["123"])
    session_id: str | None = Field(None, description="Reuse to continue a multi-turn conversation")


class ToolCallOut(BaseModel):
    name: str
    arguments: dict
    result: Any


class ChatResponse(BaseModel):
    session_id: str
    customer_id: str
    agent_type: str
    intent: str
    confidence: float
    routing_method: str
    user_query: str
    response: str
    tool_calls: list[ToolCallOut]
    sentiment: str
    frustration: float
    redactions: list[str]
    escalated: bool
    mode: str
    latency_ms: float


@app.get("/")
def health():
    return {"status": "IntentCX AI Agent running", "mode": get_agent().mode}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest | None = None, query: str | None = Query(None, description="Legacy query param")):
    if req is None:
        if not query:
            raise HTTPException(422, "Provide a JSON body {'query': ...} or a ?query= parameter")
        req = ChatRequest(query=query)
    if not get_customer_profile(req.customer_id):
        raise HTTPException(404, f"Customer {req.customer_id} not found")
    result = get_agent().handle(req.query, customer_id=req.customer_id, session_id=req.session_id)
    data = result.to_dict()
    data["user_query"] = req.query
    return data


@app.post("/route")
def route(req: ChatRequest):
    """Inspect routing only (no LLM call) - useful for debugging intent coverage."""
    return classify_keywords(req.query).to_dict()


@app.get("/customers/{customer_id}")
def customer(customer_id: str):
    profile = get_customer_profile(customer_id)
    if not profile:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return profile


@app.get("/sessions/{session_id}")
def session(session_id: str):
    s = get_agent().sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    return {"session_id": s.session_id, "customer_id": s.customer_id, "created_at": s.created_at,
            "escalated": s.escalated, "intents": s.intents, "messages": s.messages}


@app.delete("/sessions/{session_id}")
def end_session(session_id: str):
    if not get_agent().sessions.delete(session_id):
        raise HTTPException(404, "Session not found")
    return {"deleted": session_id}


@app.get("/tickets")
def tickets(customer_id: str | None = None):
    return {"tickets": list_tickets(customer_id)}


@app.get("/metrics")
def metrics():
    return get_agent().get_metrics()


_CHAT_UI = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IntentCX Agent</title><style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#fafafa;color:#222}
h1{color:#e20074;font-size:1.4rem}#log{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;min-height:320px}
.u{margin:8px 0;text-align:right}.u span{background:#e20074;color:#fff;padding:6px 10px;border-radius:12px;display:inline-block}
.a{margin:8px 0}.a span{background:#eee;padding:6px 10px;border-radius:12px;display:inline-block;white-space:pre-wrap}
.meta{font-size:.75rem;color:#777;margin-top:2px}form{display:flex;gap:8px;margin-top:12px}
input{flex:1;padding:10px;border:1px solid #ccc;border-radius:6px}select,button{padding:10px;border-radius:6px;border:1px solid #ccc}
button{background:#e20074;color:#fff;border:none;cursor:pointer}</style></head><body>
<h1>T-Mobile IntentCX AI Agent</h1>
<div id="log"></div>
<form id="f"><select id="c"><option value="123">John (123)</option><option value="456">Priya (456)</option><option value="789">Marcus (789)</option></select>
<input id="q" placeholder="Ask about your bill, network, plans, upgrades..." autocomplete="off"><button>Send</button></form>
<script>
let sid=null;const log=document.getElementById('log');
function add(cls,html){const d=document.createElement('div');d.className=cls;d.innerHTML=html;log.appendChild(d);d.scrollIntoView();}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
document.getElementById('c').onchange=()=>{sid=null;log.innerHTML='';};
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q').value.trim();if(!q)return;
document.getElementById('q').value='';add('u','<span>'+esc(q)+'</span>');
const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({query:q,customer_id:document.getElementById('c').value,session_id:sid})});
const j=await r.json();if(!r.ok){add('a','<span>Error: '+esc(JSON.stringify(j.detail))+'</span>');return;}sid=j.session_id;
const tools=j.tool_calls.map(t=>t.name).join(', ')||'none';
add('a','<span>'+esc(j.response)+'</span><div class="meta">'+j.agent_type.toUpperCase()+' agent · intent '+j.intent+
' ('+Math.round(j.confidence*100)+'%) · sentiment '+j.sentiment+' · tools: '+tools+(j.escalated?' · ESCALATED':'')+' · '+j.latency_ms+' ms</div>');};
</script></body></html>"""


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return _CHAT_UI
