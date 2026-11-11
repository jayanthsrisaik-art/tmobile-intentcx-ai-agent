# T-Mobile IntentCX AI Agent (Customer Support + Sales)

An OpenAI-powered, multi-agent customer experience system inspired by T-Mobile's IntentCX initiative.

It shows how LLM agents can improve support and sales workflows with **intent routing**, **tool-grounded
answers** (function calling against backend APIs), **guardrails**, **multi-turn memory** and
**human escalation**. Everything also runs **offline without an API key**, so you can demo and test it anywhere.

---

## Features

| Capability | What it does |
|---|---|
| **Hybrid intent routing** | Weighted-keyword classifier over 7 intents (billing, technical support, account, plan sales, device sales, escalation, general) with confidence scores; falls back to LLM classification when confidence is low. |
| **Specialist agents** | Support, Sales and Escalation agents, each with its own prompt and a **least-privilege tool set**. |
| **Tool-grounded answers** | OpenAI function calling against 9 simulated backend APIs: customer profile, billing history, network status, plan catalog, plan recommender, devices, upgrade eligibility, support tickets, human hand-off. |
| **Guardrails** | PII redaction (card numbers, SSNs, emails) before anything reaches the LLM or memory; frustration detection triggers proactive escalation. |
| **Multi-turn memory** | Per-session conversation history (bounded), shared across turns via `session_id`. |
| **Observability** | Every response includes the routing decision, tools called with results, sentiment and latency; `/metrics` reports volume, escalation rate and intent mix. |
| **Offline mode** | Deterministic responder uses the same tools when `OPENAI_API_KEY` is missing — great for demos and CI. |
| **Production-ready shape** | FastAPI service with typed schemas, built-in web chat UI, Dockerfile, GitHub Actions CI, 39 tests. |

---

## Architecture

```
Customer message
      │
      ▼
 Guardrails ── PII redaction, sentiment / frustration score
      │
      ▼
 Intent Router ── keyword scoring ──(low confidence)──▶ LLM classifier
      │                     (high frustration ──▶ Escalation agent)
      ▼
 Specialist Agent (Support │ Sales │ Escalation)
      │   system prompt + session history + customer context
      ▼
 OpenAI tool-calling loop ◀──▶ Backend tools (scoped per agent)
      │
      ▼
 Grounded response + trace (intent, confidence, tool calls, sentiment, latency)
```

| File | Role |
|---|---|
| `src/router.py` | Intent classification and agent mapping |
| `src/agent.py` | Orchestrator, LLM tool-calling loop, offline responder, metrics |
| `src/tools.py` | Simulated backend APIs and their OpenAI schemas |
| `src/prompts.py` | Agent prompts and per-agent tool permissions |
| `src/guardrails.py` | PII redaction and sentiment scoring |
| `src/memory.py` | Session store |
| `src/config.py` | Environment-driven settings |
| `app.py` | FastAPI service and web chat UI |
| `data/` | Sample customers, plan/device catalog, network status (illustrative only) |

---

## ▶️ Run the demo

```bash
pip install -r requirements.txt
```

Optionally set your key (otherwise it runs in offline mode):

```bash
cp .env.example .env   # then add OPENAI_API_KEY
```

Scripted showcase (billing → network → upgrade → plan → escalation):

```bash
python demo.py --scripted
```

Interactive, as a different customer (`123` John, `456` Priya — has a local outage, `789` Marcus):

```bash
python demo.py --customer 456
```

Example output:

```
Customer: Why did my bill go up this month?

[Routing -> SUPPORT AGENT | intent=billing conf=80% via keyword | sentiment=neutral]
[Tools: get_customer_profile, get_billing_history]
Agent Response:
 Hi John, your bill went up by $25.00 from 2026-09 to 2026-10. Reason: Promo credit ($15) expired;
 international roaming pass ($10). Would you like me to open a billing review ticket?
```

---

## 🌐 Run the API

```bash
uvicorn app:app --reload
```

- Web chat UI: http://localhost:8000/ui
- Interactive docs: http://localhost:8000/docs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check and mode (`llm` / `offline`) |
| POST | `/chat` | Chat turn. Body: `{"query", "customer_id", "session_id"}` (legacy `?query=` still works) |
| POST | `/route` | Routing decision only (debug intent coverage) |
| GET | `/customers/{id}` | Customer profile |
| GET / DELETE | `/sessions/{id}` | Conversation history / end session |
| GET | `/tickets?customer_id=` | Tickets and escalations created by the agents |
| GET | `/metrics` | Requests, escalation rate, intent mix, avg latency |

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "Why did my bill go up?", "customer_id": "123"}'
```

### Docker

```bash
docker build -t intentcx-agent .
```

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY intentcx-agent
```

---

## ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | – | Enables LLM mode |
| `INTENTCX_MODEL` | `gpt-4o-mini` | Chat model |
| `INTENTCX_LLM_ROUTING` | `true` | LLM fallback for low-confidence routing |
| `INTENTCX_OFFLINE` | `false` | Force offline mode |
| `INTENTCX_MAX_TOOL_ROUNDS` | `5` | Max tool-calling iterations per turn |
| `INTENTCX_MAX_HISTORY` | `10` | Turns of history kept per session |
| `INTENTCX_DATA_DIR` | `./data` | Location of sample data |

---

## ✅ Tests

```bash
pip install -r requirements-dev.txt
```

```bash
pytest -q
```

Tests cover routing, guardrails, tools, the offline agent, the LLM tool-calling loop (with a scripted fake
OpenAI client — including tool-permission enforcement) and all API endpoints. They never call the network.

---

## Roadmap

- Swap JSON-backed tools for real billing / CRM / network APIs
- Persistent sessions and tickets (Redis / Postgres)
- RAG over support knowledge-base articles
- Streaming responses and voice channel (speech-to-text / text-to-speech)
- Evaluation harness for routing accuracy and answer groundedness

> Sample data, prices and plan details are illustrative only and not real T-Mobile offers.
