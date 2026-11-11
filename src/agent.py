"""
IntentCX agent orchestrator.

Pipeline per customer turn:
    guardrails (PII redaction + frustration) -> intent routing -> specialist agent
    -> OpenAI tool-calling loop (grounded in backend APIs) -> response + trace

When no OPENAI_API_KEY is configured (or INTENTCX_OFFLINE=1), a deterministic
rule-based responder calls the same tools, so the demo, API and tests all work offline.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.guardrails import apply_guardrails
from src.memory import SessionStore
from src.prompts import AGENT_TOOLS, PROMPTS
from src.router import RoutingDecision, classify
from src.tools import execute_tool, get_customer_profile, schemas_for


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: Any


@dataclass
class AgentResponse:
    session_id: str
    customer_id: str
    agent_type: str
    intent: str
    confidence: float
    routing_method: str
    response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    sentiment: str = "neutral"
    frustration: float = 0.0
    redactions: list[str] = field(default_factory=list)
    escalated: bool = False
    mode: str = "llm"
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class IntentCXAgent:
    def __init__(self, settings: Settings | None = None, client=None, sessions: SessionStore | None = None):
        self.settings = settings or get_settings()
        self.sessions = sessions or SessionStore(self.settings.max_history_turns)
        self.client = client
        if self.client is None and self.settings.llm_enabled:
            from openai import OpenAI

            self.client = OpenAI()
        self.metrics: dict[str, Any] = {
            "requests": 0, "escalations": 0, "tool_calls": 0, "by_agent": {}, "by_intent": {},
            "total_latency_ms": 0.0,
        }

    @property
    def mode(self) -> str:
        return "llm" if self.client is not None else "offline"

    # ------------------------------------------------------------------ public
    def handle(self, query: str, customer_id: str = "123", session_id: str | None = None) -> AgentResponse:
        start = time.perf_counter()
        session = self.sessions.get_or_create(session_id, customer_id)
        customer_id = session.customer_id

        with self.sessions.lock(session.session_id):
            guard = apply_guardrails(query)
            routing = classify(
                guard.text,
                client=self.client if self.settings.llm_routing else None,
                model=self.settings.model,
            )
            # Proactive escalation on high frustration.
            if guard.should_escalate and routing.agent != "escalation":
                routing = RoutingDecision(
                    "escalation", "escalation", max(routing.confidence, guard.frustration),
                    method=f"{routing.method}+sentiment", scores=routing.scores, matched=routing.matched,
                )

            tool_calls: list[ToolCall] = []
            if self.client is not None:
                text = self._run_llm(guard.text, routing.agent, customer_id, session.messages, tool_calls)
            else:
                text = self._run_offline(guard.text, routing, customer_id, tool_calls)

            escalated = any(tc.name == "escalate_to_human" for tc in tool_calls)
            session.escalated = session.escalated or escalated
            session.intents.append(routing.intent)
            self.sessions.append(session, "user", guard.text)
            self.sessions.append(session, "assistant", text)

        latency = round((time.perf_counter() - start) * 1000, 1)
        self._record(routing, tool_calls, escalated, latency)
        return AgentResponse(
            session_id=session.session_id, customer_id=customer_id, agent_type=routing.agent,
            intent=routing.intent, confidence=routing.confidence, routing_method=routing.method,
            response=text, tool_calls=tool_calls, sentiment=guard.sentiment, frustration=guard.frustration,
            redactions=guard.redactions, escalated=escalated, mode=self.mode, latency_ms=latency,
        )

    def get_metrics(self) -> dict:
        m = dict(self.metrics)
        m["avg_latency_ms"] = round(m["total_latency_ms"] / m["requests"], 1) if m["requests"] else 0.0
        m["escalation_rate"] = round(m["escalations"] / m["requests"], 3) if m["requests"] else 0.0
        m["active_sessions"] = len(self.sessions)
        m["mode"] = self.mode
        return m

    # --------------------------------------------------------------- internals
    def _record(self, routing: RoutingDecision, tool_calls: list[ToolCall], escalated: bool, latency: float):
        m = self.metrics
        m["requests"] += 1
        m["tool_calls"] += len(tool_calls)
        m["escalations"] += int(escalated)
        m["total_latency_ms"] += latency
        m["by_agent"][routing.agent] = m["by_agent"].get(routing.agent, 0) + 1
        m["by_intent"][routing.intent] = m["by_intent"].get(routing.intent, 0) + 1

    def _run_llm(self, query: str, agent_type: str, customer_id: str, history: list[dict],
                 tool_calls: list[ToolCall]) -> str:
        profile = get_customer_profile(customer_id)
        context = (
            f"Customer ID: {customer_id}\n"
            f"Customer profile (summary): {json.dumps(profile) if profile else 'not found'}"
        )
        messages: list[dict] = [
            {"role": "system", "content": PROMPTS[agent_type]},
            {"role": "system", "content": context},
            *history,
            {"role": "user", "content": query},
        ]
        tools = schemas_for(AGENT_TOOLS[agent_type])

        for _ in range(self.settings.max_tool_rounds):
            resp = self.client.chat.completions.create(
                model=self.settings.model, messages=messages, tools=tools, temperature=0.3,
            )
            msg = resp.choices[0].message
            if not getattr(msg, "tool_calls", None):
                return msg.content or ""

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                if name not in AGENT_TOOLS[agent_type]:
                    result: Any = {"error": f"Tool {name} not permitted for {agent_type} agent"}
                    args: dict = {}
                else:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = execute_tool(name, args)
                tool_calls.append(ToolCall(name, args, result))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})

        return ("I'm sorry, I wasn't able to complete that request. "
                "I can connect you with a T-Mobile expert if you'd like.")

    # Deterministic responder used when no LLM is available.
    def _run_offline(self, query: str, routing: RoutingDecision, customer_id: str,
                     tool_calls: list[ToolCall]) -> str:
        def call(name: str, **kwargs) -> Any:
            result = execute_tool(name, kwargs)
            tool_calls.append(ToolCall(name, kwargs, result))
            return result

        profile = call("get_customer_profile", customer_id=customer_id)
        name = profile.get("name", "there") if profile else "there"
        q = query.lower()

        if routing.agent == "escalation":
            res = call("escalate_to_human", customer_id=customer_id, reason=f"Customer request: {query[:120]}")
            return (f"I'm really sorry for the trouble, {name}. I've connected you with a T-Mobile expert "
                    f"(ticket {res['ticket_id']}); expected wait is about {res['estimated_wait_minutes']} minutes.")

        if routing.intent == "billing":
            bills = call("get_billing_history", customer_id=customer_id)
            change = bills.get("latest_change") if isinstance(bills, dict) else None
            if change and change["delta"]:
                direction = "up" if change["delta"] > 0 else "down"
                return (f"Hi {name}, your bill went {direction} by ${abs(change['delta']):.2f} from "
                        f"{change['from_month']} to {change['to_month']}. Reason: {change['explanation']}. "
                        "Would you like me to open a billing review ticket?")
            if bills.get("bills"):
                latest = bills["bills"][-1]
                return (f"Hi {name}, your latest bill ({latest['month']}) is ${latest['amount']:.2f} "
                        f"and hasn't changed recently. Anything specific you'd like me to look into?")
            return "I couldn't find billing records for this account. Let me open a ticket for a specialist."

        if routing.intent == "technical_support":
            zip_code = profile.get("zip_code") if profile else None
            if zip_code:
                status = call("check_network_status", zip_code=zip_code)
                if status["status"] != "normal":
                    return (f"Hi {name}, there's a known issue in your area ({zip_code}): {status['issue']}. "
                            f"Expected fix: {status['eta']}. Wi-Fi Calling is a good workaround until then.")
            ticket = call("create_support_ticket", customer_id=customer_id, category="network",
                          summary=query[:200], priority="normal")
            return (f"Hi {name}, I don't see any outages near you. Please try restarting your device and "
                    f"toggling airplane mode. I've also opened ticket {ticket['ticket_id']} so our team can follow up.")

        if routing.intent == "sales_device":
            elig = call("check_upgrade_eligibility", customer_id=customer_id)
            brand = next((b for k, b in (("iphone", "Apple"), ("apple", "Apple"), ("galaxy", "Samsung"),
                                         ("samsung", "Samsung"), ("pixel", "Google"), ("google", "Google"))
                          if k in q), None)
            devices = (call("list_devices", brand=brand) if brand else call("list_devices"))["devices"][:3]
            names = ", ".join(f"{d['name']} (${d['price']})" for d in devices)
            if elig.get("eligible"):
                bonus = " As a long-time customer you may also qualify for loyalty offers." if elig.get("loyalty_bonus") else ""
                return (f"Great news, {name} — you're eligible to upgrade from your {elig['current_device']}! "
                        f"Popular picks: {names}. Trade-in credits can lower the cost.{bonus}")
            return (f"{name}, you're not upgrade-eligible just yet ({elig.get('notes', '')}). "
                    f"When you are, top picks include {names}.")

        if routing.intent == "sales_plan":
            rec = call(
                "recommend_plan",
                lines=int(profile.get("lines", 1)) if profile else 1,
                data_usage="high" if any(w in q for w in ("unlimited", "stream", "hotspot", "a lot")) else "medium",
                travels_internationally=any(w in q for w in ("international", "travel", "abroad")),
                wants_streaming=any(w in q for w in ("netflix", "stream", "tv")),
            )
            top = rec["recommendations"][0]
            current = profile.get("current_plan") if profile else None
            suffix = " — that's the plan you already have, so you're well covered!" if top["plan"] == current else "."
            return (f"Based on your needs, {name}, I'd recommend {top['plan']} "
                    f"(about ${top['estimated_monthly_total']}/mo, great for {top['why']}){suffix}")

        if routing.intent == "account":
            ticket = call("create_support_ticket", customer_id=customer_id, category="account",
                          summary=query[:200], priority="normal")
            return (f"Hi {name}, for account changes we need to verify your identity securely. "
                    f"I've opened ticket {ticket['ticket_id']}; you can also manage this in the T-Mobile app.")

        return (f"Hi {name}! I can help with billing, network issues, plans, and device upgrades. "
                "What can I do for you today?")


# ---------------------------------------------------------------------------
# Backward-compatible functional API
# ---------------------------------------------------------------------------
_default_agent: IntentCXAgent | None = None


def get_agent() -> IntentCXAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = IntentCXAgent()
    return _default_agent


def run_agent(query: str, agent_type: str | None = None, customer_id: str = "123") -> str:
    """Single-turn helper kept for compatibility with the original API."""
    return get_agent().handle(query, customer_id=customer_id).response
