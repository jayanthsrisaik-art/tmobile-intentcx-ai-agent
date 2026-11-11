"""
Agent tools: simulated T-Mobile backend APIs.

Each tool is a plain Python function plus an OpenAI function-calling schema, so the
LLM can decide which APIs to call to ground its answer. Swap the JSON-backed
implementations for real service clients in production.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from src.config import get_settings


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def _data(name: str) -> dict:
    return _load_json(str(get_settings().data_dir / name))


def _customers() -> dict:
    return _data("sample_customer_data.json")


def _catalog() -> dict:
    return _data("catalog.json")


# In-memory ticket store (replace with a CRM / ticketing integration).
_TICKETS: dict[str, dict] = {}
_TICKETS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_customer_profile(customer_id: str) -> dict:
    """Return the customer's account profile (without full billing history)."""
    profile = _customers().get(str(customer_id))
    if not profile:
        return {}
    return {k: v for k, v in profile.items() if k != "billing_history"}


def get_billing_history(customer_id: str, months: int = 3) -> dict:
    """Return recent bills and explain month-over-month changes."""
    profile = _customers().get(str(customer_id))
    if not profile:
        return {"error": f"Customer {customer_id} not found"}
    history = profile.get("billing_history", [])[-max(1, int(months)):]
    change = None
    if len(history) >= 2:
        prev, curr = history[-2], history[-1]
        delta = round(curr["amount"] - prev["amount"], 2)
        change = {
            "from_month": prev["month"],
            "to_month": curr["month"],
            "delta": delta,
            # A decrease is explained by what the previous bill had that this one doesn't.
            "explanation": curr.get("notes") if delta >= 0 else f"Previous bill included: {prev.get('notes')}",
        }
    return {"customer_id": str(customer_id), "bills": history, "latest_change": change}


def check_network_status(zip_code: str) -> dict:
    """Check for known network outages / degradations in a ZIP code."""
    status = _data("network_status.json").get(str(zip_code))
    if not status:
        return {"zip_code": str(zip_code), "status": "normal", "issue": None, "eta": None}
    return {"zip_code": str(zip_code), **status}


def list_plans() -> dict:
    """Return the available rate plans."""
    return {"plans": _catalog().get("plans", [])}


def recommend_plan(lines: int = 1, data_usage: str = "medium", travels_internationally: bool = False,
                   wants_streaming: bool = False, budget_per_line: float | None = None) -> dict:
    """Score plans against customer needs and return a ranked recommendation."""
    plans = _catalog().get("plans", [])
    usage_rank = {"low": 0, "medium": 1, "high": 2}.get(str(data_usage).lower(), 1)
    scored = []
    for plan in plans:
        score = 0.0
        tier = {"Essentials": 0, "Magenta": 1, "Magenta Max": 2}.get(plan["name"], 1)
        score -= abs(tier - usage_rank) * 2
        if travels_internationally and tier == 2:
            score += 2
        if wants_streaming and plan.get("streaming_perks"):
            score += len(plan["streaming_perks"])
        if budget_per_line is not None and plan["price_per_line"] > float(budget_per_line):
            score -= 5
        scored.append({
            "plan": plan["name"],
            "score": round(score, 2),
            "estimated_monthly_total": plan["price_per_line"] * max(1, int(lines)),
            "why": plan.get("best_for"),
        })
    scored.sort(key=lambda s: s["score"], reverse=True)
    return {"recommendations": scored, "top_pick": scored[0]["plan"] if scored else None}


def list_devices(brand: str | None = None, max_price: float | None = None) -> dict:
    """Return devices, optionally filtered by brand and max price."""
    devices = _catalog().get("devices", [])
    if brand:
        devices = [d for d in devices if d["brand"].lower() == brand.lower()]
    if max_price is not None:
        devices = [d for d in devices if d["price"] <= float(max_price)]
    return {"devices": devices}


def check_upgrade_eligibility(customer_id: str) -> dict:
    """Determine whether a customer can upgrade their device today."""
    profile = _customers().get(str(customer_id))
    if not profile:
        return {"error": f"Customer {customer_id} not found"}
    eligible = bool(profile.get("eligible_upgrade"))
    loyalty = profile.get("tenure_months", 0) >= 24
    return {
        "customer_id": str(customer_id),
        "eligible": eligible,
        "current_device": profile.get("device"),
        "loyalty_bonus": loyalty,
        "notes": (
            "Eligible for upgrade with trade-in credit."
            + (" Long-tenure customer: mention loyalty offers." if loyalty else "")
        ) if eligible else "Not yet eligible; device balance must be paid down first.",
    }


def create_support_ticket(customer_id: str, category: str, summary: str, priority: str = "normal") -> dict:
    """Open a support ticket for follow-up."""
    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": str(customer_id),
        "category": category,
        "summary": summary,
        "priority": priority if priority in {"low", "normal", "high", "urgent"} else "normal",
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _TICKETS_LOCK:
        _TICKETS[ticket_id] = ticket
    return ticket


def escalate_to_human(customer_id: str, reason: str) -> dict:
    """Hand the conversation off to a live agent (creates an urgent ticket)."""
    ticket = create_support_ticket(customer_id, "escalation", reason, priority="urgent")
    return {"escalated": True, "ticket_id": ticket["ticket_id"], "estimated_wait_minutes": 4}


def list_tickets(customer_id: str | None = None) -> list[dict]:
    with _TICKETS_LOCK:
        tickets = list(_TICKETS.values())
    if customer_id:
        tickets = [t for t in tickets if t["customer_id"] == str(customer_id)]
    return tickets


# ---------------------------------------------------------------------------
# Registry + OpenAI schemas
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_customer_profile": get_customer_profile,
    "get_billing_history": get_billing_history,
    "check_network_status": check_network_status,
    "list_plans": list_plans,
    "recommend_plan": recommend_plan,
    "list_devices": list_devices,
    "check_upgrade_eligibility": check_upgrade_eligibility,
    "create_support_ticket": create_support_ticket,
    "escalate_to_human": escalate_to_human,
}


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


_CID = {"customer_id": {"type": "string", "description": "The customer's account ID"}}

TOOL_SCHEMAS: dict[str, dict] = {
    "get_customer_profile": _fn("get_customer_profile", "Look up the customer's account profile.", _CID, ["customer_id"]),
    "get_billing_history": _fn(
        "get_billing_history",
        "Get recent bills and an explanation of the latest bill change.",
        {**_CID, "months": {"type": "integer", "description": "Number of months to return", "default": 3}},
        ["customer_id"],
    ),
    "check_network_status": _fn(
        "check_network_status",
        "Check for known network outages in a ZIP code.",
        {"zip_code": {"type": "string"}},
        ["zip_code"],
    ),
    "list_plans": _fn("list_plans", "List available rate plans with pricing and perks.", {}),
    "recommend_plan": _fn(
        "recommend_plan",
        "Rank rate plans against the customer's needs.",
        {
            "lines": {"type": "integer"},
            "data_usage": {"type": "string", "enum": ["low", "medium", "high"]},
            "travels_internationally": {"type": "boolean"},
            "wants_streaming": {"type": "boolean"},
            "budget_per_line": {"type": "number"},
        },
    ),
    "list_devices": _fn(
        "list_devices",
        "List devices for sale, optionally filtered by brand or max price.",
        {"brand": {"type": "string"}, "max_price": {"type": "number"}},
    ),
    "check_upgrade_eligibility": _fn(
        "check_upgrade_eligibility", "Check whether the customer can upgrade their device.", _CID, ["customer_id"]
    ),
    "create_support_ticket": _fn(
        "create_support_ticket",
        "Open a support ticket for issues that need follow-up.",
        {
            **_CID,
            "category": {"type": "string", "enum": ["billing", "network", "device", "account", "other"]},
            "summary": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
        },
        ["customer_id", "category", "summary"],
    ),
    "escalate_to_human": _fn(
        "escalate_to_human",
        "Transfer the customer to a live human agent. Use when the customer asks for a human, "
        "is very frustrated, or the issue cannot be resolved by the AI.",
        {**_CID, "reason": {"type": "string"}},
        ["customer_id", "reason"],
    ),
}


def schemas_for(tool_names: list[str]) -> list[dict]:
    return [TOOL_SCHEMAS[name] for name in tool_names if name in TOOL_SCHEMAS]


def execute_tool(name: str, arguments: dict | str | None) -> Any:
    """Run a tool by name, returning a JSON-serializable result (errors included)."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"error": "Invalid JSON arguments"}
    try:
        return fn(**(arguments or {}))
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
