"""
Intent router.

Two-stage routing:
1. A fast, deterministic weighted-keyword classifier (no API call, ~microseconds).
2. Optional LLM classification when keyword confidence is low.

Fine-grained intents are mapped onto specialist agents (support / sales / escalation).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

# intent -> {pattern: weight}
INTENT_KEYWORDS: dict[str, dict[str, float]] = {
    "billing": {
        r"bill(s|ing|ed)?": 3, r"charge[ds]?": 3, r"payment": 2, r"pay": 1, r"refund": 3,
        r"overcharge[ds]?": 4, r"fee[s]?": 2, r"autopay": 3, r"invoice": 3, r"credit": 1,
        r"(went|gone) up": 2, r"increase[ds]?": 2, r"expensive": 1,
    },
    "technical_support": {
        r"network": 3, r"signal": 3, r"coverage": 3, r"outage": 4, r"dropp?(ed|ing)? calls?": 4,
        r"no service": 4, r"slow (data|internet)": 3, r"not working": 3, r"broken": 2, r"problem": 1,
        r"issue": 1, r"wi-?fi calling": 3, r"esim": 2, r"sim": 1, r"reset": 2, r"hotspot not": 3,
        r"can'?t (connect|call|text)": 4, r"5g": 1,
    },
    "account": {
        r"password": 3, r"pin": 2, r"login|log in|sign in": 3, r"address": 2, r"cancel": 3,
        r"port(ing)? (out|my number)": 3, r"add (a )?line": 2, r"remove (a )?line": 3, r"account": 1,
    },
    "sales_plan": {
        r"plans?": 3, r"unlimited": 2, r"cheaper": 2, r"switch": 2, r"family": 2, r"lines": 1,
        r"international": 2, r"travel(ing|ling)?": 2, r"netflix|streaming": 2, r"deal[s]?": 2,
        r"promo(tion)?s?": 2, r"save money": 2,
    },
    "sales_device": {
        r"upgrade": 4, r"new phone": 4, r"buy": 3, r"iphone": 3, r"galaxy|samsung": 3, r"pixel": 3,
        r"trade[- ]?in": 4, r"device": 2, r"phone": 1, r"tablet|watch": 2,
    },
    "escalation": {
        r"(speak|talk) (to|with) (a )?(human|person|someone|agent|representative|manager)": 8,
        r"real person": 8, r"supervisor": 6, r"manager": 4, r"complaint": 3, r"ridiculous": 2,
        r"unacceptable": 3, r"lawyer|sue|fcc": 6,
    },
}

INTENT_TO_AGENT = {
    "billing": "support",
    "technical_support": "support",
    "account": "support",
    "sales_plan": "sales",
    "sales_device": "sales",
    "escalation": "escalation",
    "general": "support",
}

_COMPILED = {
    intent: [(re.compile(rf"\b(?:{pat})\b", re.IGNORECASE), w) for pat, w in kws.items()]
    for intent, kws in INTENT_KEYWORDS.items()
}

LOW_CONFIDENCE = 0.55


@dataclass
class RoutingDecision:
    intent: str
    agent: str
    confidence: float
    method: str = "keyword"
    scores: dict[str, float] = field(default_factory=dict)
    matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def classify_keywords(user_query: str) -> RoutingDecision:
    scores: dict[str, float] = {}
    matched: list[str] = []
    for intent, patterns in _COMPILED.items():
        total = 0.0
        for regex, weight in patterns:
            for m in regex.finditer(user_query):
                total += weight
                matched.append(m.group(0).lower())
        if total:
            scores[intent] = total

    if not scores:
        return RoutingDecision("general", INTENT_TO_AGENT["general"], 0.3, scores={}, matched=[])

    # Escalation requests always win when clearly signalled.
    if scores.get("escalation", 0) >= 6:
        best = "escalation"
    else:
        best = max(scores, key=scores.get)
    total = sum(scores.values())
    # Confidence blends dominance over other intents with absolute evidence strength.
    dominance = scores[best] / total
    strength = min(1.0, scores[best] / 6)
    confidence = round(0.6 * dominance + 0.4 * strength, 2)
    return RoutingDecision(best, INTENT_TO_AGENT[best], confidence, scores=scores, matched=sorted(set(matched)))


_LLM_ROUTER_PROMPT = (
    "Classify the customer's message for a wireless carrier into exactly one intent: "
    + ", ".join(INTENT_TO_AGENT)
    + '. Reply with JSON only: {"intent": "<intent>", "confidence": <0-1>}.'
)


def classify_llm(user_query: str, client, model: str) -> RoutingDecision | None:
    """Ask the LLM to classify intent. Returns None on any failure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_ROUTER_PROMPT},
                {"role": "user", "content": user_query},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        intent = data.get("intent")
        if intent not in INTENT_TO_AGENT:
            return None
        return RoutingDecision(intent, INTENT_TO_AGENT[intent], float(data.get("confidence", 0.7)), method="llm")
    except Exception:
        return None


def classify(user_query: str, client=None, model: str | None = None) -> RoutingDecision:
    """Keyword routing first; fall back to the LLM if confidence is low and a client is given."""
    decision = classify_keywords(user_query)
    if decision.confidence < LOW_CONFIDENCE and client is not None and model:
        llm_decision = classify_llm(user_query, client, model)
        if llm_decision is not None:
            llm_decision.scores = decision.scores
            llm_decision.matched = decision.matched
            return llm_decision
    return decision


def route_intent(user_query: str) -> str:
    """Backward-compatible helper: returns the agent name ('support', 'sales' or 'escalation')."""
    return classify_keywords(user_query).agent
