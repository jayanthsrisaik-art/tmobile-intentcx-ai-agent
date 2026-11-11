"""
Lightweight guardrails applied before any text reaches the LLM.

- PII redaction: card numbers, SSNs, emails are masked so they never leave the system.
- Frustration detection: a simple lexical sentiment signal used to trigger
  proactive escalation to a human agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PII_PATTERNS = [
    ("CARD_NUMBER", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
]

_NEGATIVE = {
    "angry": 2, "furious": 3, "frustrated": 2, "frustrating": 2, "terrible": 2, "awful": 2,
    "worst": 3, "ridiculous": 2, "unacceptable": 3, "hate": 2, "useless": 2, "scam": 3,
    "again": 1, "still": 1, "never": 1, "cancel": 2, "sick of": 3, "fed up": 3, "third time": 3,
}
_POSITIVE = {"thanks": 1, "thank you": 2, "great": 1, "awesome": 2, "love": 1, "perfect": 1, "helpful": 1}

FRUSTRATION_ESCALATION_THRESHOLD = 0.7


@dataclass
class GuardrailResult:
    text: str
    redactions: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    frustration: float = 0.0

    @property
    def should_escalate(self) -> bool:
        return self.frustration >= FRUSTRATION_ESCALATION_THRESHOLD


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[REDACTED_{label}]", text)
    return text, found


def _count(lexicon: dict[str, int], lowered: str) -> int:
    return sum(w for term, w in lexicon.items() if re.search(rf"\b{re.escape(term)}\b", lowered))


def score_sentiment(text: str) -> tuple[str, float]:
    lowered = text.lower()
    neg = _count(_NEGATIVE, lowered)
    pos = _count(_POSITIVE, lowered)
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        neg += 2  # SHOUTING
    neg += min(text.count("!"), 3) // 2
    frustration = round(min(1.0, max(0.0, (neg - pos) / 6)), 2)
    if frustration >= 0.3:
        label = "negative"
    elif pos > neg:
        label = "positive"
    else:
        label = "neutral"
    return label, frustration


def apply_guardrails(text: str) -> GuardrailResult:
    clean, redactions = redact_pii(text)
    sentiment, frustration = score_sentiment(text)
    return GuardrailResult(clean, redactions, sentiment, frustration)
