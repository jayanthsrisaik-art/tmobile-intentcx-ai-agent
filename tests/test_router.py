import pytest

from src.router import classify, classify_keywords, route_intent


@pytest.mark.parametrize("query,intent,agent", [
    ("Why did my bill go up?", "billing", "support"),
    ("I was overcharged a late fee", "billing", "support"),
    ("I keep getting dropped calls", "technical_support", "support"),
    ("Is there a network outage near me?", "technical_support", "support"),
    ("What plans do you have for a family of four?", "sales_plan", "sales"),
    ("I want to upgrade to a new phone", "sales_device", "sales"),
    ("Can I trade in my Pixel for an iPhone?", "sales_device", "sales"),
    ("I want to speak to a human", "escalation", "escalation"),
    ("How do I reset my account password?", "account", "support"),
])
def test_keyword_routing(query, intent, agent):
    d = classify_keywords(query)
    assert d.intent == intent
    assert d.agent == agent
    assert 0 < d.confidence <= 1


def test_unknown_query_defaults_to_general_support():
    d = classify_keywords("hello there")
    assert d.intent == "general" and d.agent == "support"
    assert d.confidence < 0.55


def test_backward_compatible_route_intent():
    assert route_intent("bill problem") == "support"
    assert route_intent("buy a new plan") == "sales"


class _FakeLLM:
    def __init__(self, content):
        self.content = content
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        msg = type("M", (), {"content": self.content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_llm_fallback_used_when_low_confidence():
    d = classify("hello there", client=_FakeLLM('{"intent": "sales_plan", "confidence": 0.9}'), model="x")
    assert d.method == "llm" and d.agent == "sales"


def test_llm_fallback_ignores_bad_output():
    d = classify("hello there", client=_FakeLLM("not json"), model="x")
    assert d.method == "keyword" and d.intent == "general"


def test_llm_not_called_when_confident():
    d = classify("Why did my bill go up?", client=_FakeLLM('{"intent": "sales_plan"}'), model="x")
    assert d.intent == "billing" and d.method == "keyword"
