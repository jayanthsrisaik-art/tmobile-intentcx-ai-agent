import json
from types import SimpleNamespace as NS

from src.agent import IntentCXAgent
from src.config import Settings


def offline_agent():
    return IntentCXAgent(settings=Settings(offline=True))


def test_offline_billing_is_grounded():
    r = offline_agent().handle("Why did my bill go up?", customer_id="123")
    assert r.mode == "offline" and r.agent_type == "support"
    assert "$25.00" in r.response
    assert [t.name for t in r.tool_calls] == ["get_customer_profile", "get_billing_history"]


def test_offline_outage_detected():
    r = offline_agent().handle("I have no service and dropped calls", customer_id="456")
    assert "Tower maintenance" in r.response


def test_multi_turn_session_memory():
    agent = offline_agent()
    r1 = agent.handle("Why did my bill go up?", customer_id="123")
    r2 = agent.handle("Can I upgrade my phone?", session_id=r1.session_id)
    assert r1.session_id == r2.session_id and r2.customer_id == "123"
    s = agent.sessions.get(r1.session_id)
    assert len(s.messages) == 4 and s.intents == ["billing", "sales_device"]


def test_frustration_escalates_proactively():
    r = offline_agent().handle("I am so frustrated and fed up, your service is the worst!!", customer_id="123")
    assert r.agent_type == "escalation" and r.escalated


def test_pii_never_reaches_agent_history():
    agent = offline_agent()
    r = agent.handle("My bill is wrong, card 4111 1111 1111 1111", customer_id="123")
    assert r.redactions == ["CARD_NUMBER"]
    assert "4111" not in json.dumps(agent.sessions.get(r.session_id).messages)


def test_metrics():
    agent = offline_agent()
    agent.handle("bill", customer_id="123")
    agent.handle("I want to talk to a human", customer_id="123")
    m = agent.get_metrics()
    assert m["requests"] == 2 and m["escalations"] == 1 and m["escalation_rate"] == 0.5


# ---- LLM tool-calling loop, exercised with a scripted fake OpenAI client ----

class FakeOpenAI:
    """Returns a tool call on the first request, then a final answer."""

    def __init__(self, tool_name, args):
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))
        self._script = [
            NS(content=None, tool_calls=[NS(id="call_1", function=NS(name=tool_name, arguments=json.dumps(args)))]),
            NS(content="Your bill rose $25 because a promo credit expired.", tool_calls=None),
        ]

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if "tools" not in kwargs:  # router LLM call
            return NS(choices=[NS(message=NS(content='{"intent": "billing", "confidence": 0.9}'))])
        return NS(choices=[NS(message=self._script.pop(0))])


def test_llm_tool_calling_loop():
    fake = FakeOpenAI("get_billing_history", {"customer_id": "123"})
    agent = IntentCXAgent(settings=Settings(offline=False, llm_routing=False), client=fake)
    r = agent.handle("Why did my bill go up?", customer_id="123")
    assert r.mode == "llm" and r.response.startswith("Your bill rose")
    assert r.tool_calls[0].name == "get_billing_history"
    assert r.tool_calls[0].result["latest_change"]["delta"] == 25.0
    # Second request must include the tool result message.
    assert any(m.get("role") == "tool" for m in fake.requests[1]["messages"])
    # Support agent is only offered support tools.
    offered = {t["function"]["name"] for t in fake.requests[0]["tools"]}
    assert "recommend_plan" not in offered and "get_billing_history" in offered


def test_llm_cannot_call_tools_outside_its_scope():
    fake = FakeOpenAI("recommend_plan", {})
    agent = IntentCXAgent(settings=Settings(offline=False, llm_routing=False), client=fake)
    r = agent.handle("Why did my bill go up?", customer_id="123")
    assert "not permitted" in r.tool_calls[0].result["error"]
