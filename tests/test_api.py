from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    assert client.get("/").json()["mode"] == "offline"


def test_chat_json_body_and_session_flow():
    r = client.post("/chat", json={"query": "Why did my bill go up?", "customer_id": "123"})
    assert r.status_code == 200
    body = r.json()
    assert body["agent_type"] == "support" and body["intent"] == "billing"
    sid = body["session_id"]
    r2 = client.post("/chat", json={"query": "any phone upgrades?", "customer_id": "123", "session_id": sid})
    assert r2.json()["agent_type"] == "sales"
    s = client.get(f"/sessions/{sid}").json()
    assert len(s["messages"]) == 4
    assert client.delete(f"/sessions/{sid}").status_code == 200
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_chat_legacy_query_param():
    r = client.post("/chat", params={"query": "network problem"})
    assert r.status_code == 200 and r.json()["agent_type"] == "support"


def test_chat_validation_and_unknown_customer():
    assert client.post("/chat").status_code == 422
    assert client.post("/chat", json={"query": "hi", "customer_id": "999"}).status_code == 404


def test_route_customers_tickets_metrics_ui():
    assert client.post("/route", json={"query": "upgrade my phone"}).json()["agent"] == "sales"
    assert client.get("/customers/456").json()["name"] == "Priya"
    client.post("/chat", json={"query": "I want a real person now", "customer_id": "789"})
    assert any(t["category"] == "escalation" for t in client.get("/tickets", params={"customer_id": "789"}).json()["tickets"])
    assert client.get("/metrics").json()["requests"] >= 1
    assert "IntentCX" in client.get("/ui").text
