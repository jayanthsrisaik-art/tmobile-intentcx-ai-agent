from src import tools


def test_profile_hides_billing_history():
    p = tools.get_customer_profile("123")
    assert p["name"] == "John" and "billing_history" not in p
    assert tools.get_customer_profile("nope") == {}


def test_billing_change_explained():
    up = tools.get_billing_history("123")["latest_change"]
    assert up["delta"] == 25.0 and "Promo" in up["explanation"]
    down = tools.get_billing_history("456")["latest_change"]
    assert down["delta"] == -5.0 and "Late payment" in down["explanation"]


def test_network_status_known_and_default():
    assert tools.check_network_status("10001")["status"] == "degraded"
    assert tools.check_network_status("00000")["status"] == "normal"


def test_recommend_plan_traveler_streamer():
    rec = tools.recommend_plan(lines=2, data_usage="high", travels_internationally=True, wants_streaming=True)
    assert rec["top_pick"] == "Magenta Max"


def test_recommend_plan_budget():
    rec = tools.recommend_plan(lines=1, data_usage="low", budget_per_line=60)
    assert rec["top_pick"] == "Essentials"


def test_device_filters():
    devices = tools.list_devices(brand="apple", max_price=800)["devices"]
    assert [d["name"] for d in devices] == ["iPhone 16"]


def test_tickets_and_escalation():
    t = tools.create_support_ticket("789", "billing", "question", priority="bogus")
    assert t["priority"] == "normal" and t["status"] == "open"
    e = tools.escalate_to_human("789", "wants human")
    assert e["escalated"]
    ids = {x["ticket_id"] for x in tools.list_tickets("789")}
    assert {t["ticket_id"], e["ticket_id"]} <= ids


def test_execute_tool_errors_are_returned_not_raised():
    assert "error" in tools.execute_tool("does_not_exist", {})
    assert "error" in tools.execute_tool("list_devices", "{bad json")
    assert "error" in tools.execute_tool("list_devices", {"unknown_arg": 1})
