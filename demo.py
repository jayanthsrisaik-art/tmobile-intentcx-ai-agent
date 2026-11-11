"""Interactive CLI demo. Works with or without OPENAI_API_KEY (offline mode)."""

import argparse

from src.agent import IntentCXAgent

SCRIPTED = [
    "Why did my bill go up this month?",
    "I keep getting dropped calls at home",
    "I want to upgrade to a new iPhone, any trade-in deals?",
    "What's the best plan for my family? We travel internationally and love Netflix",
    "This is ridiculous, I want to speak to a real person!",
]


def show(agent: IntentCXAgent, query: str, customer_id: str, session_id: str | None) -> str:
    r = agent.handle(query, customer_id=customer_id, session_id=session_id)
    tools = ", ".join(tc.name for tc in r.tool_calls) or "none"
    print(f"\n[Routing -> {r.agent_type.upper()} AGENT | intent={r.intent} "
          f"conf={r.confidence:.0%} via {r.routing_method} | sentiment={r.sentiment}]")
    print(f"[Tools: {tools}]" + (" [ESCALATED]" if r.escalated else ""))
    print("Agent Response:\n", r.response)
    print("\n--------------------------------------")
    return r.session_id


def main():
    parser = argparse.ArgumentParser(description="T-Mobile IntentCX AI Agent demo")
    parser.add_argument("--customer", default="123", help="Customer ID (123, 456, 789)")
    parser.add_argument("--scripted", action="store_true", help="Run a scripted showcase conversation")
    args = parser.parse_args()

    agent = IntentCXAgent()
    print(f"\n=== T-Mobile IntentCX AI Agent Demo ({agent.mode} mode, customer {args.customer}) ===")

    session_id = None
    if args.scripted:
        for q in SCRIPTED:
            print(f"\nCustomer: {q}")
            session_id = show(agent, q, args.customer, session_id)
        print("\nMetrics:", agent.get_metrics())
        return

    print("Type 'exit' to quit.\n")
    while True:
        try:
            query = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query:
            session_id = show(agent, query, args.customer, session_id)


if __name__ == "__main__":
    main()
