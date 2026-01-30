from src.router import route_intent
from src.agent import run_agent

print("\n=== T-Mobile IntentCX AI Agent Demo ===\n")

while True:
    query = input("Customer: ")

    if query.lower() in ["exit", "quit"]:
        break

    agent_type = route_intent(query)

    print(f"\n[Routing -> {agent_type.upper()} AGENT]\n")

    answer = run_agent(query, agent_type)

    print("Agent Response:\n", answer)
    print("\n--------------------------------------\n")
