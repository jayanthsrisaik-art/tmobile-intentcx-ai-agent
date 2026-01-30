def route_intent(user_query: str):
    query = user_query.lower()

    if "bill" in query or "problem" in query or "network" in query:
        return "support"

    if "plan" in query or "upgrade" in query or "buy" in query:
        return "sales"

    return "support"
