from fastapi import FastAPI
from src.router import route_intent
from src.agent import run_agent

app = FastAPI()

@app.get("/")
def health():
    return {"status": "IntentCX AI Agent running"}

@app.post("/chat")
def chat(query: str):
    agent_type = route_intent(query)
    response = run_agent(query, agent_type)

    return {
        "agent_type": agent_type,
        "user_query": query,
        "response": response
    }

