# T-Mobile IntentCX AI Agent (Customer Support + Sales)

This project implements an OpenAI-powered AI Agent inspired by T-Mobile’s IntentCX initiative.

It demonstrates how LLM-based agents can improve customer support and sales workflows through intent routing and API-grounded responses.

---

## Features

- GPT-powered conversational agent for Support + Sales use cases  
- Intent routing (Customer Support vs Sales Assistant)  
- Customer profile grounding via API-style tool integration  
- Designed for extension into multi-agent + voice workflows  

---

##  Architecture

1. User Query  
2. Intent Router (`router.py`)  
3. Customer Context Tool (`tools.py`)  
4. LLM Agent Execution (`agent.py`)  
5. Grounded Response Returned  

---

## Run the Demo

```bash
pip install -r requirements.txt
python demo.py

### Environment Setup

Set your OpenAI API key before running:

```bash
export OPENAI_API_KEY="your_key_here"
