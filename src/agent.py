from openai import OpenAI
from src.prompts import SUPPORT_PROMPT, SALES_PROMPT
from src.tools import get_customer_profile

client = OpenAI()

def run_agent(query: str, agent_type: str, customer_id="123"):
    customer_context = get_customer_profile(customer_id)

    system_prompt = SUPPORT_PROMPT if agent_type == "support" else SALES_PROMPT

    full_prompt = f"""
Customer Profile:
{customer_context}

User Query:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]
    )

    return response.choices[0].message.content
