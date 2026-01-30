import json

def get_customer_profile(customer_id: str):
    """
    Simulated API call (replace with real API integration).
    """
    with open("data/sample_customer_data.json") as f:
        customers = json.load(f)

    return customers.get(customer_id, {})
