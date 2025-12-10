# market_platform.py
market_prices = {
    "wheat": 25,  # price per kg
    "rice": 40,
    "tomato": 30
}

def place_order(produce, qty):
    if produce.lower() in market_prices:
        price = market_prices[produce.lower()] * qty
        return f"Order placed for {qty}kg {produce}. Total price: ₹{price}"
    else:
        return "Produce not available in the market"

# Example Usage
print(place_order("wheat", 10))
