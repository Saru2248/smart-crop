# storage_monitor.py
def check_storage(temp, humidity):
    if temp > 30 or humidity > 70:
        return "⚠️ Warning: Risk of spoilage! Adjust storage conditions."
    else:
        return "✅ Storage conditions are optimal."

# Example Usage
temperature = 32
humidity = 75
print(check_storage(temperature, humidity))
