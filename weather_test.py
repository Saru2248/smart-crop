import requests

def get_weather_data(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        rainfall = data.get('rain', {}).get('1h', 0)
        print(f"✅ Weather data for {city}:")
        print(f"🌡️ Temperature: {temperature}°C")
        print(f"💧 Humidity: {humidity}%")
        print(f"☔ Rainfall: {rainfall} mm (last hour)")
    else:
        print("❌ Error fetching data. Check city name or API key.")
        print(f"Status Code: {response.status_code}")
        print("Response:", data)

# 🧠 Example usage
api_key = "9d08ae621065c5763506dd2a2472c7b5"  # your OpenWeatherMap API key
city_name = "Aurangabad"  # change this to test different cities
get_weather_data(city_name, api_key)
