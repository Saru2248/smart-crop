# weather_backend.py
import requests

API_KEY = "79a13de45233d9a4271f"  # your API key

def get_weather_data(city_name):
    """
    Fetches live weather data from OpenWeather API.
    Input: city_name (string)
    Output: dictionary with temperature, humidity, condition, rainfall
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name},IN&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            pressure = data["main"]["pressure"]
            rainfall = data.get("rain", {}).get("1h", 0)
            condition = data["weather"][0]["description"]

            return {
                "Temperature (°C)": temp,
                "Humidity (%)": humidity,
                "Pressure (hPa)": pressure,
                "Rainfall (mm)": rainfall,
                "Condition": condition.title()
            }
        else:
            return {"Error": "City not found or invalid API key"}
    except Exception as e:
        return {"Error": str(e)}
