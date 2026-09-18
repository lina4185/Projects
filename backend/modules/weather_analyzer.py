# import requests

# class WeatherAnalyzer:

#     def __init__(self):
#         self.api_key = "YOUR_API_KEY"

#     def fetch_weather(self, lat=35.6762, lon=139.6503):

#         url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"

#         response = requests.get(url)

#         return response.json()

#     def compute_penalty(self):

#         data = self.fetch_weather()

#         temp = data["main"]["temp"]
#         wind = data["wind"]["speed"]

#         penalty = 0

#         if temp < 10:
#             penalty += 0.12

#         if wind > 8:
#             penalty += 0.08

#         return {
#             "weather_penalty": penalty,
#             "temperature": temp,
#             "wind_speed": wind
#         }

class WeatherAnalyzer:

    def compute_penalty(self):

        return {
            "temperature_c": 10,
            "wind_speed_kmh": 28,
            "weather_penalty_percent": 11,
            "explanation":
                "Cold weather and headwind increasing energy consumption."
        }