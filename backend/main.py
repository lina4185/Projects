from fastapi import FastAPI
from modules.weather_analyzer import WeatherAnalyzer
from modules.load_analyzer import LoadAnalyzer
from modules.road_conditions import RoadConditionsAnalyzer
from modules.battery_health import BatteryHealthMonitor
from modules.driver_behavior import DriverBehaviorAnalyzer
from modules.range_predictor import RangePredictor

app = FastAPI()

weather = WeatherAnalyzer()
load_module = LoadAnalyzer()
road = RoadConditionsAnalyzer()
battery = BatteryHealthMonitor()
driver = DriverBehaviorAnalyzer()
predictor = RangePredictor()

@app.get("/predict")
def predict_range():

    weather_penalty = weather.compute_penalty()
    load_penalty = load_module.compute_penalty(3, 40)
    road_penalty = road.compute_penalty()
    battery_score = battery.compute_health()
    driver_penalty = driver.compute_penalty()

    result = predictor.predict(
        weather_penalty,
        load_penalty,
        road_penalty,
        battery_score,
        driver_penalty
    )

    return result