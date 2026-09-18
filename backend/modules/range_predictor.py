# class RangePredictor:

#     def predict(self,
#                 weather,
#                 load,
#                 road,
#                 battery,
#                 driver):

#         base_range = 320

#         total_penalty = (
#             weather["weather_penalty_percent"] +
#             load["load_penalty"] +
#             road["road_penalty"] +
#             driver["driver_penalty"]
#         )

#         adjusted_range = base_range * (1 - total_penalty)

#         adjusted_range *= battery["battery_health"]

#         confidence = 0.92 - total_penalty

#         return {
#             "estimated_range_km": round(adjusted_range, 2),
#             "confidence": round(confidence, 2),
#             "explanation": "Weather and traffic conditions reducing range"
#         }

class RangePredictor:

    def predict(self,
                weather,
                load,
                road,
                battery,
                driver):

        base_range = 320

        total_penalty = (
            weather["weather_penalty_percent"]
            + load["load_penalty"]
            + road["road_penalty_percent"]
            + driver["driver_penalty_percent"]
        )

        adjusted_range = (
            base_range * (1 - total_penalty / 100)
        )

        adjusted_range *= battery["battery_health"]

        confidence = max(
            0.5,
            1 - (total_penalty / 100)
        )

        return {
            "estimated_range_km":
                round(adjusted_range, 2),

            "confidence":
                round(confidence, 2),

            "explanation":
                "Weather and traffic conditions reducing range"
        }