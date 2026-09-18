class ChargingStationManager:

    def recommend_station(self):

        stations = [
            {
                "name": "Tesla Supercharger A",
                "distance_km": 8,
                "wait_time_min": 5
            },
            {
                "name": "Tesla Supercharger B",
                "distance_km": 15,
                "wait_time_min": 0
            }
        ]

        best_station = min(
            stations,
            key=lambda x: x["distance_km"] + x["wait_time_min"]
        )

        return best_station