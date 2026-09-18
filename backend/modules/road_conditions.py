import random

class RoadConditionsAnalyzer:

    def compute_penalty(self):

        traffic_level = random.choice([
            "smooth",
            "moderate",
            "stop_and_go"
        ])

        if traffic_level == "smooth":
            penalty = 0.02

        elif traffic_level == "moderate":
            penalty = 0.07

        else:
            penalty = 0.15

        return {
            "traffic": traffic_level,
            "road_penalty_percent": penalty
        }