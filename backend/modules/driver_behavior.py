import random

class DriverBehaviorAnalyzer:

    def compute_penalty(self):

        acceleration_score = random.uniform(0, 1)

        if acceleration_score > 0.7:
            penalty = 0.12
            style = "aggressive"

        elif acceleration_score > 0.4:
            penalty = 0.06
            style = "normal"

        else:
            penalty = 0.02
            style = "efficient"

        return {
            "driver_style": style,
            "driver_penalty_percent": penalty
        }